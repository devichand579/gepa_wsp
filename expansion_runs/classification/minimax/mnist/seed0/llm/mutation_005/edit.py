def edit(w, f):
    import numpy as np

    # ---------------- data ----------------
    X = np.asarray(f["x"], dtype=np.float64).reshape(len(f["y"]), -1)   # (N,784)
    y = np.asarray(f["y"]).astype(np.int64)
    N, D = X.shape
    K = 10
    H = 128

    mu = X.mean(axis=0)                       # global mean image
    Xc = X - mu
    onehot = np.zeros((N, K))
    onehot[np.arange(N), y] = 1.0

    # ---------------- within-class scatter (for whitening) ----------------
    Xw = Xc.copy()
    for c in range(K):
        m = (y == c)
        if m.any():
            Xw[m] -= Xc[m].mean(axis=0)
    Sw = (Xw.T @ Xw) / N
    tr = np.trace(Sw) / D
    lam_shrink = 0.15
    Sw_reg = (1.0 - lam_shrink) * Sw + lam_shrink * tr * np.eye(D)

    # eigendecomposition -> whitening basis (descending eigenvalues)
    evals, evecs = np.linalg.eigh(Sw_reg)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 1e-8)
    evecs = evecs[:, order]
    Wh = evecs / np.sqrt(evals)               # (784,784) whitening map

    # ---------------- top-128 directions of whitened total scatter ----------------
    Z = Xc @ Wh                               # (N,784) whitened, centered
    C = (Z.T @ Z) / N
    ew, Vw = np.linalg.eigh(C)
    idx = np.argsort(ew)[::-1][:H]
    Vw = Vw[:, idx]                           # (784,128)
    # project back to pixel space: W = Wh @ Vw  -> rows are directions in pixel space
    W = (Wh @ Vw).T                           # (128,784)

    # ---------------- build fc1 (weights + biases) ----------------
    n_lin = 96                                # always-active (linear) units
    n_nl = H - n_lin                          # rectified (nonlinear) units

    W1 = np.zeros((H, D))
    b1 = np.zeros(H)

    # linear block: bias = +k * std of projection so ReLU never clips
    proj_lin = X @ W[:n_lin].T                # (N,96)
    s_lin = proj_lin.std(axis=0) + 1e-8
    W1[:n_lin] = W[:n_lin]
    b1[:n_lin] = 4.0 * s_lin + 1e-3

    # nonlinear block: rectified projections (zero bias -> genuine nonlinearity)
    W1[n_lin:] = W[n_lin:]
    b1[n_lin:] = 0.0

    # ---------------- forward pass with new fc1 ----------------
    A = X @ W1.T + b1                         # (N,128)
    F = np.maximum(A, 0.0)                    # features

    # ---------------- ridge fit of fc2 (closed form) ----------------
    # targets: +-1 (equivalent to one-hot up to affine map), centered
    T = 2.0 * onehot - 1.0
    T = T - T.mean(axis=0, keepdims=True)

    Fm = F.mean(axis=0)
    Fc = F - Fm
    G = Fc.T @ Fc                             # (128,128)
    B = Fc.T @ T                              # (128,10)

    # closed-form LOO selection of ridge lambda (no iterative optimizer)
    def solve(G, B, lam):
        Areg = G + lam * np.eye(G.shape[0])
        return np.linalg.solve(Areg, B)

    def loo_err(W2):
        # residual r_i = t_i - f_i ; leverage from ridge hat matrix
        pred = Fc @ W2
        R = T - pred
        # leverage: diag of Fc (Fc'Fc+lam I)^-1 Fc'
        try:
            Areg_inv = np.linalg.inv(G + lam * np.eye(G.shape[0]))
        except np.linalg.LinAlgError:
            return 1e18
        Hd = np.einsum('ij,jk,ik->i', Fc, Areg_inv, Fc)
        denom = np.maximum(1.0 - Hd, 1e-6)
        e = R / denom[:, None]
        return float(np.mean(e ** 2))

    lams = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    best_lam, best_err = None, None
    for lam in lams:
        W2 = solve(G, B, lam)
        err = loo_err(W2)
        if best_err is None or err < best_err:
            best_err, best_lam, best_W2 = err, lam, W2

    W2 = best_W2                              # (128,10)
    b2 = T.mean(axis=0) - Fm @ W2              # (10,)

    # ---------------- write back ----------------
    w["fc1.weight"] = np.clip(W1, -1000.0, 1000.0).astype(np.float32)
    w["fc1.bias"] = np.clip(b1, -1000.0, 1000.0).astype(np.float32)
    w["fc2.weight"] = np.clip(W2.T, -1000.0, 1000.0).astype(np.float32)
    w["fc2.bias"] = np.clip(b2, -1000.0, 1000.0).astype(np.float32)

    # ---------------- diagnostics ----------------
    Anew = X @ w["fc1.weight"].astype(np.float64).T + w["fc1.bias"].astype(np.float64)
    Fnew = np.maximum(Anew, 0.0)
    Lnew = Fnew @ w["fc2.weight"].astype(np.float64).T + w["fc2.bias"].astype(np.float64)
    pred = Lnew.argmax(axis=1)
    acc = float((pred == y).mean())
    Ls = Lnew - Lnew.max(axis=1, keepdims=True)
    P = np.exp(Ls)
    P /= P.sum(axis=1, keepdims=True)
    ce = float(-np.log(np.clip(P[np.arange(N), y], 1e-12, None)).mean())
    per_class = [float((pred[y == c] == c).mean()) for c in range(K)]

    diagnostics = {
        "hypothesis": "fc1 = whitened-PCA directions (96 always-active linear + 32 rectified nonlinear); fc2 = ridge normal equations, lambda by closed-form LOO",
        "train_acc_after": round(acc, 4),
        "train_ce_after": round(ce, 4),
        "ridge_lambda": float(best_lam),
        "loo_mse": float(best_err),
        "per_class_acc": [round(v, 3) for v in per_class],
        "fc1_active_frac_linear": round(float((Anew[:, :n_lin] > 0).mean()), 4),
        "fc1_active_frac_nonlin": round(float((Anew[:, n_lin:] > 0).mean()), 4),
        "fc1_w_abs_max": round(float(np.abs(w["fc1.weight"]).max()), 4),
        "fc2_w_abs_max": round(float(np.abs(w["fc2.weight"]).max()), 4),
        "fc1_b_min_max": [round(float(w["fc1.bias"].min()), 4), round(float(w["fc1.bias"].max()), 4)],
        "fc2_b": [round(float(v), 4) for v in w["fc2.bias"]],
        "n_train": int(N),
        "top_eigvals_whitened": [round(float(v), 3) for v in ew[idx][:8]],
    }
    return {"weights": w, "diagnostics": diagnostics}
