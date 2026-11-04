import numpy as np

def edit(w, f):
    # ---------------- data ----------------
    X = f["x"].reshape(-1, 784).astype(np.float64)
    Y = f["y"].astype(np.int64)
    N, D = X.shape
    C = 10
    onehot = np.zeros((N, C)); onehot[np.arange(N), Y] = 1.0

    mu = X.mean(0)
    Xc = X - mu
    cls_mean = np.zeros((C, D))
    for c in range(C):
        m = (Y == c)
        cls_mean[c] = X[m].mean(0)

    # ---------------- regularized within-class covariance ----------------
    Sw = np.zeros((D, D))
    for c in range(C):
        m = (Y == c)
        Xc_c = X[m] - cls_mean[c]
        Sw += Xc_c.T @ Xc_c
    Sw /= max(N - C, 1)
    tr = np.trace(Sw) / D
    Sw_reg = Sw + (0.30 * tr + 1e-6) * np.eye(D)

    # ---------------- whitened PCA basis (top 64 eigvecs of regularized cov) ----------------
    St = Xc.T @ Xc / N
    trt = np.trace(St) / D
    St_reg = St + (0.10 * trt + 1e-6) * np.eye(D)
    evals, evecs = np.linalg.eigh(St_reg)
    order = np.argsort(evals)[::-1][:64]
    Vp = evecs[:, order]                      # D x 64
    lam_p = np.maximum(evals[order], 1e-10)

    # ---------------- class discriminative directions ----------------
    Wd = np.linalg.solve(Sw_reg, (cls_mean - mu).T)   # D x 10
    # ---------------- templates ----------------
    templates = []          # list of (D,) vectors
    for c in range(C):
        templates.append(cls_mean[c])
    for c in range(C):
        v = Wd[:, c]
        nv = np.linalg.norm(v)
        if nv > 1e-12:
            v = v / nv
        for s in (0.6, 1.2):
            templates.append(cls_mean[c] + s * v)
    templates = np.array(templates)             # (64, D)

    # ---------------- build hidden features (128 units) ----------------
    def build_rows(rows, biases):
        pre = X @ rows.T + biases[None, :]
        return np.maximum(pre, 0.0)

    # (a) template units: threshold at 60th percentile of response
    rows_t = templates.copy()
    pre_t = X @ rows_t.T
    thr_t = np.percentile(pre_t, 60, axis=0)
    Ht = np.maximum(pre_t - thr_t[None, :], 0.0)

    # (b) projection units: whitened PCA direction, median threshold
    rows_p = (Vp / np.sqrt(lam_p)[None, :]).T          # 64 x D
    pre_p = X @ rows_p.T
    thr_p = np.percentile(pre_p, 50, axis=0)
    Hp = np.maximum(pre_p - thr_p[None, :], 0.0)

    H = np.concatenate([Ht, Hp], axis=1)                # N x 128
    H = np.concatenate([H, np.ones((N, 1))], axis=1)     # + bias column

    # ---------------- ridge normal equations with closed-form LOO selection ----------------
    T = onehot / np.sqrt(2.0)
    G = H.T @ H
    B = H.T @ T
    n = H.shape[0]
    trG = np.trace(G)

    def solve_ridge(lam):
        A = G + lam * np.eye(G.shape[0])
        try:
            Ai = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            Ai = np.linalg.pinv(A)
        return Ai @ B, Ai

    def loo_err(lam):
        Wl, Ai = solve_ridge(lam)
        # hat diagonal
        Hd = np.einsum('ij,jk,ik->i', H, Ai, H)
        Hd = np.clip(Hd, 0.0, 1.0 - 1e-6)
        R = T - H @ Wl
        Rloo = R / np.maximum(1.0 - Hd, 1e-6)[:, None]
        return float(np.mean(Rloo * Rloo)), Wl

    lams = [1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    best = None
    for lam in lams:
        e, Wl = loo_err(lam)
        if best is None or e < best[0]:
            best = (e, lam, Wl)
    loo_mse, lam_star, W2 = best

    # ---------------- assemble new weights ----------------
    W1_new = np.zeros((128, 784))
    b1_new = np.zeros(128)
    W1_new[:64, :] = rows_t
    b1_new[:64] = -thr_t
    W1_new[64:, :] = rows_p
    b1_new[64:] = -thr_p

    W2_new = W2[:C, :128].T.copy()          # 128 x 10
    b2_new = W2[:C, 128].copy()             # 10

    # scale control: keep logits in a sane range
    logits = H @ W2
    spread = np.std(logits)
    if spread > 0:
        s = min(1.0, 6.0 / spread)
        W2_new *= s
        b2_new *= s

    def f32(a):
        a = np.asarray(a, dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(a, -1000.0, 1000.0)

    out = dict(w)
    out["fc1.weight"] = f32(W1_new)
    out["fc1.bias"] = f32(b1_new)
    out["fc2.weight"] = f32(W2_new)
    out["fc2.bias"] = f32(b2_new)

    # ---------------- diagnostics (in-pool, computed with new weights) ----------------
    Hn = np.maximum(X @ W1_new.T + b1_new[None, :], 0.0)
    L = Hn @ W2_new + b2_new[None, :]
    L = L - L.max(1, keepdims=True)
    P = np.exp(L); P /= P.sum(1, keepdims=True)
    pred = P.argmax(1)
    acc = float((pred == Y).mean())
    ce = float(-np.log(np.maximum(P[np.arange(N), Y], 1e-12)).mean())
    per_class = {int(c): float((pred[Y == c] == c).mean()) for c in range(C)}
    frac_active = float((Hn > 0).mean())

    diagnostics = {
        "method": "supervised template+PCA ReLU features, ridge normal equations, closed-form LOO lambda",
        "lambda_star": float(lam_star),
        "loo_mse": float(loo_mse),
        "in_pool_accuracy": acc,
        "in_pool_cross_entropy": ce,
        "per_class_accuracy": per_class,
        "hidden_active_fraction": frac_active,
        "fc1_weight_absmax": float(np.abs(out["fc1.weight"]).max()),
        "fc2_weight_absmax": float(np.abs(out["fc2.weight"]).max()),
        "logit_std": float(np.std(L)),
        "template_rows": 64,
        "projection_rows": 64,
        "top_eigvals": [float(v) for v in evals[order][:10]],
        "lda_norm": [float(np.linalg.norm(Wd[:, c])) for c in range(C)],
    }
    return {"weights": out, "diagnostics": diagnostics}
