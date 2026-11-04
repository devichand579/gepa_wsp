import numpy as np

def edit(w, f):
    # ---------------- data ----------------
    X = f["x"].reshape(-1, 784).astype(np.float64)      # (2000,784)
    y = np.asarray(f["y"]).astype(np.int64)
    N, D = X.shape
    C = 10

    # ---------------- helpers ----------------
    def softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def acc(logits, yy):
        return float((logits.argmax(1) == yy).mean())

    def onehot(yy):
        Y = np.zeros((len(yy), C))
        Y[np.arange(len(yy)), yy] = 1.0
        return Y

    def ridge_fit(H, T, lam):
        """closed-form ridge with intercept; H (n,d), T (n,k)"""
        Hm = H.mean(0)
        Tm = T.mean(0)
        Hc = H - Hm
        Tc = T - Tm
        G = Hc.T @ Hc
        G.flat[:: G.shape[0] + 1] += lam
        B = np.linalg.solve(G, Hc.T @ Tc)
        b = Tm - Hm @ B
        return B, b

    # ---------------- feature construction ----------------
    # supervised subspace: whitened pixels -> LDA directions
    mu = X.mean(0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / N
    # PCA (eigh of covariance)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 0.0)
    evecs = evecs[:, order]
    # shrink whitening: scale = 1/sqrt(evals + eps)
    eps = 1e-3 * max(evals[0], 1e-12)
    scale = 1.0 / np.sqrt(evals + eps)
    keep = 200
    V = evecs[:, :keep] * scale[:keep]           # (784, keep) whitening map
    Z = Xc @ V                                   # whitened coords

    # class means in whitened space
    Zm = np.zeros((C, Z.shape[1]))
    for c in range(C):
        Zm[c] = Z[y == c].mean(0)
    gmean = Zm.mean(0)
    # within-class scatter in whitened space
    Sw = np.zeros((Z.shape[1], Z.shape[1]))
    for c in range(C):
        Zc = Z[y == c] - Zm[c]
        Sw += Zc.T @ Zc
    Sw /= N
    Sb = np.zeros_like(Sw)
    for c in range(C):
        d = (Zm[c] - gmean).reshape(-1, 1)
        Sb += len(np.unique(y[y == c])) * (d @ d.T)
    Sb /= N
    # regularized LDA: solve Sw_reg w = Sb v  -> eigenproblem
    Sw_reg = Sw + 1e-2 * np.trace(Sw) / Sw.shape[0] * np.eye(Sw.shape[0])
    L = np.linalg.cholesky(Sw_reg)
    Li = np.linalg.inv(L)
    M = Li @ Sb @ Li.T
    M = (M + M.T) / 2.0
    ew, ev = np.linalg.eigh(M)
    order = np.argsort(ew)[::-1]
    ew = ew[order]
    ev = ev[:, order]
    # discriminant directions in whitened space (up to 9 informative)
    n_lda = 9
    Wlda_z = Li.T @ ev[:, :n_lda]              # (keep, n_lda) orthonormal-ish
    # map back to pixel space
    Wlda_x = V @ Wlda_z                        # (784, n_lda)

    # PCA directions of residual (after removing LDA span) in whitened space
    Zres = Z - (Z @ Wlda_z) @ Wlda_z.T
    cov_res = (Zres.T @ Zres) / N
    ev2, evec2 = np.linalg.eigh(cov_res)
    order2 = np.argsort(ev2)[::-1]
    ev2 = np.maximum(ev2[order2], 0.0)
    evec2 = evec2[:, order2]
    n_pca = 128 - n_lda
    Wpca_z = evec2[:, :n_pca]                  # (keep, n_pca)
    Wpca_x = V @ Wpca_z

    # ---------------- candidate 1: supervised (LDA + residual PCA) ----------------
    # ---------------- candidate 2: plain PCA-128 (unsupervised) ----------------
    Wpca_only_x = evecs[:, :128]

    def build(Wdirs, margin_mult=1.5):
        """Wdirs (128,784) row directions -> fc1 weights/bias making units linear on data"""
        W1 = Wdirs.copy()
        proj = X @ W1.T                     # (N,128)
        m = np.abs(proj).max(0)
        b1 = -(margin_mult * m + 1e-6)
        return W1, b1

    def make_features(W1, b1):
        return np.maximum(X @ W1.T + b1, 0.0)

    # readout: ridge on class-mean targets (LDA-equivalent) and ridge on one-hot
    def fit_readout(H, yy, lam, mode="means"):
        if mode == "means":
            T = np.zeros((len(yy), C))
            for c in range(C):
                T[yy == c, c] = 1.0
            # class-mean targets: each sample -> its class mean vector
            cls = np.zeros((C, C))
            for c in range(C):
                cls[c] = T[yy == c].mean(0)
            Tt = cls[yy]
        else:
            Tt = onehot(yy)
        return ridge_fit(H, Tt, lam)

    # ---------------- evaluation split ----------------
    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    n_tr = 1600
    tr, te = perm[:n_tr], perm[n_tr:]

    def score_candidate(Wdirs, mode, lam):
        W1, b1 = build(Wdirs)
        H = make_features(W1, b1)
        B, b = fit_readout(H[tr], y[tr], lam, mode)
        lg = H[te] @ B + b
        return acc(lg, y[te]), (W1, b1, B, b, H)

    cands = {}
    Wsup = np.concatenate([Wlda_x.T, Wpca_x.T], axis=0)[:128]
    Wpca = Wpca_only_x.T[:128]

    best = None
    for name, Wd in (("sup", Wsup), ("pca", Wpca)):
        for mode in ("means", "onehot"):
            for lam in (1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0):
                a, pack = score_candidate(Wd, mode, lam)
                if best is None or a > best[0]:
                    best = (a, name, mode, lam, pack)

    hold_acc, name, mode, lam, (W1, b1, B, b, H) = best

    # ---------------- refit on full pool with selected lambda ----------------
    Bf, bf = fit_readout(H, y, lam, mode)
    train_acc = acc(H @ Bf + bf, y)

    # ---------------- baseline: current weights on pool (sanity) ----------------
    W1c = np.asarray(w["fc1.weight"], dtype=np.float64)
    b1c = np.asarray(w["fc1.bias"], dtype=np.float64)
    W2c = np.asarray(w["fc2.weight"], dtype=np.float64)
    b2c = np.asarray(w["fc2.bias"], dtype=np.float64)
    Hc = np.maximum(X @ W1c.T + b1c, 0.0)
    cur_acc = acc(Hc @ W2c + b2c, y)

    # ---------------- write out ----------------
    newW1 = np.clip(W1, -1000, 1000).astype(np.float32)
    newb1 = np.clip(b1, -1000, 1000).astype(np.float32)
    newW2 = np.clip(Bf, -1000, 1000).astype(np.float32)
    newb2 = np.clip(bf, -1000, 1000).astype(np.float32)

    w["fc1.weight"] = newW1
    w["fc1.bias"] = newb1
    w["fc2.weight"] = newW2
    w["fc2.bias"] = newb2

    # per-class accuracy of new model on pool
    lg = H @ Bf + bf
    pred = lg.argmax(1)
    per_class = {}
    for c in range(C):
        m = (y == c)
        per_class[str(c)] = round(float((pred[m] == c).mean()), 4)

    diagnostics = {
        "selected": {
            "variant": name,
            "readout_mode": mode,
            "ridge_lambda": lam,
            "holdout_acc": round(hold_acc, 4),
            "n_train": int(n_tr),
        },
        "train_acc_in_sample": round(train_acc, 4),
        "current_weights_acc_on_pool": round(cur_acc, 4),
        "per_class_acc": per_class,
        "fc1_weight_abs_max": float(np.abs(newW1).max()),
        "fc1_bias_min": float(newb1.min()),
        "fc1_bias_max": float(newb1.max()),
        "fc2_weight_abs_max": float(np.abs(newW2).max()),
        "feature_mean": float(H.mean()),
        "feature_std": float(H.std()),
        "feature_zero_frac": float((H == 0).mean()),
        "logit_std": float(lg.std()),
        "n_samples": int(N),
    }
    return {"weights": w, "diagnostics": diagnostics}
