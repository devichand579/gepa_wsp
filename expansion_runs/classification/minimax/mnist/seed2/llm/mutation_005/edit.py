def edit(w, f):
    import numpy as np

    # ---------------- data ----------------
    X = f["x"].reshape(-1, 784).astype(np.float64)
    y = f["y"].astype(np.int64)
    N, D = X.shape
    Y = np.zeros((N, 10))
    Y[np.arange(N), y] = 1.0
    mu = X.mean(axis=0)
    Xc = X - mu

    # ---------------- PCA (closed form) ----------------
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = 128
    V = Vt[:k]                       # k x D  (row space)
    Z = Xc @ V.T                     # N x k
    ev = (S[:k] ** 2) / (N - 1.0)    # eigenvalues

    # ---------------- LDA directions in PCA space ----------------
    m = np.zeros((10, k))
    for c in range(10):
        m[c] = Z[y == c].mean(axis=0)
    Sw = np.zeros((k, k))
    for c in range(10):
        Zc = Z[y == c] - m[c]
        Sw += Zc.T @ Zc
    Sw /= N
    a_shrink = 0.15
    Sw_s = (1.0 - a_shrink) * Sw + a_shrink * (np.trace(Sw) / k) * np.eye(k)
    d = np.linalg.solve(Sw_s, (m - m.mean(axis=0)).T).T     # 10 x k
    # scale each LDA direction so its pre-activation has unit std on train data
    for c in range(10):
        s = Z @ d[c]
        sd = s.std()
        if sd > 1e-9:
            d[c] = d[c] / sd

    # ---------------- candidate hidden-layer designs ----------------
    rng = np.random.default_rng(12345)

    def rand_dirs(n, alpha, seed):
        r = np.random.default_rng(seed)
        sc = (ev + 1e-3) ** (-alpha)
        R = r.standard_normal((n, k)) * sc[None, :]
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        return R

    def build(n_lda, n_rand, alpha, seed, bias_scale):
        rows = []
        if n_lda > 0:
            rows.append(d[:n_lda])
        if n_rand > 0:
            rows.append(rand_dirs(n_rand, alpha, seed))
        Wz = np.concatenate(rows, axis=0)          # H x k  (in PCA space)
        W1 = Wz @ V                                  # H x D  (pixel space)
        b1 = rng.normal(0.0, bias_scale, size=W1.shape[0])
        return W1, b1

    def features(W1, b1, Xm):
        return np.maximum(Xm @ W1.T + b1[None, :], 0.0)

    # ---------------- closed-form ridge with LOO selection ----------------
    def ridge_fit(H, Yt, lams):
        Hc = H - H.mean(axis=0, keepdims=True)
        G = Hc.T @ Hc
        C = Hc.T @ Yt
        n = Hc.shape[0]
        best = None
        for lam in lams:
            A = G + lam * np.eye(G.shape[0])
            try:
                Wt = np.linalg.solve(A, C)
            except np.linalg.LinAlgError:
                continue
            # closed-form leave-one-out residual via hat matrix
            try:
                P = Hc @ np.linalg.solve(A, Hc.T)
            except np.linalg.LinAlgError:
                continue
            dg = np.clip(np.diag(P), -0.999, 0.999)
            R = Yt - Hc @ Wt
            loo = R / (1.0 - dg)[:, None]
            err = float(np.mean(np.sum(loo ** 2, axis=1)))
            if best is None or err < best[0]:
                best = (err, lam, Wt)
        err, lam, Wt = best
        mu_h = H.mean(axis=0)
        W2 = Wt.T                                   # 10 x H
        b2 = Yt.mean(axis=0) - W2 @ mu_h            # intercept
        return W2, b2, lam, err

    def evaluate(W1, b1, tr, va):
        Htr = features(W1, b1, X[tr])
        Hva = features(W1, b1, X[va])
        lams = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0]
        W2, b2, lam, err = ridge_fit(Htr, Y[tr], lams)
        logits = Hva @ W2.T + b2[None, :]
        acc = float(np.mean(logits.argmax(axis=1) == y[va]))
        return acc, W2, b2, lam, err

    # deterministic train/validation split for design selection
    perm = np.random.default_rng(7).permutation(N)
    tr, va = perm[:1600], perm[1600:]

    designs = [
        dict(n_lda=10, n_rand=118, alpha=0.35, seed=11, bias_scale=0.5),
        dict(n_lda=10, n_rand=118, alpha=0.60, seed=12, bias_scale=0.5),
        dict(n_lda=10, n_rand=118, alpha=0.15, seed=13, bias_scale=0.5),
        dict(n_lda=10, n_rand=118, alpha=0.35, seed=14, bias_scale=1.0),
        dict(n_lda=10, n_rand=118, alpha=0.35, seed=15, bias_scale=0.2),
        dict(n_lda=0,  n_rand=128, alpha=0.35, seed=16, bias_scale=0.5),
    ]

    best = None
    for des in designs:
        W1, b1 = build(des["n_lda"], des["n_rand"], des["alpha"], des["seed"], des["bias_scale"])
        acc, W2, b2, lam, err = evaluate(W1, b1, tr, va)
        if best is None or acc > best[0]:
            best = (acc, des, W1, b1, W2, b2, lam, err)

    acc_val, des, W1, b1, W2, b2, lam, err = best

    # ---------------- refit on the full pool with the chosen design ----------------
    Hfull = features(W1, b1, X)
    lams = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0]
    W2f, b2f, lamf, errf = ridge_fit(Hfull, Y, lams)

    # ---------------- write back ----------------
    out = {}
    for kk, vv in w.items():
        out[kk] = np.array(vv, dtype=np.float32, copy=True)

    out["fc1.weight"] = np.clip(W1.astype(np.float32), -1000.0, 1000.0)
    out["fc1.bias"] = np.clip(b1.astype(np.float32), -1000.0, 1000.0)
    out["fc2.weight"] = np.clip(W2f.astype(np.float32), -1000.0, 1000.0)
    out["fc2.bias"] = np.clip(b2f.astype(np.float32), -1000.0, 1000.0)

    # ---------------- diagnostics ----------------
    Hf = features(W1, b1, X)
    logits = Hf @ W2f.T + b2f[None, :]
    pred = logits.argmax(axis=1)
    train_acc = float(np.mean(pred == y))
    per_class = [float(np.mean(pred[y == c] == c)) for c in range(10)]
    frac_zero = float(np.mean(Hf == 0.0))
    logit_std = float(logits.std())

    diagnostics = {
        "design": des,
        "val_acc_selected": round(acc_val, 4),
        "train_acc_full": round(train_acc, 4),
        "ridge_lambda": lamf,
        "loo_mse_full": round(errf, 5),
        "per_class_train_acc": [round(v, 3) for v in per_class],
        "fc1_weight_absmax": round(float(np.abs(out["fc1.weight"]).max()), 5),
        "fc1_weight_std": round(float(out["fc1.weight"].std()), 5),
        "fc2_weight_absmax": round(float(np.abs(out["fc2.weight"]).max()), 5),
        "fc2_bias": [round(float(v), 4) for v in out["fc2.bias"]],
        "hidden_zero_fraction": round(frac_zero, 4),
        "hidden_mean": round(float(Hf.mean()), 4),
        "hidden_std": round(float(Hf.std()), 4),
        "logit_std": round(logit_std, 4),
        "pca_top10_var_ratio": round(float(np.sum(ev[:10]) / np.sum(ev)), 4),
        "n_train": int(N),
    }
    return {"weights": out, "diagnostics": diagnostics}
