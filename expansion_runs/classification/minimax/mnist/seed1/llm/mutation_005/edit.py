def edit(w, f):
    import numpy as np

    # ---------------- data ----------------
    X = np.asarray(f["x"], dtype=np.float64).reshape(-1, 784)
    y = np.asarray(f["y"], dtype=np.int64).ravel()
    n, d = X.shape
    C = 10
    Y = np.zeros((n, C), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    xm = X.mean(axis=0)
    Xc = X - xm

    # ---------------- closed-form ridge geometry on pixels ----------------
    G = Xc.T @ Xc
    dg = float(np.trace(G)) / d
    lam0 = 1.0 * dg
    A = G + lam0 * np.eye(d)
    Ainv = np.linalg.inv(A)

    # one-vs-rest ridge directions (class evidence detectors)
    U_ova = Ainv @ (Xc.T @ Y)          # (784, 10)
    # pairwise ridge discriminants
    Yp = Y[:, :, None] - Y[:, None, :]  # Yp[i,j,n] = 1{y=i}-1{y=j}
    iu, ju = np.triu_indices(C, 1)
    Yp = Yp[iu, ju, :]                  # (45, n)
    U_pair = Ainv @ (Xc.T @ Yp.T)       # (784, 45)

    s_ova = Xc @ U_ova                  # (n, 10)
    s_pair = Xc @ U_pair                # (n, 45)

    # ---------------- hidden unit construction ----------------
    rows = []   # list of (w_vec, bias)
    n_ova_thr = 4
    for c in range(C):
        sc = s_ova[:, c]
        for q in (0.35, 0.55, 0.75, 0.90):
            thr = float(np.quantile(sc, q))
            rows.append((U_ova[:, c], -thr))
    for p in range(U_pair.shape[1]):
        sp = s_pair[:, p]
        thr = float(np.quantile(sp, 0.50))
        rows.append((U_pair[:, p], -thr))

    # random projections (generic nonlinear capacity), median threshold
    rng = np.random.default_rng(12345)
    n_rand = 128 - len(rows)
    if n_rand > 0:
        Wr = rng.normal(size=(n_rand, d)) / np.sqrt(d)
        sr = Xc @ Wr.T
        for j in range(n_rand):
            thr = float(np.quantile(sr[:, j], 0.50))
            rows.append((Wr[j], -thr))

    W1 = np.zeros((128, d), dtype=np.float64)
    b1 = np.zeros(128, dtype=np.float64)
    for i, (wv, bv) in enumerate(rows[:128]):
        W1[i] = wv
        b1[i] = bv

    # ---------------- hidden features (recomputed with new fc1) ----------------
    H = np.maximum(Xc @ W1.T + b1, 0.0)
    Hm = H.mean(axis=0)
    Hs = H.std(axis=0)
    Hs[Hs < 1e-9] = 1.0
    Hz = (H - Hm) / Hs

    # ---------------- ridge readout with closed-form LOO selection ----------------
    HtH = Hz.T @ Hz
    HtY = Hz.T @ (Y - 1.0 / C)
    p = HtH.shape[0]
    I = np.eye(p)
    best = None
    for a in (1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0):
        lam = a * p
        Areg = HtH + lam * I
        B = np.linalg.solve(Areg, HtY)
        Hinv = np.linalg.inv(Areg)
        Hd = np.linalg.solve(Areg, Hz.T)          # (p, n)
        hii = np.einsum('ij,ji->i', Hd, Hz.T)     # (n,)
        resid = (Y - 1.0 / C) - Hz @ B
        num = (resid ** 2).sum(axis=1)
        den = np.maximum(1.0 - hii, 1e-8)
        loo = float(np.mean(num / (den ** 2)))
        if best is None or loo < best[0]:
            best = (loo, a, B)
    loo_best, alpha, B = best

    W2 = B.T                                     # (10, 128)
    b2 = (1.0 / C) - Hm @ W2.T                   # (10,)

    # ---------------- write back, preserve names/shapes/dtype ----------------
    w["fc1.weight"] = W1.astype(np.float32)
    w["fc1.bias"] = b1.astype(np.float32)
    w["fc2.weight"] = W2.astype(np.float32)
    w["fc2.bias"] = b2.astype(np.float32)

    # ---------------- diagnostics ----------------
    def acc(W1_, b1_, W2_, b2_, Xa, ya):
        Hh = np.maximum(Xa @ W1_.T + b1_, 0.0)
        L = Hh @ W2_.T + b2_
        return float(np.mean(np.argmax(L, axis=1) == ya))

    tr_acc = acc(W1, b1, W2, b2, X, y)
    Ltr = np.maximum(X @ W1.T + b1, 0.0) @ W2.T + b2
    pmax = np.exp(Ltr - Ltr.max(axis=1, keepdims=True))
    pmax /= pmax.sum(axis=1, keepdims=True)
    ce = float(np.mean(-np.log(np.maximum(pmax[np.arange(n), y], 1e-12))))

    per_class = {}
    for c in range(C):
        m = (y == c)
        per_class[str(c)] = round(float(np.mean(np.argmax(Ltr[m], axis=1) == y[m])), 4)

    diag = {
        "method": "closed-form ridge readout on hand-built ReLU half-space features",
        "hidden_units": {"ova_score_detectors": 40, "pair_discriminants": 45, "random_projections": int(n_rand)},
        "ridge_alpha": alpha,
        "loo_mse_selected": round(loo_best, 6),
        "train_accuracy": round(tr_acc, 4),
        "train_cross_entropy": round(ce, 4),
        "per_class_train_acc": per_class,
        "fc1_weight_absmax": round(float(np.abs(W1).max()), 4),
        "fc2_weight_absmax": round(float(np.abs(W2).max()), 4),
        "hidden_feature_mean": round(float(H.mean()), 5),
        "hidden_feature_std": round(float(H.std()), 5),
        "hidden_active_fraction": round(float((H > 0).mean()), 4),
        "logit_std": round(float(Ltr.std()), 5),
    }
    return {"weights": w, "diagnostics": diag}
