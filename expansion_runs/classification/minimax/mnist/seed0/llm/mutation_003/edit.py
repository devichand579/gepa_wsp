def edit(w, f):
    import numpy as np

    X = np.asarray(f["x"], dtype=np.float64).reshape(-1, 784)
    y = np.asarray(f["y"]).astype(np.int64).ravel()
    N = X.shape[0]
    C = 10

    # ---------- class statistics ----------
    mu = np.zeros((C, 784))
    for c in range(C):
        m = (y == c)
        if m.sum() > 0:
            mu[c] = X[m].mean(axis=0)
    mu_bar = mu.mean(axis=0)
    Sw = np.zeros((784, 784))
    for c in range(C):
        m = (y == c)
        if m.sum() > 1:
            D = X[m] - mu[c]
            Sw += D.T @ D
    Sw /= max(N - C, 1)
    tr = np.trace(Sw) / 784.0
    lam = 0.30
    Sw_r = (1.0 - lam) * Sw + lam * tr * np.eye(784)

    # ---------- regularized LDA directions (generalized eigenproblem) ----------
    Sb = (mu - mu_bar).T @ (mu - mu_bar) / C
    try:
        L = np.linalg.cholesky(Sw_r)
        Li = np.linalg.inv(L)
        M = Li @ Sb @ Li.T
        M = 0.5 * (M + M.T)
        evals, evecs = np.linalg.eigh(M)
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]
        U = Li.T @ evecs                      # 784 x 9
        U = U[:, :9]
        # scale so each direction has unit within-class std
        for j in range(U.shape[1]):
            v = U[:, j]
            s = np.sqrt(max(v @ Sw @ v, 1e-12))
            U[:, j] = v / s
    except Exception:
        U = (mu - mu_bar).T.copy()
        for j in range(U.shape[1]):
            v = U[:, j]
            s = np.sqrt(max(v @ Sw @ v, 1e-12))
            U[:, j] = v / s

    # ---------- PCA directions (top 56) ----------
    Xc = X - X.mean(axis=0)
    Ccov = (Xc.T @ Xc) / N
    evals, evecs = np.linalg.eigh(Ccov)
    idx = np.argsort(evals)[::-1]
    P = evecs[:, idx[:56]]
    for j in range(P.shape[1]):
        s = np.sqrt(max(P[:, j] @ Ccov @ P[:, j], 1e-12))
        P[:, j] = P[:, j] / s

    # ---------- build hidden units: hinge expansions + linear-ish units ----------
    W1 = np.zeros((128, 784))
    b1 = np.zeros(128)
    rows = []
    thr_list = []

    ZL = X @ U.T                                  # LDA scores
    qlev = np.array([0.30, 0.45, 0.60, 0.72, 0.82, 0.90, 0.95, 0.985])
    for j in range(U.shape[1]):
        z = ZL[:, j]
        for q in qlev:
            rows.append(U[:, j])
            thr_list.append(np.quantile(z, q))
    for j in range(P.shape[1]):
        rows.append(P[:, j])
        thr_list.append(np.quantile(X @ P[:, j], 0.5))

    W1 = np.asarray(rows, dtype=np.float64)[:128]
    thr = np.asarray(thr_list, dtype=np.float64)[:128]
    b1 = -thr

    # ---------- hidden activations on the full pool ----------
    H = np.maximum(X @ W1.T + b1, 0.0)

    # ---------- ridge readout (closed form) ----------
    Y1 = np.zeros((N, C))
    Y1[np.arange(N), y] = 1.0
    Hm = H.mean(axis=0)
    Ym = Y1.mean(axis=0)
    Hc = H - Hm
    Yc = Y1 - Ym
    G = Hc.T @ Hc
    lam2 = 1e-3 * max(np.trace(G), 1e-12) / 128.0
    A = G + lam2 * np.eye(128)
    B = Hc.T @ Yc
    W2 = np.linalg.solve(A, B).T               # 10 x 128
    b2 = Ym - Hm @ W2.T

    # ---------- scale / sanity ----------
    def clip(a, lim=1000.0):
        return np.clip(a, -lim, lim)

    W1 = clip(W1)
    b1 = clip(b1)
    W2 = clip(W2)
    b2 = clip(b2)

    # keep magnitudes moderate: rescale fc2 rows so logits have reasonable scale
    Zl = H @ W2.T + b2
    s = Zl.std(axis=1).mean()
    if s > 1e-9:
        f2 = min(1.0, 8.0 / s)
        W2 = W2 * f2
        b2 = b2 * f2

    out = dict(w)
    out["fc1.weight"] = W1.astype(np.float32)
    out["fc1.bias"] = b1.astype(np.float32)
    out["fc2.weight"] = W2.astype(np.float32)
    out["fc2.bias"] = b2.astype(np.float32)

    # ---------- diagnostics ----------
    Znew = H @ W2.T + b2
    pred = Znew.argmax(axis=1)
    acc = float((pred == y).mean())
    # baseline: plain linear ridge on raw pixels (for reference)
    Xb = np.hstack([X, np.ones((N, 1))])
    Gb = Xb.T @ Xb + 1e-2 * np.eye(785) * np.trace(Xb.T @ Xb) / 785.0
    Wb = np.linalg.solve(Gb, Xb.T @ Y1)
    accb = float((np.hstack([X, np.ones((N, 1))]) @ Wb).argmax(axis=1).astype(np.int64) == y).mean() if False else float(((np.hstack([X, np.ones((N, 1))]) @ Wb).argmax(axis=1) == y).mean())
    diag = {
        "train_acc_new": acc,
        "train_acc_raw_pixel_ridge": accb,
        "n_hidden": int(W1.shape[0]),
        "lda_dirs": int(U.shape[1]),
        "pca_dirs": int(P.shape[1]),
        "hinge_units": int(min(9 * len(qlev), 128)),
        "H_mean": float(H.mean()),
        "H_active_frac": float((H > 0).mean()),
        "fc1_w_absmax": float(np.abs(W1).max()),
        "fc2_w_absmax": float(np.abs(W2).max()),
        "fc2_b": [float(v) for v in b2],
        "logit_std": float(Znew.std()),
        "class_acc": [float((pred[y == c] == c).mean()) for c in range(C)],
        "per_class_pred_hist": [int((pred == c).sum()) for c in range(C)],
    }
    return {"weights": out, "diagnostics": diag}
