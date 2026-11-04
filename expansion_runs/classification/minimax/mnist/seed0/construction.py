def edit(w, f):
    import numpy as np

    # ---------------- data ----------------
    X = f["x"].reshape(-1, 784).astype(np.float64)
    Y = f["y"].astype(np.int64)
    N, D = X.shape
    K = 10

    # ---------------- whitening (closed form) ----------------
    mu = X.mean(0)
    Xc = X - mu
    S = (Xc.T @ Xc) / N
    tr = np.trace(S) / D
    lam_s = 0.08 * tr
    Sreg = S + lam_s * np.eye(D)
    evals, evecs = np.linalg.eigh(Sreg)
    evals = np.maximum(evals, 1e-8)
    Wht = evecs @ np.diag(evals ** -0.5) @ evecs.T          # symmetric whitening
    Z = Xc @ Wht                                             # whitened inputs
    Zm = Z.mean(0)
    Zc = Z - Zm

    # class means / within-class scatter in whitened space
    Mc = np.zeros((K, D))
    for c in range(K):
        Mc[c] = Z[Y == c].mean(0)
    Sw = np.zeros((D, D))
    for c in range(K):
        Zs = Zc[Y == c]
        Sw += Zs.T @ Zs
    Sw /= N
    Sw_reg = Sw + 0.02 * np.eye(D)
    # ridge-regularized inverse of within-class covariance
    ev2, V2 = np.linalg.eigh(Sw_reg)
    ev2 = np.maximum(ev2, 1e-8)
    Sinv = V2 @ np.diag(ev2 ** -1.0) @ V2.T

    # ---------------- hidden-unit construction ----------------
    units = []          # list of (w_row(784), bias)
    diag = {}

    # (a) Fisher / LDA directions, one per class (hinged at class-conditional quantile)
    for c in range(K):
        d = Sinv @ (Mc[c] - Zm)
        n = np.linalg.norm(d)
        if n < 1e-9:
            continue
        d = d / n
        s = Z @ d
        thr = np.quantile(s[Y == c], 0.35)
        units.append((d, -float(thr)))
    diag["n_fisher"] = len(units)

    # (b) class-mean matched filters (hinged at class-conditional quantile)
    for c in range(K):
        t = Mc[c] - Zm
        n = np.linalg.norm(t)
        if n < 1e-9:
            continue
        t = t / n
        s = Z @ t
        thr = np.quantile(s[Y == c], 0.45)
        units.append((t, -float(thr)))
    diag["n_proto"] = len(units) - diag["n_fisher"]

    # (c) random ReLU features of whitened input (fill remaining capacity)
    n_rand = 128 - len(units)
    if n_rand > 0:
        rng = np.random.RandomState(1234)
        A = rng.randn(n_rand, D) / np.sqrt(D)
        sA = Z @ A.T
        # per-unit threshold at median of pre-activation -> ~50% active
        thr = np.quantile(sA, 0.5, axis=0)
        for j in range(n_rand):
            units.append((A[j], -float(thr[j])))
    diag["n_rand"] = n_rand

    # (d) small random set on raw (unwhitened) pixels for extra diversity
    n_raw = 8
    rng2 = np.random.RandomState(77)
    B = rng2.randn(n_raw, D) * 0.02
    sB = X @ B.T
    thrB = np.quantile(sB, 0.5, axis=0)
    for j in range(n_raw):
        units.append((B[j], -float(thrB[j])))
    diag["n_raw"] = n_raw

    # trim/pad to exactly 128 units
    if len(units) > 128:
        units = units[:128]
    while len(units) < 128:
        rng3 = np.random.RandomState(999 + len(units))
        d = rng3.randn(D) * 0.02
        units.append((d, 0.0))

    W1 = np.zeros((128, D))
    b1 = np.zeros(128)
    for j, (u, bb) in enumerate(units):
        W1[j] = u
        b1[j] = bb

    # ---------------- recompute features with the NEW fc1 ----------------
    H = np.maximum(0.0, X @ W1.T + b1)          # [N,128]
    diag["feat_mean"] = float(H.mean())
    diag["feat_std"] = float(H.std())
    diag["feat_zero_frac"] = float((H == 0).mean())

    # ---------------- closed-form ridge readout on one-hot targets ----------------
    Y1 = np.zeros((N, K))
    Y1[np.arange(N), Y] = 1.0
    # class-balanced target scaling (each class contributes equally)
    cnt = np.bincount(Y, minlength=K).astype(np.float64)
    Y1 = Y1 * (N / (K * cnt))[Y][:, None]

    # split for lambda selection
    rng4 = np.random.RandomState(7)
    perm = rng4.permutation(N)
    ntr = int(0.8 * N)
    tr, va = perm[:ntr], perm[ntr:]

    def fit_ridge(Htr, Ytr, lam):
        G = Htr.T @ Htr + lam * np.eye(Htr.shape[1])
        Wf = np.linalg.solve(G, Htr.T @ Ytr)
        bf = Ytr.mean(0) - Htr.mean(0) @ Wf
        return Wf, bf

    def acc(Hm, Wf, bf, ym):
        return float((np.argmax(Hm @ Wf + bf, axis=1) == ym).mean())

    lams = [1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0]
    best = None
    for lam in lams:
        Wf, bf = fit_ridge(H[tr], Y1[tr], lam)
        a = acc(H[va], Wf, bf, Y[va])
        if best is None or a > best[0]:
            best = (a, lam, Wf, bf)
    val_acc, lam, Wf, bf = best

    # refit on all data with selected lambda
    Wf, bf = fit_ridge(H, Y1, lam)

    tr_acc = acc(H, Wf, bf, Y)
    pred = np.argmax(H @ Wf + bf, axis=1)
    per_class = [float((pred[Y == c] == c).mean()) for c in range(K)]
    diag["lambda"] = float(lam)
    diag["train_acc"] = tr_acc
    diag["val_acc"] = float(val_acc)
    diag["per_class_acc"] = per_class
    diag["n_units"] = 128

    # ---------------- write back ----------------
    out = {}
    for k, v in w.items():
        out[k] = np.array(v, dtype=np.float32, copy=True)
    out["fc1.weight"] = np.clip(W1.T, -1000, 1000).astype(np.float32)   # [784,128]
    out["fc1.bias"] = np.clip(b1, -1000, 1000).astype(np.float32)
    out["fc2.weight"] = np.clip(Wf.T, -1000, 1000).astype(np.float32)   # [128,10]
    out["fc2.bias"] = np.clip(bf, -1000, 1000).astype(np.float32)

    # sanity: finite + range
    for k in ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"):
        assert np.all(np.isfinite(out[k])), k
        assert np.abs(out[k]).max() <= 1000.0, k
    diag["fc1_absmax"] = float(np.abs(out["fc1.weight"]).max())
    diag["fc2_absmax"] = float(np.abs(out["fc2.weight"]).max())

    return {"weights": out, "diagnostics": diag}
