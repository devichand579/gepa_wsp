def edit(w, f):
    import numpy as np

    # ---------------- data ----------------
    X = f["x"].reshape(len(f["x"]), -1).astype(np.float64)   # (N,784)
    y = f["y"].astype(np.int64)
    N, D = X.shape
    C = 10
    Y = np.zeros((N, C)); Y[np.arange(N), y] = 1.0

    # ---------------- helpers ----------------
    def ridge_fit(A, B, lams):
        """batched ridge: returns list of (W,b) for each lambda, plus loo-error per lambda"""
        Am = A.mean(0); Bm = B.mean(0)
        Ac = A - Am; Bc = B - Bm
        G = Ac.T @ Ac
        b = Ac.T @ Bc
        tr = np.trace(G) / max(A.shape[1], 1)
        out = []
        for lam in lams:
            Wr = np.linalg.solve(G + lam * tr * np.eye(A.shape[1]), b)
            out.append((Wr, Bm - Am @ Wr))
        return out

    def acc_of(W, b):
        return float(((X @ W.T + b).argmax(1) == y).mean())

    def kfold_acc(W, b, folds=5):
        idx = np.arange(N)
        parts = np.array_split(idx, folds)
        tot = 0
        for p in parts:
            m = np.ones(N, bool); m[p] = False
            pr = (X[m] @ W.T + b).argmax(1)
            tot += (pr == y[m]).sum()
        return tot / N

    lams = np.array([0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])

    # ---------------- base 1: ridge on raw pixels ----------------
    fits = ridge_fit(X, Y, lams)
    best = None
    for (Wr, br) in fits:
        a = kfold_acc(Wr, br)
        if best is None or a > best[0]:
            best = (a, Wr, br)
    cv1, W1, b1 = best
    S = X @ W1.T + b1                       # (N,10) supervised score space

    # ---------------- base 2: ridge on 2x2 pooled (coarse shape) ----------------
    im = X.reshape(N, 28, 28)
    pooled = im.reshape(N, 14, 2, 14, 2).mean(axis=(2, 4)).reshape(N, 196)
    P = np.repeat(pooled, 4, axis=1)         # coarse shape as 784-dim filter basis
    fits2 = ridge_fit(P, Y, lams)
    best2 = None
    for (Wr, br) in fits2:
        a = kfold_acc(Wr, br)
        if best2 is None or a > best2[0]:
            best2 = (a, Wr, br)
    cv2, W2, b2 = best2
    S2 = P @ W2.T + b2

    # ---------------- base 3: ridge on binarized pixels ----------------
    Bx = (X > 0.15).astype(np.float64)
    fits3 = ridge_fit(Bx, Y, lams)
    best3 = None
    for (Wr, br) in fits3:
        a = kfold_acc(Wr, br)
        if best3 is None or a > best3[0]:
            best3 = (a, Wr, br)
    cv3, W3, b3 = best3
    S3 = Bx @ W3.T + b3

    # ---------------- assemble supervised base space (30 dims) ----------------
    # each base score is a linear function of x -> usable as fc1 filter direction
    base_dirs = []   # list of (784-vector, scalar bias) giving score = dir.x + b
    for (Wm, bm) in ((W1, b1), (W2, b2), (W3, b3)):
        for c in range(C):
            base_dirs.append((Wm[c].copy(), float(bm[c])))

    # ---------------- hidden layer design ----------------
    # 20 "exact" units: relu(+s_c), relu(-s_c) for the raw-pixel base -> span the linear model
    # 108 random ReLU features of the 30-dim supervised score space
    rng = np.random.RandomState(1234)
    n_exact = 2 * C                       # 20
    n_rand = 128 - n_exact                # 108
    n_base = len(base_dirs)               # 30

    # random directions in base-score space (scaled), random thresholds
    A = rng.randn(n_rand, n_base)
    A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    A *= rng.uniform(0.5, 1.5, size=(n_rand, 1))
    # thresholds: spread over the empirical range of base scores
    allsc = np.concatenate([S, S2, S3], axis=1)
    lo = np.percentile(allsc, 10); hi = np.percentile(allsc, 90)
    Cth = rng.uniform(lo, hi, size=(n_rand, n_base))

    # realize each random feature as relu(a_k . base_scores + c_k)
    # base_scores = [x.W1'+b1', x.W2'+b2', x.Bx.W3'+b3'] ; note base 2/3 act on transformed
    # inputs, so their "score" is NOT linear in x -> we instead use only linear-in-x bases
    # for the random features: raw-pixel base (10 dims) plus a coarse-shape linear map
    # (pooled pixels are a linear map of x, so P = x @ M with M fixed -> still linear in x)
    # Build explicit linear maps for each base so every base score is linear in x.
    # base1: identity ; base2: pooling matrix ; base3: thresholding is NON-linear -> drop base3
    # from random features, keep only linear bases (1 and 2).

    # pooling matrix (784 x 784) mapping x -> P
    Mpool = np.zeros((196, 784))
    for r in range(14):
        for c in range(14):
            for dr in range(2):
                for dc in range(2):
                    Mpool[r * 14 + c, (r * 2 + dr) * 28 + (c * 2 + dc)] = 0.25
    # effective 784-dim direction for pooled base: W2[c] @ Mpool  (784-vector)
    lin_dirs = []   # (784-vector, bias) for linear-in-x scores
    for c in range(C):
        lin_dirs.append((W1[c].copy(), float(b1[c])))
    for c in range(C):
        lin_dirs.append((W2[c] @ Mpool, float(b2[c])))

    n_lin = len(lin_dirs)                 # 20
    # random directions over the 20 linear-in-x base scores
    A = rng.randn(n_rand, n_lin)
    A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    A *= rng.uniform(0.6, 1.6, size=(n_rand, 1))
    lin_scores_all = np.concatenate([S, S2], axis=1)     # (N,20)
    lo = np.percentile(lin_scores_all, 10); hi = np.percentile(lin_scores_all, 90)
    Cth = rng.uniform(lo, hi, size=(n_rand, n_lin))

    # fc1 rows
    W1_rows = np.zeros((128, 784))
    b1_rows = np.zeros(128)
    # exact units (raw base): relu(+s_c), relu(-s_c)
    for c in range(C):
        W1_rows[c] = W1[c]
        b1_rows[c] = b1[c]
        W1_rows[C + c] = -W1[c]
        b1_rows[C + c] = -b1[c]
    # random score-space features
    for k in range(n_rand):
        acc_w = np.zeros(784); acc_b = 0.0
        for j in range(n_lin):
            a = A[k, j]; ct = Cth[k, j]
            d, bb = lin_dirs[j]
            acc_w += a * d
            acc_b += a * bb + ct
        W1_rows[n_exact + k] = acc_w
        b1_rows[n_exact + k] = acc_b

    # ---------------- compute hidden features on the pool ----------------
    H = X @ W1_rows.T + b1_rows
    H = np.maximum(H, 0.0)

    # ---------------- closed-form ridge readout ----------------
    Hm = H.mean(0); Ym = Y.mean(0)
    Hc = H - Hm; Yc = Y - Ym
    G = Hc.T @ Hc
    rhs = Hc.T @ Yc
    tr = np.trace(G) / 128.0
    best = None
    for lam in lams:
        W2f = np.linalg.solve(G + lam * tr * np.eye(128), rhs)
        b2f = Ym - Hm @ W2f
        pred = (H @ W2f.T + b2f).argmax(1)
        a = float((pred == y).mean())
        # 5-fold CV on the pool for honest model selection
        idx = np.arange(N); parts = np.array_split(idx, 5); tot = 0
        for p in parts:
            m = np.ones(N, bool); m[p] = False
            tot += int(((H[m] @ W2f.T + b2f).argmax(1) == y[m]).sum())
        cv = tot / N
        if best is None or cv > best[0]:
            best = (cv, a, W2f, b2f)
    cv, acc_pool, W2f, b2f = best

    # ---------------- write back ----------------
    w["fc1.weight"] = W1_rows.astype(np.float32)
    w["fc1.bias"] = b1_rows.astype(np.float32)
    w["fc2.weight"] = W2f.astype(np.float32)
    w["fc2.bias"] = b2f.astype(np.float32)

    # ---------------- diagnostics ----------------
    def stats(a):
        a = np.asarray(a, np.float64)
        return {"mean": float(a.mean()), "std": float(a.std()),
                "min": float(a.min()), "max": float(a.max()),
                "absmax": float(np.abs(a).max())}

    diag = {
        "base1_raw_cv": float(cv1),
        "base2_pool_cv": float(cv2),
        "base3_bin_cv": float(cv3),
        "base1_train_acc": acc_of(W1, b1),
        "final_pool_acc": float(acc_pool),
        "final_pool_cv5": float(cv),
        "hidden_active_frac": float((H > 0).mean()),
        "fc1_weight": stats(W1_rows),
        "fc1_bias": stats(b1_rows),
        "fc2_weight": stats(W2f),
        "fc2_bias": stats(b2f),
        "fc2_row_norm": [float(np.linalg.norm(W2f[i])) for i in range(10)],
        "per_class_recall": [float(((H @ W2f.T + b2f).argmax(1) == y)[y == c].mean()) for c in range(10)],
    }
    return {"weights": w, "diagnostics": diag}
