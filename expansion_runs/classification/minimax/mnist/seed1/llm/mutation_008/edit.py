def edit(w, f):
    import numpy as np

    X = np.asarray(f["x"], dtype=np.float64).reshape(-1, 784)
    y = np.asarray(f["y"]).astype(np.int64).ravel()
    N, D = X.shape
    K = 10

    # ---------------- class means / within-class scatter ----------------
    mu_bar = X.mean(axis=0)
    Xc = X - mu_bar
    cls_mean = np.zeros((K, D))
    for c in range(K):
        cls_mean[c] = X[y == c].mean(axis=0)
    Xw = X - cls_mean[y]

    # ---------------- PCA denoise (closed form SVD) ----------------
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    d = 150
    d = min(d, Vt.shape[0])
    V = Vt[:d].T                      # d x D
    Z = Xc @ V                         # N x d  (PCA coords, centered)
    Zm = Z - Z.mean(axis=0)
    zbar = Z.mean(axis=0)

    # ---------------- within-class whitening in PCA space ----------------
    Wz = Z - cls_mean[y] @ V
    Sw = (Wz.T @ Wz) / float(N)
    ew, Uw = np.linalg.eigh(Sw)
    ew = np.maximum(ew, 1e-8)
    order = np.argsort(ew)[::-1]
    ew = ew[order]
    Uw = Uw[:, order]

    def whiten(beta):
        scale = ew ** (-beta)
        return (Uw * scale) @ Uw.T

    # ---------------- target coding ----------------
    T = np.full((N, K), -1.0 / (K - 1))
    T[np.arange(N), y] = 1.0

    # ---------------- helpers ----------------
    def make_features(beta, seed, n_rand=96, n_det=32, act=0.55):
        A = whiten(beta)
        Uw_feat = (Z - zbar) @ A.T          # N x d  whitened coords
        rng = np.random.default_rng(seed)
        dirs = []
        # class detector directions: Mahalanobis class means
        cm = np.zeros((K, d))
        for c in range(K):
            v = (cls_mean[c] - mu_bar) @ V @ A.T
            nrm = np.linalg.norm(v)
            cm[c] = v / (nrm + 1e-9)
        # 1) class template detectors (2 thresholds per class = 20)
        for c in range(K):
            for q in (0.75, 0.90):
                dirs.append((cm[c], q, 1.0))
        # 2) pairwise class contrasts (12): c1 - c2
        pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (0, 4),
                 (1, 7), (3, 8), (2, 6), (5, 9), (0, 7), (3, 5)]
        for (a, b) in pairs:
            v = cm[a] - cm[b]
            nrm = np.linalg.norm(v)
            dirs.append((v / (nrm + 1e-9), 0.80, 1.0))
        # 3) random probes
        while len(dirs) < n_det + n_rand:
            v = rng.normal(size=d)
            v = v / (np.linalg.norm(v) + 1e-9)
            dirs.append((v, 1.0 - act, 0.0))
        dirs = dirs[:n_det + n_rand]
        F = np.zeros((N, len(dirs)))
        for j, (v, q, isdet) in enumerate(dirs):
            p = Uw_feat @ v
            thr = np.quantile(p, q)
            F[:, j] = np.maximum(p - thr, 0.0)
        return F

    def ridge_fit(F, lam):
        mu = F.mean(axis=0)
        sd = F.std(axis=0)
        sd = np.maximum(sd, 1e-6)
        Fs = (F - mu) / sd
        G = Fs.T @ Fs
        G[np.diag_indices_from(G)] += lam
        W = np.linalg.solve(G, Fs.T @ T)
        b = T.mean(axis=0) - mu @ W
        return W, b, mu, sd

    def acc_of(F, W, b, mu, sd, idx):
        Fs = (F[idx] - mu) / sd
        pred = np.argmax(Fs @ W + b, axis=1)
        return float(np.mean(pred == y[idx]))

    # ---------------- closed-form CV model selection ----------------
    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    folds = np.array_split(perm, 5)
    lams = [1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0]
    betas = [0.5, 1.0]

    best = None
    for beta in betas:
        F = make_features(beta, seed=17)
        for lam in lams:
            accs = []
            for k in range(5):
                te = folds[k]
                tr = np.concatenate([folds[j] for j in range(5) if j != k])
                W, b, mu, sd = ridge_fit(F[tr], lam)
                accs.append(acc_of(F, W, b, mu, sd, te))
            a = float(np.mean(accs))
            if best is None or a > best[0]:
                best = (a, beta, lam)

    cv_acc, beta, lam = best
    F = make_features(beta, seed=17)
    W, b, mu, sd = ridge_fit(F, lam)

    # ---------------- write back into architecture ----------------
    # fc1: W1 [128,784], b1 [128];  fc2: W2 [10,128], b2 [10]
    A = whiten(beta)
    # reconstruct the raw-space first-layer weight for each hidden unit
    # feature j = relu( v_j . whitened(x) - thr_j )
    # whitened(x) = ((x - mu_bar) @ V) @ A.T - zbar @ A.T
    # => v_j . whitened(x) = (x - mu_bar) @ (V @ A.T @ v_j)
    n_units = F.shape[1]
    W1 = np.zeros((n_units, D))
    b1 = np.zeros(n_units)
    # recompute directions/thresholds identically
    rng = np.random.default_rng(17)
    cm = np.zeros((K, d))
    for c in range(K):
        v = (cls_mean[c] - mu_bar) @ V @ A.T
        cm[c] = v / (np.linalg.norm(v) + 1e-9)
    dirs = []
    for c in range(K):
        for q in (0.75, 0.90):
            dirs.append((cm[c], q))
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (0, 4),
             (1, 7), (3, 8), (2, 6), (5, 9), (0, 7), (3, 5)]
    for (a, b) in pairs:
        v = cm[a] - cm[b]
        dirs.append((v / (np.linalg.norm(v) + 1e-9), 0.80))
    while len(dirs) < n_units:
        v = rng.normal(size=d)
        dirs.append((v / (np.linalg.norm(v) + 1e-9), 0.45))
    dirs = dirs[:n_units]
    Uw_feat = (Z - zbar) @ A.T
    for j, (v, q) in enumerate(dirs):
        p = Uw_feat @ v
        thr = np.quantile(p, q)
        wvec = V @ (A.T @ v)          # D-vector so that v.whitened(x) = wvec . (x - mu_bar) + const
        const = -zbar @ (A.T @ v)     # v . ((x-mu_bar)@V@A.T - zbar@A.T)
        # v.whitened(x) = wvec@(x-mu_bar) + const
        W1[j] = wvec
        b1[j] = thr - const
    # note: feature = relu(v.whitened(x) - thr) = relu(W1[j].x + b1[j])
    W2 = np.zeros((K, n_units))
    b2 = np.zeros(K)
    W2[:, :] = (W / sd[:, None]).T
    b2 = b - mu @ W

    # ---------------- diagnostics ----------------
    H = np.maximum(X @ W1.T + b1[None, :], 0.0)
    logits = H @ W2.T + b2[None, :]
    pred = np.argmax(logits, axis=1)
    acc = float(np.mean(pred == y))
    mx = logits.max(axis=1)
    lse = mx + np.log(np.exp(logits - mx[:, None]).sum(axis=1))
    tgt = logits[np.arange(N), y]
    ce = float(np.mean(lse - tgt))
    # per-class accuracy
    per_cls = {}
    for c in range(K):
        m = (y == c)
        per_cls[str(c)] = round(float(np.mean(pred[m] == y[m])), 4)
    # batch diagnostics
    bx = np.asarray(f["batch.x"], dtype=np.float64).reshape(-1, 784)
    bh = np.maximum(bx @ W1.T + b1[None, :], 0.0)
    bl = bh @ W2.T + b2[None, :]
    bpred = np.argmax(bl, axis=1)
    by = np.asarray(f["batch.y"]).astype(np.int64).ravel()
    bacc = float(np.mean(bpred == by))

    def fin(a):
        a = np.asarray(a, dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(a, -1000.0, 1000.0).astype(np.float32)

    w["fc1.weight"] = fin(W1)
    w["fc1.bias"] = fin(b1)
    w["fc2.weight"] = fin(W2)
    w["fc2.bias"] = fin(b2)

    return {
        "weights": w,
        "diagnostics": {
            "cv_acc": round(cv_acc, 4),
            "beta": beta,
            "ridge_lambda": lam,
            "train_acc": round(acc, 4),
            "train_ce": round(ce, 4),
            "batch_acc": round(bacc, 4),
            "per_class_acc": per_cls,
            "fc1_absmax": float(np.abs(W1).max()),
            "fc2_absmax": float(np.abs(W2).max()),
            "feat_mean": float(H.mean()),
            "feat_std": float(H.std()),
            "feat_zero_frac": float((H == 0).mean()),
            "logit_std": float(logits.std()),
            "n_units": int(n_units),
        },
    }
