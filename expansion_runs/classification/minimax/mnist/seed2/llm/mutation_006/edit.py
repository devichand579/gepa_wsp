def edit(w, f):
    # ---------------- data ----------------
    X = np.asarray(f["x"], dtype=np.float64).reshape(len(f["x"]), -1)   # (N,784)
    y = np.asarray(f["y"]).astype(np.int64)
    N, D = X.shape
    C = 10
    mu = X.mean(axis=0)
    Xc = X - mu

    # ---------------- PCA (closed form) ----------------
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(200, len(S))
    V = Vt[:k].T                      # (784,k) principal directions
    Z = Xc @ V                        # (N,k) scores
    sd = S[:k] / np.sqrt(N - 1.0)
    sd = np.maximum(sd, 1e-8)
    # partial whitening (power 0.5) keeps low-variance noise from dominating
    Zw = Z / np.sqrt(sd)

    # ---------------- class statistics ----------------
    Y1 = np.zeros((N, C))
    Y1[np.arange(N), y] = 1.0
    cls = [np.where(y == c)[0] for c in range(C)]
    Mc = np.stack([Zw[cls[c]].mean(axis=0) for c in range(C)])   # (C,k)
    Mc_bar = Mc.mean(axis=0)
    Sw = np.zeros((k, k))
    for c in range(C):
        Dc = Zw[cls[c]] - Mc[c]
        Sw += Dc.T @ Dc
    Sw /= max(N - C, 1)
    Sb = np.zeros((k, k))
    for c in range(C):
        d = (Mc[c] - Mc_bar).reshape(-1, 1)
        Sb += cls[c].size * (d @ d.T)
    Sb /= max(N - C, 1)

    # ---------------- regularized LDA directions ----------------
    tr = np.trace(Sw) / k
    a = 0.3
    Swr = (1.0 - a) * Sw + a * tr * np.eye(k) + 1e-6 * np.eye(k)
    # solve Swr^{-1} Sb eigenproblem
    L = np.linalg.cholesky(Swr)
    Li = np.linalg.inv(L)
    M = Li @ Sb @ Li.T
    M = 0.5 * (M + M.T)
    ev, evec = np.linalg.eigh(M)
    order = np.argsort(ev)[::-1]
    ev = ev[order]
    evec = evec[:, order]
    lda_dirs = Li.T @ evec                      # (k, k) whitened-space directions
    lda_dirs = lda_dirs / (np.linalg.norm(lda_dirs, axis=0, keepdims=True) + 1e-12)
    D_lda = lda_dirs[:, :9]                     # 9 discriminant directions

    # class-pair difference directions (supervised, extra discriminability)
    pairs = []
    for i in range(C):
        for j in range(i + 1, C):
            d = Mc[i] - Mc[j]
            nd = np.linalg.norm(d)
            if nd > 1e-9:
                pairs.append((nd, d / nd))
    pairs.sort(key=lambda t: -t[0])
    npair = min(18, len(pairs))
    D_pair = np.stack([p[1] for p in pairs[:npair]], axis=1) if npair > 0 else np.zeros((k, 0))

    D_sup = np.concatenate([D_lda, D_pair], axis=1)     # (k, 27)

    # ---------------- hidden unit construction ----------------
    # supervised directions -> hinge units at projection quantiles
    proj_sup = Zw @ D_sup                                # (N, 27)
    qs = [0.35, 0.60, 0.80]
    hinge_cols = []
    hinge_meta = []
    for j in range(D_sup.shape[1]):
        p = proj_sup[:, j]
        for q in qs:
            t = float(np.quantile(p, q))
            hinge_cols.append(p - t)
            hinge_meta.append((j, t))
    Hsup = np.maximum(np.stack(hinge_cols, axis=1), 0.0)   # (N, 81)

    # random units in whitened space
    rng = np.random.default_rng(12345)
    n_rand = 128 - Hsup.shape[1]
    if n_rand < 0:
        Hsup = Hsup[:, :128]
        n_rand = 0
    R = rng.normal(size=(k, n_rand))
    R /= (np.linalg.norm(R, axis=0, keepdims=True) + 1e-12)
    pr = Zw @ R
    thr = np.quantile(pr, 0.5, axis=0) if n_rand > 0 else np.zeros(0)
    Hrand = np.maximum(pr - thr[None, :], 0.0)

    H = np.concatenate([Hsup, Hrand], axis=1)            # (N,128)
    H = np.concatenate([H, np.ones((N, 1))], axis=1)       # intercept column (bias handled separately)

    # ---------------- fc1 weights (pixel space) ----------------
    # fc1 unit j : relu( (w_j . x) + b_j ), w_j = V @ (coeff in PCA space)
    # supervised unit j uses direction D_sup[:, jj] with threshold t:
    #   w_j = V @ D_sup[:,jj] / sd_scale ; b_j = -t / sd_scale
    # PCA-space coefficient for unit with whitened direction u: V @ (u / sqrt(sd))
    coef = np.zeros((k, 128))
    bfc1 = np.zeros(128)
    n_sup_units = Hsup.shape[1]
    for idx, (jj, t) in enumerate(hinge_meta):
        u = D_sup[:, jj]
        cvec = u / np.sqrt(sd)
        coef[:, idx] = cvec
        bfc1[idx] = -t / np.sqrt(sd)
    for idx in range(n_rand):
        u = R[:, idx]
        cvec = u / np.sqrt(sd)
        coef[:, n_sup_units + idx] = cvec
        bfc1[n_sup_units + idx] = -thr[idx] / np.sqrt(sd)
    W1_new = (V @ coef).T                                  # (128,784)
    # scale down so pre-activations are O(1)
    pre = X @ W1_new.T + bfc1[None, :]
    s = np.percentile(pre, 99.0)
    if s > 1e-6:
        W1_new = W1_new / s
        bfc1 = bfc1 / s
    # re-center: subtract per-unit mean of pre-activation so relu keeps ~half active
    pre = X @ W1_new.T + bfc1[None, :]
    bfc1 = bfc1 - pre.mean(axis=0)
    Hn = np.maximum(X @ W1_new.T + bfc1[None, :], 0.0)

    # ---------------- heads ----------------
    def softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def ridge_fit(Ht, Yt, lam):
        A = Ht.T @ Ht + lam * np.eye(Ht.shape[1])
        B = Ht.T @ Yt
        return np.linalg.solve(A, B)

    def lda_fit(Ht, Yt, shrink):
        # regularized LDA: shared covariance from residuals
        mus = np.zeros((C, Ht.shape[1]))
        for c in range(C):
            idx = np.where(Yt[:, c] > 0.5)[0]
            mus[c] = Ht[idx].mean(axis=0) if len(idx) > 0 else Ht.mean(axis=0)
        Rres = Ht - Yt @ mus
        S = Rres.T @ Rres / max(Ht.shape[0] - C, 1)
        tr = np.trace(S) / S.shape[0]
        S = (1.0 - shrink) * S + shrink * tr * np.eye(S.shape[0]) + 1e-8 * np.eye(S.shape[0])
        Si = np.linalg.inv(S)
        Wl = Si @ mus.T                     # (feat, C)
        bl = -0.5 * np.sum(mus * (Si @ mus.T).T, axis=1) + np.log(0.1)
        return Wl, bl

    # closed-form 5-fold CV over a small fixed grid (ridge lambda / LDA shrinkage)
    folds = np.arange(N) % 5
    rng2 = np.random.default_rng(7)
    perm = rng2.permutation(N)
    folds = np.zeros(N, dtype=np.int64)
    folds[perm] = np.arange(N) % 5

    def cv_acc(kind, param):
        correct = 0
        for kf in range(5):
            tr = folds != kf
            te = folds == kf
            if kind == "ridge":
                Wf = ridge_fit(Hn[tr], Y1[tr], param)
                pred = Hn[te] @ Wf
            else:
                Wf, bf = lda_fit(Hn[tr], Y1[tr], param)
                pred = Hn[te] @ Wf + bf[None, :]
            correct += int((pred.argmax(axis=1) == y[te]).sum())
        return correct / float(N)

    cands = []
    for lam in [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
        cands.append(("ridge", lam, cv_acc("ridge", lam)))
    for sh in [0.02, 0.05, 0.1, 0.2, 0.4]:
        cands.append(("lda", sh, cv_acc("lda", sh)))
    cands.sort(key=lambda t: -t[2])
    kind, param, cv = cands[0]

    if kind == "ridge":
        W2 = ridge_fit(Hn, Y1, param)
        b2 = np.zeros(C)
    else:
        W2, b2 = lda_fit(Hn, Y1, param)
        # keep magnitudes moderate: scale so max |logit| reasonable
        lg = Hn @ W2 + b2[None, :]
        sc = max(1.0, np.percentile(np.abs(lg), 99.0) / 8.0)
        W2 = W2 / sc
        b2 = b2 / sc

    # ---------------- assemble ----------------
    out = {}
    out["fc1.weight"] = np.clip(W1_new.astype(np.float32), -1000.0, 1000.0)
    out["fc1.bias"] = np.clip(bfc1.astype(np.float32), -1000.0, 1000.0)
    out["fc2.weight"] = np.clip(W2.T.astype(np.float32), -1000.0, 1000.0)
    out["fc2.bias"] = np.clip(b2.astype(np.float32), -1000.0, 1000.0)

    # ---------------- diagnostics (own forward pass) ----------------
    lg = Hn @ W2 + b2[None, :]
    p = softmax(lg)
    pred = lg.argmax(axis=1)
    acc = float((pred == y).mean())
    ce = float(-np.log(np.maximum(p[np.arange(N), y], 1e-12)).mean())
    per_cls = [float((pred[cls[c]] == c).mean()) for c in range(C)]
    old_acc = float((np.asarray(f["logits"]).argmax(axis=1) == y).mean())

    diag = {
        "design": {
            "n_units": 128, "n_sup_hinge": int(n_sup_units), "n_rand": int(n_rand),
            "n_lda": 9, "n_pair": int(npair), "lda_shrink": a,
            "head": kind, "param": float(param), "cv_acc": float(cv),
            "candidates": [[c[0], float(c[1]), float(c[2])] for c in cands[:6]],
        },
        "train": {
            "acc": acc, "ce": ce, "old_acc": acc * 0 + old_acc,
            "per_class_acc": per_cls,
            "hidden_zero_frac": float((Hn == 0).mean()),
            "hidden_mean": float(Hn.mean()), "hidden_std": float(Hn.std()),
            "logit_std": float(lg.std()), "max_abs_logit": float(np.abs(lg).max()),
            "fc1_absmax": float(np.abs(out["fc1.weight"]).max()),
            "fc2_absmax": float(np.abs(out["fc2.weight"]).max()),
            "fc1_bias_absmax": float(np.abs(out["fc1.bias"]).max()),
        },
    }
    return {"weights": out, "diagnostics": diag}
