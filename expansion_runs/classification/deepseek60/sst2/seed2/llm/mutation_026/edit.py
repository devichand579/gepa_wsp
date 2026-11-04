import numpy as np

def edit(w, f):
    X = f["x"].astype(np.int64)          # [4000,64]
    y = f["y"].astype(np.int64)          # [4000]
    N, L = X.shape
    V = w["embed.weight"].shape[0]       # 2000
    D = w["embed.weight"].shape[1]       # 32

    # ---------- 1. Bag-of-words (counts, padding id 0 excluded) ----------
    mask = (X != 0)
    BOW = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), L)
    cols = X.reshape(-1)
    m = mask.reshape(-1)
    np.add.at(BOW, (rows[m], cols[m]), 1.0)
    # length-normalized version (helps a lot for SST-2)
    lens = mask.sum(1).astype(np.float64)
    lens[lens == 0] = 1.0
    BOWN = BOW / lens[:, None]

    # ---------- 2. Closed-form ridge classifier on BOW (dual form) ----------
    # center features and labels
    yc = y.astype(np.float64) * 2.0 - 1.0            # +/-1
    mu = BOWN.mean(0, keepdims=True)
    Xc = BOWN - mu
    ym = yc.mean()
    ycen = yc - ym

    def ridge_w(Xc, ycen, lam):
        # solve (Xc^T Xc + lam I) w = Xc^T ycen  via dual (n=4000 < p=2000? p=2000)
        n, p = Xc.shape
        if p <= n:
            A = Xc.T @ Xc + lam * np.eye(p)
            return np.linalg.solve(A, Xc.T @ ycen)
        else:
            K = Xc @ Xc.T + lam * np.eye(n)
            a = np.linalg.solve(K, ycen)
            return Xc.T @ a

    # choose lambda by 5-fold CV on the training pool (closed form, no iter)
    rng = np.random.RandomState(0)
    idx = rng.permutation(N)
    folds = np.array_split(idx, 5)
    best_lam, best_acc = 1.0, -1.0
    for lam in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
        accs = []
        for k in range(5):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(5) if j != k])
            Xtr, Xte = Xc[tr], Xc[te]
            ytr = ycen[tr]
            ww = ridge_w(Xtr, ytr, lam)
            sc = Xte @ ww
            pred = (sc > 0).astype(np.int64)
            accs.append((pred == y[te]).mean())
        a = float(np.mean(accs))
        if a > best_acc:
            best_acc, best_lam = a, lam
    w_bow = ridge_w(Xc, ycen, best_lam)              # [2000]
    w_bow = w_bow - w_bow.mean()                     # zero-mean for stability

    # ---------- 3. Build embedding: embed[t] = w_bow[t]*u + c*v ----------
    u = np.zeros(D); u[0] = 1.0; u[1] = -1.0; u /= np.linalg.norm(u)
    v = np.zeros(D); v[2] = 1.0; v[3] = -1.0; v /= np.linalg.norm(v)
    # ensure orthonormal & zero-mean
    u = u - u.mean(); u /= np.linalg.norm(u)
    v = v - v.mean()
    v = v - (v @ u) * u
    v /= np.linalg.norm(v)

    scale_w = np.abs(w_bow).max() + 1e-9
    c = 8.0 * scale_w + 1.0
    E = np.outer(w_bow, u) + c * np.outer(np.ones(V), v)   # [2000,32]
    # keep magnitudes bounded
    E = np.clip(E, -1000.0, 1000.0)

    w["embed.weight"] = E.astype(np.float32)
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)

    # zero out both transformer blocks (identity path)
    for b in ["block0", "block1"]:
        for name in ["norm1.weight", "norm1.bias", "qkv.weight", "qkv.bias",
                     "proj.weight", "proj.bias", "norm2.weight", "norm2.bias",
                     "ff1.weight", "ff1.bias", "ff2.weight", "ff2.bias"]:
            key = b + "." + name
            if key in w:
                w[key] = np.zeros_like(w[key], dtype=np.float32)

    # final norm identity
    w["norm.weight"] = np.ones_like(w["norm.weight"], dtype=np.float32)
    w["norm.bias"] = np.zeros_like(w["norm.bias"], dtype=np.float32)

    # ---------- 4. Compute pooled features with this construction ----------
    # blocks are zero, pos zero => h = embed[ids]; pooled = mean_t LN(embed[t])
    Emb = w["embed.weight"].astype(np.float64)          # [V,32]
    # per-token LayerNorm (eps=1e-5), gamma=1,beta=0
    mu_t = Emb.mean(1, keepdims=True)
    var_t = Emb.var(1, keepdims=True)
    LN = (Emb - mu_t) / np.sqrt(var_t + 1e-5)           # [V,32]
    # gather per example and mean over non-pad
    LNx = LN[X]                                          # [N,64,32]
    m3 = mask[..., None].astype(np.float64)
    pooled = (LNx * m3).sum(1) / lens[:, None]           # [N,32]

    # ---------- 5. Fit head by closed-form ridge on pooled ----------
    # targets +/-1
    P = pooled
    Pm = P.mean(0, keepdims=True)
    Pc = P - Pm
    best_lam_h, best_acc_h = 1.0, -1.0
    for lam in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
        accs = []
        for k in range(5):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(5) if j != k])
            A = Pc[tr].T @ Pc[tr] + lam * np.eye(D)
            ww = np.linalg.solve(A, Pc[tr].T @ ycen[tr])
            sc = Pc[te] @ ww
            accs.append(((sc > 0).astype(np.int64) == y[te]).mean())
        a = float(np.mean(accs))
        if a > best_acc_h:
            best_acc_h, best_lam_h = a, lam
    whead = np.linalg.solve(Pc.T @ Pc + best_lam_h * np.eye(D), Pc.T @ ycen)
    # head: class1 weight = +whead/2, class0 = -whead/2 (so diff = whead)
    head_w = np.zeros((2, D), dtype=np.float64)
    head_w[1] = 0.5 * whead
    head_w[0] = -0.5 * whead
    # bias so decision boundary at 0: logit1-logit0 = whead·P + (b1-b0)
    b1mb0 = -(Pm @ whead)
    head_b = np.array([-0.5 * b1mb0, 0.5 * b1mb0], dtype=np.float64)
    w["head.weight"] = head_w.astype(np.float32)
    w["head.bias"] = head_b.astype(np.float32)

    # ---------- diagnostics ----------
    logits = P @ head_w.T + head_b
    pred = logits.argmax(1)
    acc = float((pred == y).mean())
    p = np.exp(logits - logits.max(1, keepdims=True))
    p = p / p.sum(1, keepdims=True)
    ce = float(-np.log(np.clip(p[np.arange(N), y], 1e-9, 1.0)).mean())

    diag = {
        "bow_cv_acc": round(best_acc, 4),
        "bow_lambda": best_lam,
        "head_cv_acc": round(best_acc_h, 4),
        "head_lambda": best_lam_h,
        "train_acc": round(acc, 4),
        "train_ce": round(ce, 4),
        "w_bow_absmax": float(np.abs(w_bow).max()),
        "c": float(c),
        "pooled_std": float(pooled.std()),
    }
    return {"weights": w, "diagnostics": diag}
