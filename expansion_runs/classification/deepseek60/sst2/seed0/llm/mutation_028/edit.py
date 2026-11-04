import numpy as np

def edit(w, f):
    X = f["x"]            # [N,64] int64
    y = f["y"]            # [N]
    N, L = X.shape
    V = w["embed.weight"].shape[0]
    D = w["embed.weight"].shape[1]   # 32

    mask = (X != 0)
    # ---- closed-form smoothed Naive-Bayes log-odds token scores ----
    # counts of each token in each class (presence-weighted and count-weighted blend)
    alpha = 1.0
    cnt_pos = np.zeros(V, dtype=np.float64)
    cnt_neg = np.zeros(V, dtype=np.float64)
    for c, arr in ((1, cnt_pos), (0, cnt_neg)):
        sel = (y == c)
        ids = X[sel]
        m = (ids != 0)
        np.add.at(arr, ids[m].ravel(), 1.0)
    tot_pos = cnt_pos.sum()
    tot_neg = cnt_neg.sum()
    # log-odds with Laplace smoothing
    p_pos = (cnt_pos + alpha) / (tot_pos + alpha * V)
    p_neg = (cnt_neg + alpha) / (tot_neg + alpha * V)
    lo = np.log(p_pos) - np.log(p_neg)          # per-token log-odds (class1 vs class0)

    # center the scores (only relative scores matter for argmax over 2 classes)
    lo = lo - lo.mean()

    # token score vectors: s0_v = -lo_v/2 , s1_v = +lo_v/2  (so s1-s0 = lo)
    s0 = -0.5 * lo
    s1 = 0.5 * lo

    # ---- build two orthonormal zero-mean probe directions a,b in R^D ----
    # start from random, project out the all-ones direction, orthonormalise
    rng = np.random.RandomState(0)
    ones = np.ones(D, dtype=np.float64)
    P = np.eye(D) - np.outer(ones, ones) / D    # projector onto zero-mean subspace
    A = rng.randn(D, 2)
    A = P @ A
    # QR orthonormalise
    Q, _ = np.linalg.qr(A)
    a = Q[:, 0]
    b = Q[:, 1]
    a = a - a.mean()
    b = b - b.mean()
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    # re-orthogonalise b against a
    b = b - (b @ a) * a
    b = b / np.linalg.norm(b)

    # scale so that (s0_v/A)^2 + (s1_v/A)^2 <= 1 for all v
    maxnorm = np.sqrt(s0 ** 2 + s1 ** 2).max()
    A_scale = 1.5 * maxnorm + 1e-6

    # orthogonal filler direction (zero-mean, orthogonal to a and b)
    c = P @ rng.randn(D)
    c = c - (c @ a) * a - (c @ b) * b
    c = c / (np.linalg.norm(c) + 1e-12)

    # ---- construct embed rows: zero-mean vectors u_v with head_c . u_v = s^c_v/A_scale ----
    # u_v = (s0_v/A) a + (s1_v/A) b + fill_v * c   with |u_v| = 1
    U = np.zeros((V, D), dtype=np.float64)
    comp_a = s0 / A_scale
    comp_b = s1 / A_scale
    rem = 1.0 - comp_a ** 2 - comp_b ** 2
    rem = np.clip(rem, 0.0, None)
    fill = np.sqrt(rem)
    U = np.outer(comp_a, a) + np.outer(comp_b, b) + np.outer(fill, c)
    # ensure zero-mean (numerically)
    U = U - U.mean(axis=1, keepdims=True)
    # normalise each row to unit norm
    nrm = np.linalg.norm(U, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    U = U / nrm

    # embed[v] = U[v]  (zero-mean, unit norm) -> LN(embed[v]) = sqrt(D) * U[v]
    w["embed.weight"] = (U * 1.0).astype(np.float32)

    # head_c . LN(embed[v]) = head_c . (sqrt(D) U[v]) = s^c_v
    # => head_c . U[v] = s^c_v / sqrt(D)
    # with head_0 = A_scale * sqrt(D) * a , head_1 = A_scale * sqrt(D) * b :
    # head_0 . U[v] = A_scale*sqrt(D) * (s0_v/A_scale) = sqrt(D) s0_v  ->  LN dot = s0_v. good
    h0 = A_scale * np.sqrt(D) * a
    h1 = A_scale * np.sqrt(D) * b
    head = np.zeros((2, D), dtype=np.float64)
    head[0] = h0
    head[1] = h1
    w["head.weight"] = head.astype(np.float32)
    w["head.bias"] = np.zeros(2, dtype=np.float32)

    # keep everything else identity / zero (blocks, pos, norms already so)
    # explicitly zero the blocks and pos to be safe
    for k in list(w.keys()):
        if k.startswith("block0.") or k.startswith("block1."):
            if k.endswith(".weight") and ("norm" in k):
                w[k] = np.ones_like(w[k])
            elif k.endswith(".bias") and ("norm" in k):
                w[k] = np.zeros_like(w[k])
            else:
                w[k] = np.zeros_like(w[k])
    w["pos.weight"] = np.zeros_like(w["pos.weight"])
    for k in ("norm.weight",):
        w[k] = np.ones_like(w[k])
    w["norm.bias"] = np.zeros_like(w["norm.bias"])

    # ---- diagnostics: quick self-check of the induced token scores ----
    # recompute effective token scores head_c . LN(embed[v]) = sqrt(D)*head_c . U[v]
    eff0 = np.sqrt(D) * (U @ h0)
    eff1 = np.sqrt(D) * (U @ h1)
    # training accuracy of the induced BoW classifier (argmax over 2 classes)
    sc = np.zeros(N, dtype=np.float64)
    for i in range(N):
        ids = X[i][mask[i]]
        if ids.size == 0:
            continue
        sc[i] = (eff1[ids].sum() - eff0[ids].sum())
    pred = (sc > 0).astype(np.int64)
    train_acc = float((pred == y).mean())

    diag = {
        "train_acc_bow": train_acc,
        "score_std": float(lo.std()),
        "A_scale": float(A_scale),
        "head_norm": float(np.linalg.norm(head)),
        "embed_row_norm_mean": float(np.linalg.norm(U, axis=1).mean()),
        "n_tokens_scored": int((np.abs(lo) > 1e-9).sum()),
    }
    return {"weights": w, "diagnostics": diag}
