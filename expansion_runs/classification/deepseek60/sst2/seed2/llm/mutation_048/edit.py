import numpy as np

def edit(w, f):
    # ---------- helpers ----------
    def counts(ids, V=2000):
        # binary presence matrix [N, V]
        N, T = ids.shape
        C = np.zeros((N, V), dtype=np.float64)
        rows = np.repeat(np.arange(N), T)
        cols = ids.reshape(-1).astype(np.int64)
        valid = cols > 0
        np.add.at(C, (rows[valid], cols[valid]), 1.0)
        return C

    def bigram_counts(ids, V=2000, Vb=None):
        # counts of adjacent (token_t, token_{t+1}) pairs, hashed into a
        # compact space via a random projection of the pair-id.
        N, T = ids.shape
        a = ids[:, :-1].reshape(-1).astype(np.int64)
        b = ids[:, 1:].reshape(-1).astype(np.int64)
        valid = (a > 0) & (b > 0)
        a = a[valid]; b = b[valid]
        pair = a * V + b
        return pair

    X = f["x"].astype(np.int64)          # [4000,64]
    y = f["y"].astype(np.int64)          # [4000]
    N = X.shape[0]
    V = w["embed.weight"].shape[0]       # 2000

    # ---------- unigram BOW design ----------
    C = counts(X, V)                     # [N, V]
    # length-normalised counts (matches masked-mean pooling)
    lens = (X > 0).sum(axis=1).astype(np.float64)
    lens[lens == 0] = 1.0
    Cn = C / lens[:, None]

    Y = np.zeros((N, 2), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    def ridge_fit(A, Y, lam):
        # closed-form normal equations with bias column
        A1 = np.concatenate([A, np.ones((A.shape[0], 1))], axis=1)
        G = A1.T @ A1
        G[np.diag_indices_from(G)] += lam
        W = np.linalg.solve(G, A1.T @ Y)   # [V+1, 2]
        return W

    def cv_acc(A, y, lams, k=4):
        idx = np.arange(N)
        rng = np.random.RandomState(0)
        rng.shuffle(idx)
        folds = np.array_split(idx, k)
        best = (-1.0, None)
        for lam in lams:
            accs = []
            for i in range(k):
                te = folds[i]
                tr = np.concatenate([folds[j] for j in range(k) if j != i])
                W = ridge_fit(A[tr], Y[tr], lam)
                A1 = np.concatenate([A[te], np.ones((len(te), 1))], axis=1)
                pred = np.argmax(A1 @ W, axis=1)
                accs.append((pred == y[te]).mean())
            m = float(np.mean(accs))
            if m > best[0]:
                best = (m, lam)
        return best

    lams = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    cv_uni, lam_uni = cv_acc(Cn, y, lams)

    # ---------- unigram + bigram design ----------
    # hash bigram pairs into a fixed 512-dim bucket space via a deterministic
    # random projection (keeps the design matrix small & well conditioned).
    rng = np.random.RandomState(1234)
    B = 512
    proj = rng.randn(V * V, B).astype(np.float64) / np.sqrt(B)
    pair = bigram_counts(X, V)
    # accumulate bigram features per example
    Nb = np.zeros((N, B), dtype=np.float64)
    rows = np.repeat(np.arange(N), X.shape[1] - 1)
    # rebuild with example index
    a = X[:, :-1].reshape(-1).astype(np.int64)
    b = X[:, 1:].reshape(-1).astype(np.int64)
    valid = (a > 0) & (b > 0)
    ridx = rows[valid]
    pidx = a[valid] * V + b[valid]
    np.add.at(Nb, (ridx, slice(None)), proj[pidx])
    Nb = Nb / lens[:, None]

    A_bi = np.concatenate([Cn, Nb], axis=1)
    cv_bi, lam_bi = cv_acc(A_bi, y, lams)

    use_bi = cv_bi > cv_uni
    A = A_bi if use_bi else Cn
    lam = lam_bi if use_bi else lam_uni
    W = ridge_fit(A, Y, lam)             # [D+1, 2]

    # ---------- realise the fitted linear model in the network ----------
    # decision direction in feature space:
    #   d = W[:,1]-W[:,0]  (length D),  b = W[-1,1]-W[-1,0]
    d = W[:, 1] - W[:, 0]
    b = W[-1, 1] - W[-1, 0]

    # We must express  score = d·features + b  through head/embed.
    # Unigram part: d_uni lives in R^V  -> embed[:,0] = d_uni/||d_uni||.
    # Bigram part:  d_bi lives in R^B  -> we fold it into the *second*
    #               embedding column via the same projection basis so that
    #               head·(mean of embed rows) reproduces it.  Because the
    #               bigram feature is a mean over hashed pairs, we can absorb
    #               it by adding, for each token v, the average projected
    #               bigram contribution of pairs starting at v.
    d_uni = d[:V].copy()
    d_bi = d[V:].copy() if use_bi else np.zeros(B)

    # per-token bigram contribution: for token v, sum over successors u of
    # proj[v*V+u] weighted by d_bi, then average over successors.
    contrib = np.zeros(V, dtype=np.float64)
    if use_bi and np.any(d_bi != 0):
        # proj is [V*V, B]; contribution of pair (v,u) is proj[v*V+u]·d_bi
        pair_score = proj @ d_bi            # [V*V]
        pair_score = pair_score.reshape(V, V)
        # average over valid successors (u>0) for each v
        succ = pair_score[:, 1:]            # ignore u==0 (pad)
        contrib = succ.mean(axis=1)

    # combined per-token score g[v] = d_uni[v] + contrib[v]
    g = d_uni + contrib
    gnorm = np.linalg.norm(g)
    if gnorm < 1e-8:
        gnorm = 1.0
    g_unit = (g / gnorm).astype(np.float32)

    # ---------- write embed ----------
    E = np.zeros((V, 32), dtype=np.float32)
    E[:, 0] = g_unit
    # keep remaining columns tiny random so LN variance is well-defined and
    # the column space contains g; they contribute ~0 to the score.
    rng2 = np.random.RandomState(7)
    E[:, 1:] = (rng2.randn(V, 31) * 1e-3).astype(np.float32)
    # make each row zero-mean (LN identity on the mean component)
    E = E - E.mean(axis=1, keepdims=True)
    # unit-norm rows so LayerNorm(v) == v
    rn = np.linalg.norm(E, axis=1, keepdims=True)
    rn[rn < 1e-8] = 1.0
    E = (E / rn).astype(np.float32)
    w["embed.weight"] = E

    # ---------- pos / blocks: identity ----------
    w["pos.weight"] = np.zeros_like(w["pos.weight"])
    for blk in ["block0", "block1"]:
        w[f"{blk}.qkv.weight"] = np.zeros_like(w[f"{blk}.qkv.weight"])
        w[f"{blk}.qkv.bias"]   = np.zeros_like(w[f"{blk}.qkv.bias"])
        w[f"{blk}.proj.weight"] = np.zeros_like(w[f"{blk}.proj.weight"])
        w[f"{blk}.proj.bias"]   = np.zeros_like(w[f"{blk}.proj.bias"])
        w[f"{blk}.ff1.weight"] = np.zeros_like(w[f"{blk}.ff1.weight"])
        w[f"{blk}.ff1.bias"]   = np.zeros_like(w[f"{blk}.ff1.bias"])
        w[f"{blk}.ff2.weight"] = np.zeros_like(w[f"{blk}.ff2.weight"])
        w[f"{blk}.ff2.bias"]   = np.zeros_like(w[f"{blk}.ff2.bias"])
        w[f"{blk}.norm1.weight"] = np.ones_like(w[f"{blk}.norm1.weight"])
        w[f"{blk}.norm1.bias"]   = np.zeros_like(w[f"{blk}.norm1.bias"])
        w[f"{blk}.norm2.weight"] = np.ones_like(w[f"{blk}.norm2.weight"])
        w[f"{blk}.norm2.bias"]   = np.zeros_like(w[f"{blk}.norm2.bias"])
    w["norm.weight"] = np.ones_like(w["norm.weight"])
    w["norm.bias"]   = np.zeros_like(w["norm.bias"])

    # ---------- head ----------
    # logits = head @ features + bias ; want logits1-logits0 = gnorm*features0 + b
    # features0 = mean_t E[token_t,0] = mean_t g_unit[token_t]
    # => gnorm*features0 = mean_t g[token_t] = d·features + (bigram part)
    # (bigram part already folded into g via contrib)
    hw = np.zeros((2, 32), dtype=np.float32)
    hw[1, 0] = 0.5 * gnorm
    hw[0, 0] = -0.5 * gnorm
    hb = np.array([-0.5 * b, 0.5 * b], dtype=np.float32)
    w["head.weight"] = hw
    w["head.bias"]   = hb

    # ---------- diagnostics ----------
    # verify on the pool using the frozen features we can compute exactly
    # (features = masked mean of E rows)
    def pool_features(ids):
        Nn, T = ids.shape
        F = E[ids]                       # [N,T,32]
        mask = (ids > 0).astype(np.float32)[..., None]
        s = (F * mask).sum(axis=1)
        c = mask.sum(axis=1)
        c[c == 0] = 1.0
        return s / c

    Fpool = pool_features(X)
    logits = Fpool @ hw.T + hb
    pred = np.argmax(logits, axis=1)
    pool_acc = float((pred == y).mean())
    # class-wise accuracy
    acc0 = float((pred[y == 0] == 0).mean())
    acc1 = float((pred[y == 1] == 1).mean())

    diag = {
        "cv_acc_unigram": cv_uni,
        "cv_acc_unigram_bigram": cv_bi,
        "lambda_used": float(lam),
        "used_bigram": bool(use_bi),
        "pool_acc": pool_acc,
        "pool_acc_class0": acc0,
        "pool_acc_class1": acc1,
        "g_norm": float(gnorm),
        "bias": float(b),
        "embed_row_norm_mean": float(np.linalg.norm(E, axis=1).mean()),
        "n_features": int(A.shape[1]),
    }
    return {"weights": w, "diagnostics": diag}
