import numpy as np

def edit(w, f):
    x = np.asarray(f["x"]).astype(np.int64)          # [N,64]
    y = np.asarray(f["y"]).astype(np.int64)          # [N]
    N, L = x.shape
    V = w["embed.weight"].shape[0]                   # 2000
    D = w["embed.weight"].shape[1]                   # 32

    # ---------- 1. length-normalised unigram features ----------
    mask = (x != 0)                                  # [N,64]
    length = np.maximum(mask.sum(axis=1), 1).astype(np.float64)   # [N]
    X = np.zeros((N, V), dtype=np.float64)
    for t in range(L):
        col = x[:, t]
        valid = mask[:, t]
        np.add.at(X, (np.arange(N)[valid], col[valid]), 1.0)
    X = X / length[:, None]                          # mean-count features

    # ---------- 2. current per-token scores (from the live weights) ----------
    E = np.asarray(w["embed.weight"], dtype=np.float64)           # [V,32]
    hw = np.asarray(w["head.weight"], dtype=np.float64)           # [2,32]
    hb = np.asarray(w["head.bias"], dtype=np.float64)             # [2]
    hd = hw[1] - hw[0]                                            # [32]
    mu = E.mean(axis=1, keepdims=True)
    var = E.var(axis=1, keepdims=True)
    En = (E - mu) / np.sqrt(var + 1e-5)
    s_cur = En @ hd                                               # [V]
    b_cur = hb[1] - hb[0]

    # ---------- 3. closed-form ridge on the counts ----------
    t = np.where(y == 1, 1.0, -1.0)                  # ±1 targets
    # centre features so the intercept is handled separately
    Xm = X.mean(axis=0)
    Xc = X - Xm
    tm = t.mean()
    tc = t - tm
    G = Xc.T @ Xc                                    # [V,V] Gram
    # solve (G + alpha I) s = Xc.T tc  for several alphas, pick by CV
    best = None
    idx = np.arange(N)
    rng = np.random.RandomState(0)
    perm = rng.permutation(N)
    folds = np.array_split(perm, 5)
    for alpha in (0.3, 1.0, 3.0, 10.0, 30.0):
        A = G + alpha * np.eye(V)
        s = np.linalg.solve(A, Xc.T @ tc)
        b = tm - Xm @ s
        # 5-fold CV accuracy
        accs = []
        for k in range(5):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(5) if j != k])
            Gtr = Xc[tr].T @ Xc[tr]
            str_ = np.linalg.solve(Gtr + alpha * np.eye(V), Xc[tr].T @ tc[tr])
            btr = tc[tr].mean() - Xc[tr].mean(axis=0) @ str_
            pred = np.sign(Xc[te] @ str_ + btr + tc[tr].mean())
            pred[pred == 0] = 1.0
            accs.append((pred == t[te]).mean())
        cv = float(np.mean(accs))
        if best is None or cv > best[0]:
            best = (cv, alpha, s, b)
    cv, alpha, s_ridge, b_ridge = best

    # ---------- 4. blend with the current scores, then sparsify ----------
    # normalise both score vectors to comparable scale
    def _norm(v):
        sd = v.std()
        return v / sd if sd > 1e-12 else v
    s_r = _norm(s_ridge)
    s_c = _norm(s_cur)
    lam = 0.5                                        # blend weight toward ridge
    s = lam * s_r + (1.0 - lam) * s_c
    b = lam * (b_ridge / (s_ridge.std() + 1e-12)) + (1.0 - lam) * (b_cur / (s_cur.std() + 1e-12))

    # keep only the strongest K tokens (sparse, like the original 125)
    K = 400
    order = np.argsort(-np.abs(s))
    keep = order[:K]
    s_sparse = np.zeros(V, dtype=np.float64)
    s_sparse[keep] = s[keep]

    # ---------- 5. exact encoding into embed / head ----------
    # We want head_diff @ LayerNorm(embed[v]) = s(v) + b_const
    # Choose head_diff = c * e0, and embed[v] with mean 0, var 1, embed[v][0] = s(v)/c.
    # Then LayerNorm(embed[v]) = embed[v] (mean 0, var 1), so head_diff @ = c*embed[v][0] = s(v).
    smax = np.max(np.abs(s_sparse))
    c = 1.0
    if smax > 0:
        c = smax / 0.9                               # keep |embed[v][0]| <= 0.9 (feasible)
    g = np.zeros((V, D), dtype=np.float64)
    g[:, 0] = s_sparse / c
    # fill remaining dims so each row has mean 0 and variance 1
    # put the residual mass symmetrically on dims 1..D-1
    rem = -g[:, 0]                                   # need sum of dims 1.. = -g[:,0]
    sq = 1.0 - g[:, 0] ** 2                          # need sum of squares of dims 1.. = 1 - g0^2
    m = D - 1
    # distribute equally: each of dims 1..m gets value a, then m*a = rem, m*a^2 = sq
    # solve for a consistently: a = rem/m, and check m*a^2 vs sq; adjust with a small extra
    a = rem / m
    # correct the squared-sum by adding a scaled orthogonal-ish component
    # use two values: first dim gets a + d, rest get a - d/(m-1) to keep sum fixed
    if m >= 2:
        # sum fixed at rem; want sum of squares = sq
        # with values (a+d) and (a - d/(m-1)) repeated (m-1) times
        # sum of squares = (a+d)^2 + (m-1)(a - d/(m-1))^2
        # = a^2 + 2ad + d^2 + (m-1)a^2 - 2ad + d^2/(m-1)
        # = m a^2 + d^2 (1 + 1/(m-1)) = m a^2 + d^2 * m/(m-1)
        need = sq - m * a ** 2
        d2 = need * (m - 1) / m
        d2 = max(d2, 0.0)
        d = np.sqrt(d2)
        g[:, 1] = a + d
        g[:, 2:] = a[:, None] - (d / (m - 1))[:, None]
    else:
        g[:, 1] = rem

    # LayerNorm of g: mean 0, var 1 -> identity.  head_diff = c * e0
    new_embed = g.astype(np.float32)
    new_head_w = np.zeros((2, D), dtype=np.float32)
    new_head_w[1, 0] = c
    new_head_w[0, 0] = 0.0
    new_head_b = np.zeros(2, dtype=np.float32)
    new_head_b[1] = b * c                            # intercept
    new_head_b[0] = 0.0

    # ---------- 6. write back ----------
    w["embed.weight"] = new_embed
    w["pos.weight"] = np.zeros_like(np.asarray(w["pos.weight"], dtype=np.float32))
    for name in ("block0", "block1"):
        for sub in ("qkv", "proj", "ff1", "ff2"):
            w[f"{name}.{sub}.weight"] = np.zeros_like(np.asarray(w[f"{name}.{sub}.weight"], dtype=np.float32))
            w[f"{name}.{sub}.bias"] = np.zeros_like(np.asarray(w[f"{name}.{sub}.bias"], dtype=np.float32))
    for name in ("block0.norm1", "block0.norm2", "block1.norm1", "block1.norm2", "norm"):
        w[f"{name}.weight"] = np.ones_like(np.asarray(w[f"{name}.weight"], dtype=np.float32))
        w[f"{name}.bias"] = np.zeros_like(np.asarray(w[f"{name}.bias"], dtype=np.float32))
    w["head.weight"] = new_head_w
    w["head.bias"] = new_head_b

    # ---------- diagnostics ----------
    diag = {
        "cv_acc": float(cv),
        "alpha": float(alpha),
        "K": int(K),
        "s_absmax": float(np.max(np.abs(s_sparse))),
        "c": float(c),
        "blend_lambda": float(lam),
        "n_nonzero_tokens": int(np.count_nonzero(s_sparse)),
        "head_bias": float(new_head_b[1]),
    }
    return {"weights": w, "diagnostics": diag}
