import numpy as np

def edit(w, f):
    X = np.asarray(f["x"])            # [N,64] int64
    y = np.asarray(f["y"]).astype(np.float64)  # [N]
    N, L = X.shape
    V = w["embed.weight"].shape[0]    # 2000
    d = w["embed.weight"].shape[1]    # 32

    # ---- build length-normalized bag-of-words matrix M [N, V] ----
    # token 0 is padding; exclude it from the mean
    M = np.zeros((N, V), dtype=np.float64)
    for i in range(N):
        row = X[i]
        nz = row[row != 0]
        if nz.size == 0:
            continue
        np.add.at(M[i], nz, 1.0 / nz.size)

    # ---- ridge regression on centered data (closed form) ----
    # y in {0,1}; target t = 2y-1 in {-1,+1}
    t = 2.0 * y - 1.0
    mbar = M.mean(axis=0)                 # [V]
    tbar = t.mean()
    Mc = M - mbar
    tc = t - tbar
    # choose lambda relative to feature scale
    # (mean-BOW entries ~ 1/seqlen; use a small relative ridge)
    lam = 1e-3 * (Mc.shape[0])            # scale-aware
    # Solve (Mc^T Mc + lam I) w = Mc^T tc
    G = Mc.T @ Mc                         # [V,V]
    G[np.diag_indices_from(G)] += lam
    rhs = Mc.T @ tc                       # [V]
    try:
        wv = np.linalg.solve(G, rhs)
    except np.linalg.LinAlgError:
        wv = np.linalg.lstsq(G, rhs, rcond=None)[0]
    b = tbar - wv @ mbar

    # ---- realize in the network ----
    # We need head[1] . LN(e_v) = w_v  (up to a global scale s)
    # LN output g_v must satisfy: sum_i g_v[i] = 0, ||g_v|| = sqrt(d)
    # Choose h = s * e_1 (first coordinate), and set column v of G so that
    #   G[0,v] = w_v / s ,  sum_{i>0} G[i,v] = -w_v/s ,  ||g_v||^2 = d
    # Feasible iff |w_v/s| <= sqrt(d).  Pick s accordingly.
    maxw = np.max(np.abs(wv)) + 1e-12
    s = maxw / (np.sqrt(d) * 0.999)       # ensures |w_v/s| < sqrt(d)
    a = wv / s                            # first-row entries, |a_v| <= sqrt(d)

    G = np.zeros((d, V), dtype=np.float64)
    G[0, :] = a
    # distribute the remaining mass on coordinate 1 so that column sums to 0
    # and column norm is sqrt(d):  need sum_{i>=1} G[i,v] = -a_v,
    # sum_{i>=1} G[i,v]^2 = d - a_v^2
    rem = d - a * a
    rem = np.maximum(rem, 1e-9)
    # put all remaining mass on coordinate 1
    G[1, :] = -a - np.sqrt(np.maximum(rem - (a*a), 0.0)) * 0.0  # placeholder
    # Better: use two coordinates to satisfy both sum and norm exactly.
    # Let p = -a (target sum), q = sqrt(d - a^2) (target norm of tail).
    # Set G[1]=u, G[2]=v with u+v=p, u^2+v^2=q^2 -> u,v = (p ± sqrt(2q^2-p^2))/2
    p = -a
    q2 = np.maximum(d - a * a, 1e-9)
    disc = np.maximum(2.0 * q2 - p * p, 0.0)
    sq = np.sqrt(disc)
    G[1, :] = (p + sq) * 0.5
    G[2, :] = (p - sq) * 0.5

    # verify column sums ~ 0 and norms ~ sqrt(d)
    colsum = G.sum(axis=0)
    colnorm = np.sqrt((G * G).sum(axis=0))
    # fix tiny numerical drift in sums by adjusting coordinate 1
    G[1, :] -= colsum

    # Now set embed rows: e_v = c * g_v + offset*ones  (offset irrelevant to LN)
    # LN(e_v) = sqrt(d) * g_v / ||g_v||  -> we want LN(e_v) = g_v exactly,
    # so set e_v = g_v (then LN gives sqrt(d)*g_v/||g_v|| = g_v if ||g_v||=sqrt(d))
    # Actually LN(x)= (x-mean)/sqrt(var+eps); with x=g_v, mean=0, var=1 -> LN=g_v.
    # Since sum(g_v)=0 and ||g_v||^2=d, var = d/d = 1.  Good.
    embed = G.T.astype(np.float32)        # [V, d]

    # head: logit1 - logit0 = 2 * (h . pooled) + (b1-b0)
    # pooled = mean_t g_{x_t} = G m  (m = mean-BOW)
    # h . pooled = h^T G m ; we want = wv . m  ->  h^T G = wv
    # With G[0,:]=a=wv/s and h = s*e_1:  h^T G = s * G[0,:] = s*a = wv.  Good.
    h = np.zeros(d, dtype=np.float64)
    h[0] = s
    # head.weight[1] = h, head.weight[0] = -h ; bias split so b1-b0 = 2b
    head_w = np.zeros((2, d), dtype=np.float64)
    head_w[1] = h
    head_w[0] = -h
    head_b = np.zeros(2, dtype=np.float64)
    head_b[1] = b
    head_b[0] = -b

    # ---- write weights (preserve names/shapes, float32, |.|<=1000) ----
    w["embed.weight"] = np.clip(embed, -1000.0, 1000.0).astype(np.float32)
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)
    for blk in ["block0", "block1"]:
        for sub in ["qkv", "proj", "ff1", "ff2"]:
            wk = f"{blk}.{sub}.weight"
            bk = f"{blk}.{sub}.bias"
            if wk in w:
                w[wk] = np.zeros_like(w[wk], dtype=np.float32)
            if bk in w:
                w[bk] = np.zeros_like(w[bk], dtype=np.float32)
        for sub in ["norm1", "norm2"]:
            wk = f"{blk}.{sub}.weight"
            bk = f"{blk}.{sub}.bias"
            if wk in w:
                w[wk] = np.ones_like(w[wk], dtype=np.float32)
            if bk in w:
                w[bk] = np.zeros_like(w[bk], dtype=np.float32)
    w["norm.weight"] = np.ones_like(w["norm.weight"], dtype=np.float32)
    w["norm.bias"] = np.zeros_like(w["norm.bias"], dtype=np.float32)
    w["head.weight"] = np.clip(head_w, -1000.0, 1000.0).astype(np.float32)
    w["head.bias"] = np.clip(head_b, -1000.0, 1000.0).astype(np.float32)

    # ---- diagnostics ----
    # realized logit diff on training set
    pooled = M @ G.T                       # [N, d]  (mean of g_v)
    realized = pooled @ h + (head_b[1] - head_b[0])  # = 2*(wv.m + b)
    pred = (realized > 0).astype(np.int64)
    acc = float((pred == y.astype(np.int64)).mean())
    # cross-entropy of realized logits
    logit2 = np.stack([-realized * 0.5, realized * 0.5], axis=1)
    logit2 = logit2 - logit2.max(axis=1, keepdims=True)
    pr = np.exp(logit2)
    pr = pr / pr.sum(axis=1, keepdims=True)
    ce = float(-np.log(np.clip(pr[np.arange(N), y.astype(np.int64)], 1e-12, 1.0)).mean())

    diag = {
        "n_samples": int(N),
        "vocab": int(V),
        "ridge_lambda": float(lam),
        "scale_s": float(s),
        "w_norm": float(np.linalg.norm(wv)),
        "w_max": float(maxw),
        "bias": float(b),
        "realization_colsum_max": float(np.max(np.abs(colsum))),
        "realization_colnorm_err": float(np.max(np.abs(colnorm - np.sqrt(d)))),
        "train_accuracy": acc,
        "train_ce": ce,
        "nonzero_token_weights": int(np.sum(np.abs(wv) > 1e-9)),
    }
    return {"weights": w, "diagnostics": diag}
