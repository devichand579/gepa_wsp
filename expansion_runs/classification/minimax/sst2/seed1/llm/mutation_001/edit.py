def edit(w, f):
    import numpy as np

    x = np.asarray(f["x"]).astype(np.int64)          # [N,64] token ids
    y = np.asarray(f["y"]).astype(np.int64)          # [N]
    V = w["embed.weight"].shape[0]                   # 2000
    D = w["embed.weight"].shape[1]                   # 32

    # ---------------- 1. token class statistics from the permitted pool ----------------
    mask = (x != 0)
    npos = np.zeros(V, dtype=np.float64)
    nneg = np.zeros(V, dtype=np.float64)
    np.add.at(npos, x[mask & (y[:, None] == 1)], 1.0)
    np.add.at(nneg, x[mask & (y[:, None] == 0)], 1.0)

    alpha = 1.0
    P = (npos + alpha) / (npos.sum() + alpha * V)
    Q = (nneg + alpha) / (nneg.sum() + alpha * V)
    s = np.log(P / Q)                                # token log-odds, sign = sentiment
    s[0] = 0.0                                      # PAD carries no signal
    # shrink rare tokens toward 0 (variance control)
    cnt = npos + nneg
    s *= (cnt / (cnt + 2.0))

    # ---------------- 2. encoding: v = c * s|s| * e , e = (1,-1,0,...,0) --------------
    e = np.zeros(D, dtype=np.float64)
    e[0] = 1.0
    e[1] = -1.0
    c = 1.0
    E = np.zeros((V, D), dtype=np.float64)
    E[:] = (c * s * np.abs(s))[:, None] * e[None, :]  # [V,32]
    # tiny orthogonal jitter so no channel is exactly degenerate
    rng = np.random.default_rng(0)
    E += 1e-4 * rng.standard_normal((V, D))

    # ---------------- 3. make the trunk an identity for the embedding ----------------
    P0 = np.zeros((D, D), dtype=np.float64)          # proj.weight
    pb0 = np.zeros(D, dtype=np.float64)              # proj.bias
    qw0 = np.zeros((3 * D, D), dtype=np.float64)      # qkv.weight
    qb0 = np.zeros(3 * D, dtype=np.float64)          # qkv.bias
    f1w0 = np.zeros((2 * D, D), dtype=np.float64)     # ff1.weight
    f1b0 = np.zeros(2 * D, dtype=np.float64)          # ff1.bias
    f2w0 = np.zeros((D, 2 * D), dtype=np.float64)      # ff2.weight
    f2b0 = np.zeros(D, dtype=np.float64)              # ff2.bias
    pos0 = np.zeros_like(w["pos.weight"].astype(np.float64))

    nw = np.ones(D, dtype=np.float64)                 # final norm affine
    nb = np.zeros(D, dtype=np.float64)

    # ---------------- 4. head: read the pooled sentiment channel -------------------
    # pooled = (1/L) * sum_t 4*s_t*e  ->  head.weight[1]-head.weight[0] must give 2*sum s_t
    headW = np.zeros((2, D), dtype=np.float64)
    headW[0, 0] = -1.0
    headW[1, 0] = 1.0
    headb = np.zeros(2, dtype=np.float64)

    # ---------------- 5. write back, preserving names/shapes/dtype -------------------
    def put(name, arr):
        cur = w[name]
        w[name] = np.ascontiguousarray(arr, dtype=np.float32)

    put("embed.weight", E)
    put("pos.weight", pos0)
    put("block0.norm1.weight", np.ones(D)); put("block0.norm1.bias", np.zeros(D))
    put("block0.qkv.weight", qw0);    put("block0.qkv.bias", qb0)
    put("block0.proj.weight", P0);   put("block0.proj.bias", pb0)
    put("block0.norm2.weight", np.ones(D)); put("block0.norm2.bias", np.zeros(D))
    put("block0.ff1.weight", f1w0);   put("block0.ff1.bias", f1b0)
    put("block0.ff2.weight", f2w0);   put("block0.ff2.bias", f2b0)
    put("block1.norm1.weight", np.ones(D)); put("block1.norm1.bias", np.zeros(D))
    put("block1.qkv.weight", qw0);    put("block1.qkv.bias", qb0)
    put("block1.proj.weight", P0);   put("block1.proj.bias", pb0)
    put("block1.norm2.weight", np.ones(D)); put("block1.norm2.bias", np.zeros(D))
    put("block1.ff1.weight", f1w0);   put("block1.ff1.bias", f1b0)
    put("block1.ff2.weight", f2w0);   put("block1.ff2.bias", f2b0)
    put("norm.weight", nw); put("norm.bias", nb)
    put("head.weight", headW); put("head.bias", headb)

    # ---------------- 6. diagnostics: closed-form BoW score accuracy ------------------
    L = mask.sum(1).astype(np.float64)
    ssum = (s[x] * mask).sum(1)
    score = ssum
    pred = (score > 0).astype(np.int64)
    acc = float((pred == y).mean())
    # 2-fold style check: fit on half, score the other half
    n = x.shape[0]
    idx = np.arange(n)
    h1 = idx % 2 == 0
    def fit_eval(tr, te):
        np_ = np.zeros(V); nn_ = np.zeros(V)
        np.add.at(np_, x[tr][mask[tr]], 1.0)
        np.add.at(nn_, x[tr][mask[tr] & (y[tr][:, None] == 0)], 1.0)
        return None
    # simple split evaluation
    def eval_split(tr, te):
        pos_c = np.zeros(V); neg_c = np.zeros(V)
        mtr = mask[tr]
        np.add.at(pos_c, x[tr][mtr & (y[tr][:, None] == 1)], 1.0)
        np.add.at(neg_c, x[tr][mtr & (y[tr][:, None] == 0)], 1.0)
        ss = np.log((pos_c + 1.0) / (pos_c.sum() + V)) - np.log((neg_c + 1.0) / (neg_c.sum() + V))
        ss[0] = 0.0
        cc = pos_c + neg_c
        ss *= cc / (cc + 2.0)
        sc = (ss[x[te]] * mask[te]).sum(1)
        return float(((sc > 0).astype(np.int64) == y[te]).mean())
    acc_e = eval_split(h1, ~h1)
    acc_o = eval_split(~h1, h1)
    allpad = int((L == 0).sum())
    ntok = int(mask.sum())
    uniq = int((cnt > 0).sum())

    diag = {
        "design": "BoW log-odds encoded via v=c*s|s|*e (e=(1,-1,0..)); blocks zeroed to identity; head reads pooled channel",
        "bow_acc_insample": acc,
        "bow_acc_split_even": acc_e,
        "bow_acc_split_odd": acc_o,
        "vocab_used": uniq,
        "tokens_total": ntok,
        "allpad_rows": allpad,
        "mean_len": float(L.mean()),
        "s_absmax": float(np.abs(s).max()),
        "s_absmean_nonzero": float(np.abs(s[cnt > 0]).mean()),
        "embed_abs_max": float(np.abs(E).max()),
        "head_norm": float(np.abs(headW).max()),
        "note": "pooled = mean over non-pad tokens of LayerNorm(v); LayerNorm(v)=4*s*e exactly, so logit diff = 2*sum_t s_t"
    }
    return {"weights": w, "diagnostics": diag}
