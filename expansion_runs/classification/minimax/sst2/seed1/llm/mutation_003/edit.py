def edit(w, f):
    import numpy as np

    # ---------------- data ----------------
    x = np.asarray(f["x"], dtype=np.int64)          # [N,64]
    y = np.asarray(f["y"], dtype=np.int64)          # [N]
    N, L = x.shape
    V = w["embed.weight"].shape[0]                  # 2000
    t = (2.0 * y.astype(np.float64) - 1.0)          # +-1 targets

    mask = (x != 0)
    lens = mask.sum(1).astype(np.float64)
    keep = lens > 0

    # ---------------- feature construction ----------------
    # unigram binary presence
    Xb = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), lens.astype(np.int64))
    cols = x[mask]
    Xb[rows, cols] = 1.0

    # bigram binary presence (adjacent non-pad token pairs)
    big_ids = {}
    big_rows = []
    big_cols = []
    for i in range(N):
        seq = x[i]
        Ls = int(lens[i])
        if Ls >= 2:
            a = seq[:Ls - 1]
            b = seq[1:Ls]
            for u, v in zip(a.tolist(), b.tolist()):
                key = (u, v)
                j = big_ids.get(key)
                if j is None:
                    j = len(big_ids)
                    big_ids[key] = j
                big_rows.append(i)
                big_cols.append(j)
    nb = len(big_ids)
    Xg = np.zeros((N, nb), dtype=np.float64)
    if nb:
        Xg[np.asarray(big_rows, dtype=np.int64), np.asarray(big_cols, dtype=np.int64)] = 1.0

    # ---------------- NBSVM log-count ratio ----------------
    def lcr(F):
        cp = 1.0 + F[y == 1].sum(0)
        cn = 1.0 + F[y == 0].sum(0)
        return np.log(cp) - np.log(cn)

    def build(F, use_lcr):
        if not use_lcr:
            return F
        r = lcr(F)
        return F * r

    def ridge_fit(Z, lam):
        # closed-form ridge with intercept, on +-1 targets
        m = Z.shape[1]
        Zc = Z - Z.mean(0, keepdims=True)
        tm = t.mean()
        A = Zc.T @ Zc
        A[np.diag_indices(m)] += lam
        b = Zc.T @ (t - tm)
        try:
            wi = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            wi = np.linalg.lstsq(A, b, rcond=None)[0]
        intercept = tm - Z.mean(0) @ wi
        return wi, intercept

    def acc(Z, wi, b0):
        s = Z @ wi + b0
        return float(np.mean((s > 0).astype(np.int64) == y))

    def loo_acc(Z, lam):
        # closed-form leave-one-out for ridge (hat matrix diagonal)
        m = Z.shape[1]
        Zc = Z - Z.mean(0, keepdims=True)
        tm = t.mean()
        A = Zc.T @ Zc
        A[np.diag_indices(m)] += lam
        try:
            Ai = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            return 0.0, None, None
        H = Zc @ Ai @ Zc.T
        h = np.clip(np.diag(H), 0.0, 0.999)
        wi = Ai @ (Zc.T @ (t - tm))
        b0 = tm - Z.mean(0) @ wi
        s = Z @ wi + b0
        resid = t - s
        loo = (resid / (1.0 - h))
        pred = (loo > 0).astype(np.int64)
        return float(np.mean(pred == y)), wi, b0

    # ---------------- model selection (closed-form fits only) ----------------
    # candidate feature sets / lambdas; each candidate is a single normal-equation solve
    cands = []
    for use_lcr in (True, False):
        for use_bigram in (False, True):
            if use_bigram:
                F = np.hstack([Xb, Xg])
            else:
                F = Xb
            Z = build(F, use_lcr)
            for lam in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0):
                a_loo, wi, b0 = loo_acc(Z, lam)
                if wi is None:
                    continue
                a_in = acc(Z, wi, b0)
                cands.append((a_loo, a_in, use_lcr, use_bigram, lam, Z, wi, b0))

    cands.sort(key=lambda c: (-c[0], -c[1]))
    best = cands[0]
    a_loo, a_in, use_lcr, use_bigram, lam, Z, wi, b0 = best

    # ---------------- map ridge solution back to per-token scores ----------------
    m = Z.shape[1]
    # per-feature weight (for unigram part) and per-bigram weight
    if use_bigram:
        wu = wi[:V]
        wg = wi[V:]
    else:
        wu = wi
        wg = None

    # token-level score: use the fitted linear model value per token occurrence.
    # For unigram part: contribution of token t to the score = wu[t] (plus intercept spread)
    # For bigram part: contribution of bigram (a,b) = wg[big]
    # We encode per-token scores s_t so that mean_t s_t reproduces the decision:
    #   score(example) = sum_t wu[t] + sum_bigram wg[big] + b0
    # We use s_t = wu[t] + c  (c = per-token constant) so that
    #   mean_t s_t = (1/T) sum_t wu[t] + c
    # and bigram contribution is folded in via a per-token constant too:
    #   c_t = c0 + (1/T)*sum of bigram weights involving token t  -> depends on T,
    # which is not expressible; instead we drop bigram weights from the encoding
    # and only use them for the (reported) model-selection statistics.
    # Safer: rebuild the encoding from the UNIGRAM-only part of the chosen model.

    # Recompute a clean unigram-only ridge model (with LCR) for the encoding,
    # using the same lambda scaled to the feature count.
    Zu = build(Xb, use_lcr)
    lam_u = lam * (Zu.shape[1] / float(max(1, m))) if m else lam
    lam_u = max(lam_u, 1e-3)
    a_loo_u, wu_u, b0_u = loo_acc(Zu, lam_u)
    if wu_u is None:
        wu_u, b0_u = ridge_fit(Zu, lam_u)

    # per-token raw weight (undo the LCR scaling so the score is in "feature space")
    if use_lcr:
        r_u = lcr(Xb)
        raw_u = wu_u / np.where(np.abs(r_u) > 1e-9, r_u, 1.0)
    else:
        raw_u = wu_u

    # per-token constant c (length-aware intercept): choose so that the pooled
    # score threshold matches the model's own intercept.
    # pooled0 = mean_t (raw_u[t] + c) ; decision sign(alpha*(pooled0))
    # model decision: sum_t raw_u[t] + b0_u  ->  mean_t raw_u + b0_u/T
    # we approximate b0_u/T by c (constant) -> c = b0_u (per-token constant)
    c = float(b0_u)

    # ---------------- encode token scores into embeddings ----------------
    # s_t = raw_u[t] + c ; scale so that |s| stays inside the LayerNorm-safe range
    s = raw_u + c
    smax = np.max(np.abs(s)) if s.size else 0.0
    scale = 1.0
    if smax > 3.0:
        scale = 3.0 / smax
    s = s * scale

    # v_t: channel0 = s_t ; channels 1..31 compensate so mean=0 and sumsq=32
    # comp = -s/31 * ones(31) + t * d , d = (1,-1,0..0)/sqrt2
    # sumsq(comp) = s^2/31 + t^2  -> t^2 = 32 - (32/31) s^2
    s2 = s * s
    t2 = 32.0 - (32.0 / 31.0) * s2
    t2 = np.maximum(t2, 0.0)
    tt = np.sqrt(t2)
    Vt = np.zeros((V, 32), dtype=np.float64)
    Vt[:, 0] = s
    base = -s / 31.0
    Vt[:, 1] = base + tt / np.sqrt(2.0)
    Vt[:, 2] = base - tt / np.sqrt(2.0)
    Vt[:, 3:] = base[:, None]

    # numerical hygiene: exact zero mean / unit sumsq where possible
    Vt -= Vt.mean(axis=1, keepdims=True)
    nrm = np.sqrt((Vt * Vt).sum(axis=1, keepdims=True))
    nrm[nrm < 1e-12] = 1.0
    Vt = Vt / nrm * np.sqrt(32.0)

    E = np.array(w["embed.weight"], dtype=np.float64)
    E[:] = 0.0
    E[1:] = Vt[1:]           # token id 0 (PAD) stays zero
    E = np.clip(E, -1000.0, 1000.0)
    w["embed.weight"] = E.astype(np.float32)

    # ---------------- zero the transformer blocks (identity residual) -------
    for b in ("block0", "block1"):
        w[b + ".proj.weight"] = np.zeros_like(w[b + ".proj.weight"])
        w[b + ".proj.bias"] = np.zeros_like(w[b + ".proj.bias"])
        w[b + ".ff1.weight"] = np.zeros_like(w[b + ".ff1.weight"])
        w[b + ".ff1.bias"] = np.zeros_like(w[b + ".ff1.bias"])
        w[b + ".ff2.weight"] = np.zeros_like(w[b + ".ff2.weight"])
        w[b + ".ff2.bias"] = np.zeros_like(w[b + ".ff2.bias"])
        # qkv left as-is (its output is annihilated by the zero proj)

    # ---------------- positional embedding -> zero ----------------------------
    w["pos.weight"] = np.zeros_like(w["pos.weight"])

    # ---------------- final LayerNorm = identity ----------------------------
    w["norm.weight"] = np.ones_like(w["norm.weight"])
    w["norm.bias"] = np.zeros_like(w["norm.bias"])

    # ---------------- head: read pooled channel 0 only ----------------------
    alpha = 1.0
    HW = np.zeros((2, 32), dtype=np.float64)
    HW[0, 0] = -alpha
    HW[1, 0] = alpha
    HB = np.zeros((2,), dtype=np.float64)
    w["head.weight"] = HW.astype(np.float32)
    w["head.bias"] = HB.astype(np.float32)

    # ---------------- diagnostics -------------------------------------------
    # simulate the exact forward pass of the edited network on the feedback pool
    def simulate(X):
        m_ = (X != 0)
        L_ = m_.sum(1)
        H = E[X]                       # [N,64,32]
        H = H * m_[..., None]
        mu = H.mean(-1, keepdims=True)
        var = ((H - mu) ** 2).mean(-1, keepdims=True)
        LN = (H - mu) / np.sqrt(var + 1e-5)
        LN = LN * 1.0 + 0.0
        cnt = np.maximum(L_, 1)[:, None]
        pooled = (LN * m_[..., None]).sum(1) / cnt
        lg = pooled @ HW.T + HB
        return lg

    lg = simulate(x)
    pred = np.argmax(lg, axis=1)
    acc_sim = float(np.mean(pred == y))
    p1 = np.exp(lg[:, 1] - lg[:, 0])
    ce = float(np.mean(np.log(p1) * y + np.log(1.0 - p1) * (1 - y)))
    sgn = np.sign(lg[:, 1] - lg[:, 0])
    acc_sign = float(np.mean((sgn > 0).astype(np.int64) == y))

    diag = {
        "design": "exact-linear BoW readout: LayerNorm-identity embeddings (mean=0, sumsq=32) + zeroed blocks + head on pooled channel0",
        "selected_use_lcr": bool(use_lcr),
        "selected_use_bigram": bool(use_bigram),
        "selected_lambda": float(lam),
        "selected_loo_acc": float(a_loo),
        "selected_insample_acc": float(a_in),
        "n_candidates": len(cands),
        "top5_loo": [[round(c[0], 4), round(c[1], 4), bool(c[2]), bool(c[3]), float(c[4])] for c in cands[:5]],
        "encoding_lambda": float(lam_u),
        "encoding_loo_acc": float(a_loo_u),
        "n_bigram_types": int(nb),
        "mean_len": float(lens.mean()),
        "unk_token_frac": float(np.mean(x[mask] == 1)),
        "score_scale": float(scale),
        "s_absmax": float(np.max(np.abs(s))),
        "sim_acc": round(acc_sim, 4),
        "sim_acc_sign": round(acc_sign, 4),
        "sim_cross_entropy": round(ce, 4),
        "embed_absmax": float(np.max(np.abs(E))),
        "head_norm": float(np.linalg.norm(HW)),
    }
    return {"weights": w, "diagnostics": diag}
