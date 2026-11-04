def edit(w, f):
    import numpy as np

    # ---------------- data ----------------
    X = np.asarray(f["x"], dtype=np.int64)          # [N,64] token ids (0 = pad)
    Y = np.asarray(f["y"], dtype=np.int64)          # [N] labels 0/1
    N, L = X.shape
    V = 2000
    D = 32

    mask = (X != 0)
    lens = mask.sum(1).astype(np.float64)
    lens = np.maximum(lens, 1.0)

    # bag of words counts (exclude pad)
    Cnt = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), lens.astype(np.int64))
    cols = X[mask]
    np.add.at(Cnt, (rows, cols), 1.0)
    Cnt /= lens[:, None]                            # mean bag-of-words features

    y = Y.astype(np.float64)
    t = (2.0 * y - 1.0)                             # +/-1 targets
    ym = y.mean()
    tc = t - t.mean()

    # ---------------- closed-form ridge on mean BoW ----------------
    def ridge_fit(F, target, lam):
        # F: [N,K] (no intercept), target: [N]
        Fm = F.mean(0)
        Fc = F - Fm
        A = Fc.T @ Fc + lam * np.eye(F.shape[1])
        b = Fc.T @ (target - target.mean())
        wv = np.linalg.solve(A, b)
        return wv, Fm, target.mean()

    def acc_of(score, thr):
        return float(((score > thr).astype(np.int64) == Y).mean())

    # ---- branch A : linear mean-BoW ----
    best = None
    for lam in [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]:
        wv, mu0, m0 = ridge_fit(Cnt, t, lam)
        s = Cnt @ wv
        # threshold chosen by class-balanced prior (0) and by best balanced cut
        a0 = acc_of(s, 0.0)
        cand = np.unique(np.quantile(s, np.linspace(0.02, 0.98, 193)))
        accs = np.array([acc_of(s, c) for c in cand])
        k = int(np.argmax(accs))
        if best is None or accs[k] > best[0]:
            best = (accs[k], lam, wv, cand[k], a0)
    accA, lamA, wA, thrA, accA0 = best

    # ---------------- architecture construction ----------------
    # constant-norm, zero-mean embedding rows so final LayerNorm is a pure
    # per-token rescale:  row_t = r_t * u + s_t * v ,  u,v orthonormal, zero-mean
    #   -> LN(row_t) = c * row_t  (c = 1/sqrt(R^2/32 + eps))
    # u = (e0 - e1)/sqrt2 ; v = (e1+e2+...+e31 - 30*e0)/sqrt(31*30)  (zero mean, orthonormal)
    u = np.zeros(D); u[0] = 1.0 / np.sqrt(2.0); u[1] = -1.0 / np.sqrt(2.0)
    v = np.zeros(D)
    v[1:] = 1.0
    v[0] = -30.0
    v /= np.sqrt(30.0 * 31.0)
    # verify orthogonality / zero mean
    u = u - u.mean(); v = v - v.mean()
    u /= np.linalg.norm(u); v -= (v @ u) * u; v /= np.linalg.norm(v)

    R = 3.0
    EPS = 1e-5
    cscale = 1.0 / np.sqrt(R * R / D + EPS)          # LN multiplier for unit-norm rows

    # ---- branch B : soft-count nonlinearity via block0 FFN ----
    # design directions from the top ridge coefficients (class moments / ridge)
    order = np.argsort(-np.abs(wA))
    dirs = []
    for j in order:
        if len(dirs) >= 12:
            break
        if all(abs(float(Cnt[:, j] @ d)) < 1e-9 for d in dirs):
            dirs.append(Cnt[:, j].copy())
    dirs = np.array(dirs) if len(dirs) else np.zeros((0, V))

    # build candidate embedding rows carrying both branches
    def build_rows(rvec):
        rows = np.zeros((V, D), dtype=np.float64)
        s = np.sqrt(np.maximum(R * R - rvec * rvec, 1e-6))
        rows[:] = rvec[:, None] * u[None, :] + s[:, None] * v[None, :]
        rows[0] = 0.0                                  # pad row -> zero
        return rows

    # soft-count feature: mean_t relu(a . x_t + b)  with a = direction, b = shift
    # implemented as: block0.ff1 hidden unit h = relu(a . LN(embed_t) + b)
    # ff2 maps hidden unit to residual dim 1 (the 'v' direction is not used by head)
    # NOTE: block0 FFN input is norm2(h) where h = embed (attention zeroed) -> LN(embed_t)
    # so a . LN(embed_t) = cscale * a . row_t  (row_t has zero mean)
    # choose b so that relu is active for a fraction of tokens

    # feature matrix for branch B (computed analytically from counts)
    def soft_features(dirs, shifts):
        cols = []
        for d, b in zip(dirs, shifts):
            # a . row_t = cscale * (d_t * (u . a)) ; we choose a = u-direction of d
            # simpler: use a = u (so a.row_t = cscale * r_t) -> r_t = d_t
            pass
        return cols

    # Instead of analytic gymnastics, simulate the network exactly.
    def simulate(embed_w, blk0_ff1=None, blk0_ff2=None, blk0_b1=None,
                 blk0_b2=None, head_w=None, head_b=None, pos_w=None):
        E = embed_w.astype(np.float64)
        h = E[X]                                        # [N,L,D]
        if pos_w is not None:
            h = h + pos_w[None, :, :]
        # ---- block0 : attention identity (proj zero) ----
        z = layer_norm(h, blk0_n1w, blk0_n1b)
        qkv = z @ blk0_qkv_w.T + blk0_qkv_b
        q, k, v = np.split(qkv, 3, axis=-1)
        att = attention(q, k, v, mask)
        h = h + att @ blk0_proj_w.T + blk0_proj_b
        # ---- block0 FFN ----
        z2 = layer_norm(h, blk0_n2w, blk0_n2b)
        hid = np.maximum(z2 @ blk0_ff1_w.T + blk0_ff1_b, 0.0)
        h = h + hid @ blk0_ff2_w.T + blk0_ff2_b
        # ---- block1 : identity (all zeros) ----
        hn = layer_norm(h, blk1_n1w, blk1_n1b)
        qkv = hn @ blk1_qkv_w.T + blk1_qkv_b
        q, k, v = np.split(qkv, 3, axis=-1)
        att = attention(q, k, v, mask)
        h = h + att @ blk1_proj_w.T + blk1_proj_b
        z2 = layer_norm(h, blk1_n2w, blk1_n2b)
        hid = np.maximum(z2 @ blk1_ff1_w.T + blk1_ff1_b, 0.0)
        h = h + hid @ blk1_ff2_w.T + blk1_ff2_b
        # ---- final norm + masked mean pool ----
        hf = layer_norm(h, fn_w, fn_b)
        pooled = (hf * mask[:, :, None]).sum(1) / lens[:, None]
        logits = pooled @ head_w.T + head_b
        return pooled, logits

    def layer_norm(x, g, b):
        mu = x.mean(-1, keepdims=True)
        xc = x - mu
        var = (xc * xc).mean(-1, keepdims=True)
        return xc / np.sqrt(var + 1e-5) * g + b

    def attention(q, k, v, m):
        # q,k,v: [N,L,32]; 2 heads of 16; mask out pad keys
        Nn, Ll, _ = q.shape
        out = np.zeros((Nn, Ll, 32))
        big = -1e30
        for hd in range(2):
            sl = slice(hd * 16, (hd + 1) * 16)
            sc = (q[:, :, sl] @ k[:, :, sl].transpose(0, 2, 1)) / 4.0
            sc = np.where(m[:, None, :], sc, big)
            sc = sc - sc.max(-1, keepdims=True)
            e = np.exp(sc)
            e = np.where(m[:, None, :], e, 0.0)
            a = e / np.maximum(e.sum(-1, keepdims=True), 1e-30)
            out[:, :, sl] = a @ v[:, :, sl]
        return out

    # current (zero) block weights from w
    def g(k_):
        return np.asarray(w[k_], dtype=np.float64)

    blk0_n1w, blk0_n1b = g("block0.norm1.weight"), g("block0.norm1.bias")
    blk0_n2w, blk0_n2b = g("block0.norm2.weight"), g("block0.norm2.bias")
    blk0_qkv_w, blk0_qkv_b = g("block0.qkv.weight"), g("block0.qkv.bias")
    blk0_proj_w, blk0_proj_b = g("block0.proj.weight"), g("block0.proj.bias")
    blk0_ff1_w, blk0_ff1_b = g("block0.ff1.weight"), g("block0.ff1.bias")
    blk0_ff2_w, blk0_ff2_b = g("block0.ff2.weight"), g("block0.ff2.bias")
    blk1_n1w, blk1_n1b = g("block1.norm1.weight"), g("block1.norm1.bias")
    blk1_n2w, blk1_n2b = g("block1.norm2.weight"), g("block1.norm2.bias")
    blk1_qkv_w, blk1_qkv_b = g("block1.qkv.weight"), g("block1.qkv.bias")
    blk1_proj_w, blk1_proj_b = g("block1.proj.weight"), g("block1.proj.bias")
    blk1_ff1_w, blk1_ff1_b = g("block1.ff1.weight"), g("block1.ff1.bias")
    blk1_ff2_w, blk1_ff2_b = g("block1.ff2.weight"), g("block1.ff2.bias")
    fn_w, fn_b = g("norm.weight"), g("norm.bias")

    # ---------------- assemble candidate designs ----------------
    # linear branch: r_t = wA_t (ridge coef on mean BoW), threshold via head.bias
    # nonlinear branch: soft-count features from a few directions
    #   FFN hidden unit j:  h_j = relu( a_j . LN(embed_t) + b_j )
    #   a_j must be a direction in residual space; we use a_j = u * (1/cscale) so that
    #   a_j . LN(embed_t) = r_t  (the per-token linear score)
    #   -> relu(r_t + b_j) is a soft-count of tokens above threshold -b_j
    #   ff2 maps hidden unit j to residual dim 2 (unused by head) -> no effect!
    #   Instead map it to residual dim 0/1? that would corrupt the linear channel.
    #   => use a SEPARATE residual dimension (dim 3) for the nonlinear channel and
    #      read it with head.weight[1,3].  But LN mixes dims... LN is per-token and
    #      orthogonal-ish; it does NOT preserve dim 3 separately.  => Instead put the
    #      nonlinear channel in a dim that the linear channel does not use, and let
    #      the final LN mix them; then we FIT the head on the simulated features.
    #      (head is fitted by ridge on the true simulated pooled features)

    # design: residual dims 0..k-1 carry linear scores (r_t), dim k.. carry soft counts
    # We build embed rows with r_t in dim0 (via u) and s_t in dim1 (via v) as before,
    # plus we append nonlinear content through the FFN residual branch.
    # FFN input = LN2(h) where h = embed (block0 attention zero) -> LN2(embed_t)
    #   LN2(embed_t) = cscale * embed_t (zero mean, norm R)
    #   hidden_j = relu( w1_j . (cscale*embed_t) + b1_j )
    #   we want hidden_j to depend on the token's linear score r_t:
    #     w1_j = (1/cscale) * u * alpha_j  -> w1_j . cscale*embed_t = alpha_j * r_t
    #   then relu(alpha_j * r_t + b1_j) = soft count of tokens with alpha_j*r_t > -b1_j
    #   ff2 maps hidden_j into residual dim 2 (say) with weight 1
    #   -> residual dim 2 of h_t = sum_j relu(alpha_j r_t + b1_j)  (nonlinear per token)
    #   final LN then mixes dims; head fitted on simulated pooled features.

    # choose alphas / shifts
    alphas = np.array([1.0, 1.0, 0.6, 0.6, 0.35, 0.35])
    shifts = np.array([-0.6, 0.6, -0.4, 0.4, -0.25, 0.25])
    nh = len(alphas)

    rvec = wA.copy()
    rvec[0] = 0.0
    mx = np.max(np.abs(rvec))
    if mx > R - 0.05:
        rvec = rvec * ((R - 0.05) / mx)

    E = build_rows(rvec)

    # FFN weights: w1_j = (alpha_j / cscale) * u   (32 dims, one row per hidden unit)
    ff1_w = np.zeros((64, D))
    ff1_b = np.zeros(64)
    for j in range(nh):
        ff1_w[j] = (alphas[j] / cscale) * u
        ff1_b[j] = shifts[j]
    ff2_w = np.zeros((D, 64))
    for j in range(nh):
        ff2_w[2, j] = 1.0            # nonlinear channel -> residual dim 2
    ff2_b = np.zeros(D)

    # simulate pooled features with a temporary head (identity-ish) to get features
    head_w_tmp = np.zeros((2, D)); head_w_tmp[1, :] = 1.0
    head_b_tmp = np.zeros(2)
    pooled, _ = simulate(E, blk0_ff1_w=ff1_w, blk0_ff2_w=ff2_w,
                         blk0_ff1_b=ff1_b, blk0_ff2_b=ff2_b,
                         head_w=head_w_tmp, head_b=head_b_tmp)

    # fit head by ridge on the simulated pooled features (closed form)
    F = pooled
    Fm = F.mean(0)
    Fc = F - Fm
    bestH = None
    for lam in [1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
        A = Fc.T @ Fc + lam * np.eye(D)
        b = Fc.T @ tc
        hw = np.linalg.solve(A, b)
        s = F @ hw
        cand = np.unique(np.quantile(s, np.linspace(0.02, 0.98, 193)))
        accs = np.array([acc_of(s, c) for c in cand])
        k = int(np.argmax(accs))
        if bestH is None or accs[k] > bestH[0]:
            bestH = (accs[k], lam, hw, cand[k])
    accH, lamH, hw, thrH = bestH

    # assemble final head: class 1 logit = hw . pooled + shift, class 0 logit = 0
    head_w = np.zeros((2, D))
    head_w[1, :] = hw
    head_b = np.zeros(2)
    head_b[1] = -thrH

    # final check with the true assembled weights
    pooled2, logits2 = simulate(E, blk0_ff1_w=ff1_w, blk0_ff2_w=ff2_w,
                                blk0_ff1_b=ff1_b, blk0_ff2_b=ff2_b,
                                head_w=head_w, head_b=head_b)
    pred = logits2.argmax(1)
    acc = float((pred == Y).mean())
    ce = float(-np.mean(np.log(np.clip(np.exp(logits2 - logits2.max(1, keepdims=True))
                                        / np.exp(logits2 - logits2.max(1, keepdims=True)).sum(1, keepdims=True), 1e-12, 1.0)
                            [np.arange(N), Y])))

    # ---------------- write weights ----------------
    out = {k_: np.array(v_, dtype=np.float32) for k_, v_ in w.items()}

    def put(name, arr):
        out[name] = np.asarray(arr, dtype=np.float32)

    put("embed.weight", E)
    put("pos.weight", np.zeros((64, D)))
    put("block0.qkv.weight", np.zeros((96, D)))
    put("block0.qkv.bias", np.zeros(96))
    put("block0.proj.weight", np.zeros((D, D)))
    put("block0.proj.bias", np.zeros(D))
    put("block0.ff1.weight", ff1_w)
    put("block0.ff1.bias", ff1_b)
    put("block0.ff2.weight", ff2_w)
    put("block0.ff2.bias", ff2_b)
    put("block1.qkv.weight", np.zeros((96, D)))
    put("block1.qkv.bias", np.zeros(96))
    put("block1.proj.weight", np.zeros((D, D)))
    put("block1.proj.bias", np.zeros(D))
    put("block1.ff1.weight", np.zeros((64, D)))
    put("block1.ff1.bias", np.zeros(64))
    put("block1.ff2.weight", np.zeros((D, 64)))
    put("block1.ff2.bias", np.zeros(D))
    put("norm.weight", np.ones(D))
    put("norm.bias", np.zeros(D))
    put("head.weight", head_w)
    put("head.bias", head_b)

    # safety: finite, bounded
    for k_ in out:
        a = out[k_]
        if not np.all(np.isfinite(a)):
            out[k_] = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        out[k_] = np.clip(out[k_], -1000.0, 1000.0).astype(np.float32)

    diagnostics = {
        "method": "ridge_meanBoW + block0_softcount_FFN + ridge_head_on_simulated_pooled",
        "pool_acc_new": round(acc, 4),
        "pool_acc_linear_only": round(float(accA), 4),
        "pool_acc_linear_zero_thr": round(float(accA0), 4),
        "pool_acc_head_ridge": round(float(accH), 4),
        "pool_ce_new": round(ce, 4),
        "ridge_lam_linear": lamA,
        "ridge_lam_head": lamH,
        "thr_linear": round(float(thrA), 4),
        "thr_head": round(float(thrH), 4),
        "r_absmax": round(float(np.max(np.abs(rvec))), 4),
        "R": R,
        "cscale": round(float(cscale), 5),
        "mean_len": round(float(lens.mean()), 3),
        "unk_rate": round(float((X == 1).sum() / max(mask.sum(), 1)), 4),
        "feature_std_pooled": [round(float(v), 4) for v in pooled.std(0)[:8]],
        "head_w_norm": round(float(np.linalg.norm(hw)), 4),
        "n_hidden_softcount": nh,
    }
    return {"weights": out, "diagnostics": diagnostics}
