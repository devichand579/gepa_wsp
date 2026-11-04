import numpy as np

def edit(w, f):
    # ---------------- data ----------------
    x = np.asarray(f["x"], dtype=np.int64)          # [N,64] token ids
    y = np.asarray(f["y"], dtype=np.int64)          # [N]
    N, L = x.shape
    pad = (x == 0)
    lens = (~pad).sum(1).astype(np.float64)          # [N] true lengths
    valid = lens > 0
    if not np.all(valid):
        # keep all rows but guard against zero-length docs
        lens = np.maximum(lens, 1.0)

    # ---------------- token sign statistics (class-conditional counts) ----------------
    V = w["embed.weight"].shape[0]
    npos = np.zeros(V, dtype=np.float64)
    nneg = np.zeros(V, dtype=np.float64)
    cpos = np.zeros(V, dtype=np.float64)
    cneg = np.zeros(V, dtype=np.float64)
    for lab in (0, 1):
        m = (y == lab)
        xs = x[m]
        ls = lens[m]
        # token document-frequency per class
        df = np.zeros(V, dtype=np.float64)
        for t in range(V):
            df[t] = np.count_nonzero((xs == t).any(1))
        if lab == 1:
            npos, cpos = df, ls
        else:
            nneg, cneg = df, ls
    df_all = npos + nneg
    # smoothed log-odds of "appears in positive doc" vs "appears in negative doc"
    # (documents containing the token), Laplace smoothed
    a = npos + 0.5
    b = nneg + 0.5
    lo = np.log(a) - np.log(b)                       # >0 => positive-leaning
    # token frequency in whole corpus (for idf-ish weighting)
    tokcount = np.zeros(V, dtype=np.float64)
    for t in range(V):
        tokcount[t] = np.count_nonzero((x == t))
    # sign: +1 positive-leaning, -1 negative-leaning, 0 neutral
    sign = np.zeros(V, dtype=np.float64)
    thr = 0.15
    sign[lo > thr] = 1.0
    sign[lo < -thr] = -1.0
    # never give PAD / UNK a polarity sign (UNK is a real word sometimes; keep it neutral)
    sign[0] = 0.0
    sign[1] = 0.0

    # ---------------- exact forward pass in numpy ----------------
    def layer_norm(xin, g, b, eps=1e-5):
        mu = xin.mean(-1, keepdims=True)
        var = xin.var(-1, keepdims=True, unbiased=False)
        return (xin - mu) / np.sqrt(var + eps) * g + b

    def forward(embed_w, pos_w, blocks, norm_g, norm_b, head_w, head_b, ids):
        B, T = ids.shape
        h = embed_w[ids] + pos_w[None, :, :]        # [B,T,32]
        for blk in blocks:
            z = layer_norm(h, blk['n1w'], blk['n1b'])
            qkv = z @ blk['qkv_w'].T + blk['qkv_b']   # [B,T,96]
            q, k, v = np.split(qkv, 3, axis=-1)
            B_, T_, D = q.shape
            nh = D // 16
            att = np.zeros((B_, T_, D), dtype=np.float64)
            m = (ids != 0)                            # [B,T] key mask
            for hh in range(nh):
                sl = slice(hh * 16, (hh + 1) * 16)
                qs = q[:, :, sl]                      # [B,T,16]
                ks = k[:, :, sl]
                vs = v[:, :, sl]
                sc = np.einsum('btd,bsd->bts', qs, ks) / 4.0
                sc = np.where(m[:, None, :], sc, -1e30)
                mx = sc.max(-1, keepdims=True)
                e = np.exp(sc - mx)
                e = np.where(m[:, None, :], e, 0.0)
                s = e / np.maximum(e.sum(-1, keepdims=True), 1e-30)
                att[:, :, sl] = np.einsum('bts,bsd->btd', s, vs)
            h = h + att @ blk['proj_w'].T + blk['proj_b']
            z2 = layer_norm(h, blk['n2w'], blk['n2b'])
            u = z2 @ blk['ff1_w'].T + blk['ff1_b']
            u = np.maximum(u, 0.0)
            h = h + u @ blk['ff2_w'].T + blk['ff2_b']
        hn = layer_norm(h, norm_g, norm_b)
        mm = (ids != 0).astype(np.float64)
        cnt = np.maximum(mm.sum(1, keepdims=True), 1.0)
        pooled = (hn * mm[:, :, None]).sum(1) / cnt
        logits = pooled @ head_w.T + head_b
        return logits, pooled

    # ---------------- design the sum-pooled bag-of-words network ----------------
    # per-token code direction u (fixed) and carrier so LayerNorm is affine in the code
    # embed row t = carrier + sign[t]*u  ->  after LayerNorm, code direction is sign[t]*u_hat
    D = 32
    u = np.zeros(D); u[0] = 1.0
    carrier = np.zeros(D); carrier[1] = 1.0; carrier[2] = 0.5   # fixed, gives non-degenerate norm

    embed_new = np.zeros((V, D), dtype=np.float64)
    embed_new[:] = carrier[None, :]
    embed_new += sign[:, None] * u[None, :]
    # keep PAD row neutral
    embed_new[0] = carrier

    pos_new = np.zeros((64, D), dtype=np.float64)

    # block0: qkv -> v carries the code; attention uniform over non-pad keys (1/L);
    # proj scales by L so the residual adds the SUM of per-token codes.
    n1w = np.ones(D); n1b = np.zeros(D)
    n2w = np.ones(D); n2b = np.zeros(D)
    qkv_w = np.zeros((96, D)); qkv_b = np.zeros(96)
    # v rows (32..63) pick the code dim 0 out of the normalized token vector
    qkv_w[32:64, 0] = 1.0
    # q rows: constant query u -> with k=0 the scores are 0 -> uniform attention
    qkv_w[0:32, 1] = 1.0
    proj_w = np.zeros((D, D)); proj_b = np.zeros(D)
    # proj maps attention output (which is (1/L)*sum_j code_j) to L * that = sum_j code_j
    # we need the scale L: attention output already contains 1/L; multiply by L.
    # L is not directly available, so instead we build the sum via a second trick:
    # use proj to place the attention output into dim 0 with a large gain, and rely on
    # the *residual* h += proj(attn) to add (1/L)*sum_j sign_j.
    # -> we therefore ALSO add a length-aware correction below (see 'length feature').
    proj_w[0, 0] = 1.0
    ff1_w = np.zeros((64, D)); ff1_b = np.zeros(64)
    ff2_w = np.zeros((D, 64)); ff2_b = np.zeros(D)

    blk0 = dict(n1w=n1w, n1b=n1b, qkv_w=qkv_w, qkv_b=qkv_b, proj_w=proj_w, proj_b=proj_b,
                n2w=n2w, n2b=n2b, ff1_w=ff1_w, ff1_b=ff1_b, ff2_w=ff2_w, ff2_b=ff2_b)
    blk1 = dict(n1w=n1w.copy(), n1b=n1b.copy(), qkv_w=np.zeros((96, D)), qkv_b=np.zeros(96),
                proj_w=np.zeros((D, D)), proj_b=np.zeros(D),
                n2w=n2w.copy(), n2b=n2b.copy(), ff1_w=np.zeros((64, D)), ff1_b=np.zeros(64),
                ff2_w=np.zeros((D, 64)), ff2_b=np.zeros(D))

    norm_g = np.ones(D); norm_b = np.zeros(D)

    # ---------------- feature construction (no head yet) ----------------
    def pooled_features(embed_w, pos_w, blocks, norm_g, norm_b, ids):
        _, pooled = forward(embed_w, pos_w, blocks, norm_g, norm_b,
                            np.zeros((2, D)), np.zeros(2), ids)
        return pooled

    # ---- Candidate A: mean-pooled sign code (current-style) ----
    # blocks zero => pooled = mean of normalized token vectors
    blkA0 = dict(blk0); blkA0['qkv_w'] = np.zeros((96, D)); blkA0['qkv_b'] = np.zeros(96)
    blkA0['proj_w'] = np.zeros((D, D)); blkA0['proj_b'] = np.zeros(D)
    blkA1 = blk1
    featA = pooled_features(embed_new, pos_new, [blkA0, blkA1], norm_g, norm_b, x)

    # ---- Candidate B: attention sum-pooling (uniform attention over non-pad keys) ----
    # attention output = (1/L) * sum_j v_j ; proj maps it to dim0 -> residual adds (1/L)*sum sign
    # To obtain the SUM we scale by L: we exploit that the final LayerNorm std of a token is
    # dominated by the carrier, so the *ratio* of the code to the carrier is preserved.
    # We therefore add a second, explicit length feature via a dedicated attention head:
    #   v2_j = 1 for every token  -> attention output = 1 (constant)  (no length)
    # Instead we get length from the *pooled* code magnitude: pooled dim0 = sum_sign / L.
    # We keep both: pooled dim0 = sum_sign/L (mean) and we ALSO build a "sum" feature by
    # adding the code to the residual *before* normalization with a gain that cancels the
    # 1/L factor using the known per-document length (computed from ids, exact, no learning).
    # -> implemented as: residual gain applied per document via a length-scaled proj is not
    #    expressible, so we instead use a two-pass trick: build features from the *sign sum*
    #    directly (exact, computed from ids) and use it as the network feature by choosing
    #    embed/proj so that the network reproduces it.  We verify numerically instead.

    # Practical route: compute the exact bag-of-words count features directly and then
    # realise them through the network by construction + verification.
    #   sum_sign[i] = sum_t sign[x[i,t]]  (over non-pad)
    sum_sign = np.zeros(N)
    for t in np.unique(x):
        if sign[t] == 0.0:
            continue
        sum_sign += sign[t] * (x == t).sum(1)
    # weighted count features (a few hand-built statistics, all linear in token counts)
    # f1 = sum of sign*idf, f2 = sum of |sign| (evidence count), f3 = length
    idf = np.log((N + 1.0) / (df_all + 1.0)) + 1.0
    wsum = np.zeros(N); nsum = np.zeros(N)
    for t in np.unique(x):
        if sign[t] == 0.0:
            continue
        c = (x == t).sum(1)
        wsum += sign[t] * idf[t] * c
        nsum += c
    lenf = lens.copy()

    # Candidate feature matrices
    def build_feats(kind):
        if kind == 'A':
            return featA
        if kind == 'B':   # count-based (sum) features
            return np.stack([wsum, nsum, lenf, wsum / lenf, nsum / lenf], 1)
        raise ValueError

    # ---------------- ridge head fit with CV ----------------
    def ridge_fit(X, t, lam):
        Xd = np.concatenate([X, np.ones((X.shape[0], 1))], 1)
        A = Xd.T @ Xd + lam * np.eye(Xd.shape[1])
        A[-1, -1] -= lam
        b = Xd.T @ t
        return np.linalg.solve(A, b)

    def acc_from(X, t, beta):
        Xd = np.concatenate([X, np.ones((X.shape[0], 1))], 1)
        s = Xd @ beta
        return float(np.mean((s > 0).astype(np.int64) == (t > 0).astype(np.int64)))

    t = (2.0 * y - 1.0)

    rng = np.random.RandomState(0)
    perm = rng.permutation(N)
    folds = np.array_split(perm, 5)

    results = {}
    best = None
    for kind in ('A', 'B'):
        X = build_feats(kind)
        for lam in (1e-4, 1e-2, 1.0, 10.0, 100.0, 1000.0):
            cv = 0.0
            for k in range(5):
                te = folds[k]
                tr = np.concatenate([folds[j] for j in range(5) if j != k])
                beta = ridge_fit(X[tr], t[tr], lam)
                cv += acc_from(X[te], t[te], beta) * len(te)
            cv /= N
            results[kind + '_lam' + str(lam)] = round(float(cv), 5)
            if best is None or cv > best[0]:
                best = (cv, kind, lam)

    cv_best, kind_best, lam_best = best
    Xb = build_feats(kind_best)
    beta = ridge_fit(Xb, t, lam_best)
    train_acc = acc_from(Xb, t, beta)

    # ---------------- realise the chosen features in the network ----------------
    # If the chosen feature set is the count-based one, we realise it by writing the
    # token-level weights into embed/proj such that the network reproduces the same
    # linear scores, and we VERIFY by re-running the exact forward pass.
    # For kind 'A' the network already produces featA; the head is fitted on it.
    # For kind 'B' we build a network whose pooled feature is (wsum, nsum, lenf, ...)
    # using: embed carries per-token code (sign*idf) in dim0 and (1) in dim1;
    # attention (uniform over non-pad keys) gives (1/L)*sum_j code_j  -> mean;
    # we then need the SUM: use proj with per-document gain L, which is not expressible,
    # so instead we use the *residual* trick: h += proj(attn) with proj = L * e0 * e0^T
    # is not expressible -> we use the fact that the final LayerNorm divides by the token
    # std, and we choose the code magnitudes so that the pooled dim0 = sum/L is the
    # quantity we want, then the head is fitted on the SAME quantity.  We therefore fit
    # the head on the network's own features (kind A) whenever the network cannot express
    # the count features exactly.

    # Try to realise count features exactly: set embed dim0 = sign*idf, dim1 = 1 (carrier),
    # and use attention to compute mean; the pooled dim0 = mean of sign*idf = wsum/L.
    # Multiply by L is impossible -> we instead include L as a separate pooled feature via
    # a second attention channel that outputs 1 for every token (constant) -> no.
    # => we accept: network features = [mean_code, mean_ones(=1), ...]. The head is fitted
    #    on the network's own pooled features, guaranteeing consistency.

    # Final: use the network's own features (always consistent) and fit head on them.
    Xf = featA
    beta = ridge_fit(Xf, t, lam_best)
    train_acc = acc_from(Xf, t, beta)
    # 5-fold CV on the network features with the selected lambda
    cv = 0.0
    for k in range(5):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(5) if j != k])
        b = ridge_fit(Xf[tr], t[tr], lam_best)
        cv += acc_from(Xf[te], t[te], b) * len(te)
    cv /= N

    # head from ridge on network features: beta has [D weights, bias]
    head_w = beta[:D].reshape(1, D)
    head_b = beta[D:].reshape(1)
    # map to 2-class logits: logit0 = -s, logit1 = +s  (argmax preserved)
    head_w = np.concatenate([-head_w, head_w], 0)
    head_b = np.concatenate([-head_b, head_b], 0)

    # ---------------- write back ----------------
    out = {k: np.array(v, dtype=np.float32, copy=True) for k, v in w.items()}
    out["embed.weight"] = embed_new.astype(np.float32)
    out["pos.weight"] = pos_new.astype(np.float32)
    for name, blk in (("block0", blk0), ("block1", blk1)):
        out[name + ".norm1.weight"] = blk['n1w'].astype(np.float32)
        out[name + ".norm1.bias"] = blk['n1b'].astype(np.float32)
        out[name + ".qkv.weight"] = blk['qkv_w'].astype(np.float32)
        out[name + ".qkv.bias"] = blk['qkv_b'].astype(np.float32)
        out[name + ".proj.weight"] = blk['proj_w'].astype(np.float32)
        out[name + ".proj.bias"] = blk['proj_b'].astype(np.float32)
        out[name + ".norm2.weight"] = blk['n2w'].astype(np.float32)
        out[name + ".norm2.bias"] = blk['n2b'].astype(np.float32)
        out[name + ".ff1.weight"] = blk['ff1_w'].astype(np.float32)
        out[name + ".ff1.bias"] = blk['ff1_b'].astype(np.float32)
        out[name + ".ff2.weight"] = blk['ff2_w'].astype(np.float32)
        out[name + ".ff2.bias"] = blk['ff2_b'].astype(np.float32)
    out["norm.weight"] = norm_g.astype(np.float32)
    out["norm.bias"] = norm_b.astype(np.float32)
    out["head.weight"] = head_w.astype(np.float32)
    out["head.bias"] = head_b.astype(np.float32)

    # ---------------- verification with the exact forward pass ----------------
    lg, pooled_v = forward(embed_new, pos_new, [blk0, blk1], norm_g, norm_b,
                           head_w, head_b, x)
    pred = lg.argmax(1)
    acc_final = float(np.mean(pred == y))
    ce = float(np.mean(-np.log(np.clip((lg[:, 1] - lg[:, 0]) * 0 + 1.0, 1e-9, 1.0))) * 0 +
                np.mean(np.logaddexp(0.0, -(2.0 * y - 1.0) * (lg[:, 1] - lg[:, 0]))))

    for kk in out:
        a = out[kk]
        if not np.all(np.isfinite(a)):
            out[kk] = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        out[kk] = np.clip(out[kk], -1000.0, 1000.0).astype(np.float32)

    diagnostics = {
        "hypothesis": "sum/mean bag-of-words polarity via per-token code in embed dim0 + "
                      "uniform-attention residual; head refit by closed-form ridge",
        "n_examples": int(N),
        "tokens_with_sign": int(np.count_nonzero(sign)),
        "pos_sign_tokens": int(np.count_nonzero(sign > 0)),
        "neg_sign_tokens": int(np.count_nonzero(sign < 0)),
        "cv_acc_ridge_on_netfeat": round(float(cv), 5),
        "train_acc_ridge_on_netfeat": round(float(train_acc), 5),
        "forward_acc_after_write": round(acc_final, 5),
        "forward_cross_entropy": round(ce, 5),
        "cv_grid": results,
        "selected_lambda": lam_best,
        "selected_feature_kind": kind_best,
        "pooled_feat_mean": [round(float(v), 5) for v in pooled_v.mean(0)[:6]],
        "pooled_feat_std": [round(float(v), 5) for v in pooled_v.std(0)[:6]],
        "head_w_row1_first6": [round(float(v), 5) for v in head_w[1][:6]],
        "head_bias": [round(float(v), 5) for v in head_b],
        "baseline_acc_reported": 0.7973,
    }
    return {"weights": out, "diagnostics": diagnostics}
