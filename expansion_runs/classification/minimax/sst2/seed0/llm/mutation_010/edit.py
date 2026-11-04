import numpy as np
import math

def edit(w, f):
    # ----------------------------------------------------------------------------
    # 0. helpers
    # ----------------------------------------------------------------------------
    def lin(x, W, b):
        # PyTorch-style linear: y = x @ W.T + b
        return x @ W.T + b

    def layer_norm(x, g, b, eps=1e-5):
        mu = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True, unbiased=False)
        return (x - mu) / np.sqrt(var + eps) * g + b

    def softmax(x, axis=-1):
        m = x.max(axis=axis, keepdims=True)
        e = np.exp(x - m)
        return e / e.sum(axis=axis, keepdims=True)

    # ----------------------------------------------------------------------------
    # 1. data
    # ----------------------------------------------------------------------------
    x = np.asarray(f["x"], dtype=np.int64)      # [N,64]
    y = np.asarray(f["y"], dtype=np.int64)      # [N]
    N, L = x.shape
    mask = (x != 0)                            # [N,L] bool, True = real token
    lens = mask.sum(1).astype(np.float64)      # [N]
    V = 2000

    # ----------------------------------------------------------------------------
    # 2. token statistics from the permitted pool only
    # ----------------------------------------------------------------------------
    cnt = np.zeros((N, V), dtype=np.float64)
    for j in range(L):
        ids = x[:, j]
        nz = ids > 0
        if nz.any():
            np.add.at(cnt, (np.nonzero(nz)[0], ids[nz]), 1.0)
    freq = cnt.sum(0)                           # [V] document frequency
    dfreq = (cnt > 0).sum(0).astype(np.float64)

    # negation lexicon derived from the *pool text statistics only*:
    # tokens that are (a) frequent, (b) much more common in the class whose
    # documents contain them, and (c) syntactically negation-like.
    # We use a purely statistical construction: for each token compute the
    # class-conditional document frequency and keep tokens whose negative-class
    # share is extreme (these are the "not / n't / never / no ..." family).
    neg_docs = (y == 0)
    pos_docs = (y == 1)
    df_neg = np.zeros(V)
    df_pos = np.zeros(V)
    for c, sel in ((0, neg_docs), (1, pos_docs)):
        sub = cnt[sel]
        df_neg = (sub > 0).sum(0) if c == 0 else df_neg
        df_pos = (sub > 0).sum(0) if c == 1 else df_pos
    # NOTE: df_neg/df_pos computed above are counts of docs in each class that
    # contain the token; normalise by class size.
    df_neg = df_neg / max(1.0, float(neg_docs.sum()))
    df_pos = df_pos / max(1.0, float(pos_docs.sum()))
    share_neg = (df_neg + 1e-6) / (df_neg + df_pos + 2e-6)
    # candidate negators: high frequency and strongly negative-class-associated
    cand = np.where((dfreq >= 20) & (share_neg > 0.72))[0]
    # keep only tokens that look like negations by *shape* of their usage:
    # a negator is a token that appears in the first third of documents far
    # more often than in the last third (English negation is fronted).
    pos_frac = np.zeros(V)
    for j in range(L):
        ids = x[:, j]
        nz = ids > 0
        if nz.any():
            np.add.at(pos_frac, ids[nz], 1.0)
    # front/back ratio computed on token occurrences
    early = np.zeros(V)
    late = np.zeros(V)
    for j in range(L):
        ids = x[:, j]
        nz = ids > 0
        if nz.any():
            if j < L // 3:
                np.add.at(early, ids[nz], 1.0)
            elif j > 2 * L // 3:
                np.add.at(late, ids[nz], 1.0)
    fb = (early + 1.0) / (late + 1.0)
    neg_ids = cand[fb[cand] > 1.15]
    if neg_ids.size == 0:                       # safe fallback: top-scoring tokens
        neg_ids = cand[:12]
    neg_mask = np.zeros(V, dtype=bool)
    neg_mask[neg_ids] = True

    # ----------------------------------------------------------------------------
    # 3. build candidate feature matrices (per-document, count based)
    # ----------------------------------------------------------------------------
    # 3a. negation-window counts: token inside a window of W tokens after a negator
    WINS = [3, 5]
    negcnt = {W: np.zeros((N, V), dtype=np.float64) for W in WINS}
    # also "negator token itself" counts
    negself = np.zeros((N, V), dtype=np.float64)
    for W in WINS:
        for i in range(N):
            ids = x[i]
            m = mask[i]
            if not m.any():
                continue
            toks = ids[m]
            neg_flags = neg_mask[toks]
            # window flag: token j is in scope if some negator occurs within W
            # tokens before it
            scope = np.zeros(len(toks), dtype=bool)
            run = 0
            for t in range(len(toks)):
                if neg_flags[t]:
                    run = W
                elif run > 0:
                    scope[t] = True
                    run -= 1
            if scope.any():
                np.add.at(negcnt[W][i], toks[scope], 1.0)
            if neg_flags.any():
                np.add.at(negself[i], toks[neg_flags], 1.0)

    def build_features(kind):
        """Return (X [N,d], names list)."""
        cols = []
        names = []
        if kind in ("uni", "uni_neg", "uni_neg_len", "uni_neg_next"):
            cols.append(cnt)
            names.append("uni")
        if kind in ("uni_neg", "uni_neg_len", "uni_neg_next"):
            cols.append(negcnt[3])
            names.append("neg3")
        if kind in ("uni_neg_len", "uni_neg_next"):
            cols.append(negcnt[5])
            names.append("neg5")
            cols.append(negself)
            names.append("negself")
        if kind == "uni_neg_len":
            cols.append(lens.reshape(-1, 1))
            names.append("len")
        if kind == "uni_neg_next":
            # "next token after negator" counts: a bigram-ish feature
            cols.append(negcnt[1])
            names.append("neg1")
        X = np.concatenate(cols, axis=1)
        return X, names

    # ----------------------------------------------------------------------------
    # 4. closed-form ridge with 5-fold CV (normal equations, no gradients)
    # ----------------------------------------------------------------------------
    t = np.where(y == 1, 1.0, -1.0)

    def ridge_fit(X, t, lam):
        # centre & scale columns (ridge on standardised design)
        mu = X.mean(0)
        sd = X.std(0)
        sd[sd < 1e-9] = 1.0
        Z = (X - mu) / sd
        A = Z.T @ Z + lam * np.eye(Z.shape[1])
        b = Z.T @ t
        w = np.linalg.solve(A, b)
        return (mu, sd, w)

    def ridge_pred(X, par):
        mu, sd, w = par
        Z = (X - mu) / sd
        return Z @ w

    def cv_score(X, lam, folds=5, seed=0):
        rng = np.random.RandomState(seed)
        idx = rng.permutation(N)
        parts = np.array_split(idx, folds)
        accs = []
        for k in range(folds):
            te = parts[k]
            tr = np.concatenate([parts[j] for j in range(folds) if j != k])
            par = ridge_fit(X[tr], t[tr], lam)
            s = ridge_pred(X[te], par)
            accs.append(float(np.mean((s > 0).astype(np.int64) == y[te])))
        return float(np.mean(accs))

    lams = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
    kinds = ["uni", "uni_neg", "uni_neg_len", "uni_neg_next"]
    cv_table = {}
    best = None
    for kind in kinds:
        X, names = build_features(kind)
        row = []
        for lam in lams:
            a = cv_score(X, lam)
            row.append([lam, round(a, 4)])
            if best is None or a > best[0] + 1e-9:
                best = (a, kind, lam, names)
        cv_table[kind] = row

    cv_acc, kind, lam, names = best
    X, names = build_features(kind)

    # ----------------------------------------------------------------------------
    # 5. realise the fitted score inside the fixed architecture
    #
    #   channel 0 of the pooled vector  = mean_t  uni_weight(id_t)
    #   channel 1 of the pooled vector  = mean_t  neg_weight(id_t)
    #
    # Both are per-token scalars, so the pooled value equals the fitted
    # (count-based) feature up to the affine map implied by the ridge fit.
    # The head then applies the fitted linear coefficients.
    #
    # The two channels are realised through the *attention* of block1:
    #   v_j  carries  [uni_w(id_j), neg_w(id_j), 0 ... 0]
    #   q_i  is constant  -> attention output at position i = mean over
    #   non-pad positions of v  (a global average of the two channels)
    #   proj is the identity on the first two channels
    # The residual stream keeps the token embedding, so the final pooled
    # vector holds [mean uni_w, mean neg_w] in channels 0/1 and the head
    # reads them out linearly.
    # ----------------------------------------------------------------------------
    par = ridge_fit(X, t, lam)
    mu, sd, beta = par
    # split the ridge coefficients back into the per-channel groups
    # (columns of X are ordered as in build_features)
    d = X.shape[1]
    # column index ranges
    ranges = []
    off = 0
    for nm in names:
        if nm == "uni":
            ranges.append(("uni", off, off + V)); off += V
        elif nm == "neg3":
            ranges.append(("neg3", off, off + V)); off += V
        elif nm == "neg5":
            ranges.append(("neg5", off, off + V)); off += V
        elif nm == "negself":
            ranges.append(("negself", off, off + V)); off += V
        elif nm == "neg1":
            ranges.append(("neg1", off, off + V)); off += V
        elif nm == "len":
            ranges.append(("len", off, off + 1)); off += 1
    assert off == d

    def group_coeffs(gname):
        for nm, a, b in ranges:
            if nm == gname:
                return beta[a:b]
        return np.zeros(0)

    uni_beta = group_coeffs("uni").copy()
    neg_beta = group_coeffs("neg3").copy() if "neg3" in names else np.zeros(V)
    if neg_beta.size == 0:
        neg_beta = np.zeros(V)
    len_beta = group_coeffs("len")
    len_mu = mu[ranges[[n for n, a, b in ranges].index("len")][0]] if "len" in names else 0.0
    len_sd = sd[ranges[[n for n, a, b in ranges].index("len")][0]] if "len" in names else 1.0

    # per-token weights (in standardised space -> raw count space)
    # raw coefficient on count of token v :  beta_v / sd_v
    uni_raw = uni_beta / np.maximum(sd[:V], 1e-9)
    neg_raw = neg_beta / np.maximum(sd[V:2 * V] if neg_beta.size else np.ones(V), 1e-9)
    # intercept: t_hat = (X_std @ beta) ; reconstruct intercept from ridge
    # (ridge without explicit intercept: intercept = mean(t) - mean(Z@beta))
    Zfull = (X - mu) / sd
    pred_full = Zfull @ beta
    intercept = float(t.mean() - pred_full.mean())

    # scale down so that |embed| stays O(1): normalise the two weight vectors
    def nrm(v):
        s = np.abs(v).max()
        return v / s if s > 1e-12 else v

    uni_raw = nrm(uni_raw)
    neg_raw = nrm(neg_raw)
    su = float(np.abs(uni_raw).max()) or 1.0
    sn = float(np.abs(neg_raw).max()) or 1.0

    # ---- embedding construction -------------------------------------------
    # rows: e(v) = [ uni_raw(v)*su , neg_raw(v)*sn , filler... ]
    # filler channels are chosen so that each row has zero mean and unit std
    # (keeps LayerNorm(embed[id]) == embed[id], i.e. blocks stay ~identity)
    D = 32
    E = np.zeros((V, D), dtype=np.float64)
    E[:, 0] = uni_raw * su
    E[:, 1] = neg_raw * sn
    # remaining channels: a fixed orthonormal-ish pattern scaled to restore
    # zero mean / unit std per row.
    rng = np.random.RandomState(1234)
    F = rng.randn(V, D - 2)
    F -= F.mean(1, keepdims=True)
    F /= (F.std(1, keepdims=True) + 1e-12)
    # scale filler so that total row std = 1 and row mean = 0
    content = E[:, :2]
    cvar = content.var(1, keepdims=True)
    # want total var 1 -> filler var = 1 - cvar ; clip to >= 0
    fscale = np.sqrt(np.maximum(1.0 - cvar, 0.0))
    E[:, 2:] = F * fscale
    # re-centre / re-normalise exactly
    E -= E.mean(1, keepdims=True)
    E /= (E.std(1, keepdims=True) + 1e-12)
    # now channel 0/1 no longer exactly uni_raw/neg_raw; recompute the
    # effective per-token weights from the *final* embedding rows so the
    # closed-form readout stays exact.
    eff_uni = E[:, 0].copy()
    eff_neg = E[:, 1].copy()
    # the pooled feature (before final LayerNorm) will be
    #   p0 = mean_t eff_uni(id_t), p1 = mean_t eff_neg(id_t)
    # and the head uses coefficients c0, c1 with an intercept.
    # We want the head score to equal  alpha0*p0 + alpha1*p1 + b0  (a linear
    # function of the fitted count features).  Because the final LayerNorm
    # rescales by a per-example sigma, we instead realise the score through
    # the *pre-normalisation* pooled vector via the head on the normalised
    # features; to keep the mapping exact we use the identity
    #   features = (p - mu(p)) / sigma(p)
    # and choose the head so that the decision is equivalent to thresholding
    # the fitted score.  Concretely we use the direction-based readout:
    #   score_dir = c0*features0 + c1*features1
    # which is monotone in the fitted score for the (dominant) channel.
    # -> Use a single dominant channel: fold the second channel into the
    #    first is impossible (different per-token weights), so instead we
    #    keep both channels and let the head read both; the LayerNorm scale
    #    is common to both and therefore does not change the *ordering* of
    #    the linear combination except through the intercept, which we fit.
    # Fit the head coefficients directly by least squares on the realised
    # features (closed form, no gradients) using the pool.
    # ---------------------------------------------------------------------
    # Build the realised pooled features with a NumPy forward pass and fit
    # head.weight / head.bias by closed-form least squares on the *labels*
    # (ridge on 2 classes -> linear discriminant on the 2-dim feature).
    # ---------------------------------------------------------------------
    def forward(ids, W, headW=None, headb=None, want_feats=False):
        """Full forward pass replicating the described architecture."""
        n = ids.shape[0]
        m = (ids != 0)
        h = W["embed.weight"][ids] + W["pos.weight"][None, :, :]
        for bi in range(2):
            p = "block%d" % bi
            z = layer_norm(h, W[p + ".norm1.weight"], W[p + ".norm1.bias"])
            qkv = lin(z, W[p + ".qkv.weight"], W[p + ".qkv.bias"])
            q, k, v = np.split(qkv, 3, axis=-1)
            att = np.zeros_like(v)
            for hd in range(2):
                sl = slice(hd * 16, (hd + 1) * 16)
                qs = q[:, :, sl]
                ks = k[:, :, sl]
                vs = v[:, :, sl]
                sc = np.einsum("nid,njd->nij", qs, ks) / math.sqrt(16.0)
                sc = np.where(m[:, None, :], sc, -1e30)
                a = softmax(sc, axis=-1)
                att[:, :, sl] = np.einsum("nij,njd->nid", a, vs)
            h = h + lin(att, W[p + ".proj.weight"], W[p + ".proj.bias"])
            z2 = layer_norm(h, W[p + ".norm2.weight"], W[p + ".norm2.bias"])
            a1 = lin(z2, W[p + ".ff1.weight"], W[p + ".ff1.bias"])
            a1 = np.maximum(a1, 0.0)
            h = h + lin(a1, W[p + ".ff2.weight"], W[p + ".ff2.bias"])
        h = layer_norm(h, W["norm.weight"], W["norm.bias"])
        cnt_tok = m.sum(1).astype(np.float64)
        pooled = (h * m[:, :, None]).sum(1) / np.maximum(cnt_tok, 1.0)[:, None]
        feats = pooled
        if headW is None:
            return feats
        logits = lin(feats, headW, headb)
        return feats, logits

    # assemble candidate weights
    Wnew = {k: np.array(v, dtype=np.float32) for k, v in w.items()}
    Wnew["embed.weight"] = E.astype(np.float32)
    Wnew["pos.weight"] = np.zeros((L, D), dtype=np.float32)
    for p in ("block0", "block1"):
        Wnew[p + ".norm1.weight"] = np.ones(D, dtype=np.float32)
        Wnew[p + ".norm1.bias"] = np.zeros(D, dtype=np.float32)
        Wnew[p + ".norm2.weight"] = np.ones(D, dtype=np.float32)
        Wnew[p + ".norm2.bias"] = np.zeros(D, dtype=np.float32)
        Wnew[p + ".qkv.weight"] = np.zeros((3 * D, D), dtype=np.float32)
        Wnew[p + ".qkv.bias"] = np.zeros(3 * D, dtype=np.float32)
        Wnew[p + ".proj.weight"] = np.zeros((D, D), dtype=np.float32)
        Wnew[p + ".proj.bias"] = np.zeros(D, dtype=np.float32)
        Wnew[p + ".ff1.weight"] = np.zeros((2 * D, D), dtype=np.float32)
        Wnew[p + ".ff1.bias"] = np.zeros(2 * D, dtype=np.float32)
        Wnew[p + ".ff2.weight"] = np.zeros((D, 2 * D), dtype=np.float32)
        Wnew[p + ".ff2.bias"] = np.zeros(D, dtype=np.float32)
    Wnew["norm.weight"] = np.ones(D, dtype=np.float32)
    Wnew["norm.bias"] = np.zeros(D, dtype=np.float32)

    # --- realise the two channels through block1 attention (global mean) -----
    # v_j = [eff_uni(id_j), eff_neg(id_j), 0...]
    # q_i = 0 -> uniform attention over non-pad keys -> attention output at
    # every position = mean of v over non-pad positions.
    # proj = identity (first two channels), residual adds it to h.
    # The final pooled vector then has channel0 = mean_t eff_uni + mean_t eff_uni
    # (residual + attention) and channel1 = 2*mean_t eff_neg.
    # We compensate the factor 2 in the head fit below.
    vmat = np.zeros((3 * D, D), dtype=np.float64)
    vmat[2 * D + 0, 0] = 1.0        # v channel 0 -> dim 0
    vmat[2 * D + 1, 1] = 1.0        # v channel 1 -> dim 1
    Wnew["block1.qkv.weight"] = vmat.astype(np.float32)
    pmat = np.zeros((D, D), dtype=np.float64)
    pmat[0, 0] = 1.0
    pmat[1, 1] = 1.0
    Wnew["block1.proj.weight"] = pmat.astype(np.float32)

    # realise features
    feats = forward(x, Wnew)
    feats = feats.astype(np.float64)

    # closed-form head: ridge on the 2 (or few) informative channels.
    # Use a small set of candidate channels and pick by CV over the pool.
    cand_cols = {
        "c01": [0, 1],
        "c0": [0],
        "c012": [0, 1, 2],
        "c0123": [0, 1, 2, 3],
    }
    head_best = None
    for hname, cols in cand_cols.items():
        Fm = feats[:, cols]
        for hlam in [1e-6, 1e-3, 1e-2, 1e-1, 1.0]:
            # 5-fold CV accuracy of a linear (ridge) threshold on Fm
            rng = np.random.RandomState(7)
            idx = rng.permutation(N)
            parts = np.array_split(idx, 5)
            accs = []
            for k in range(5):
                te = parts[k]
                tr = np.concatenate([parts[j] for j in range(5) if j != k])
                A = np.concatenate([Fm[tr], np.ones((tr.size, 1))], axis=1)
                B = np.concatenate([Fm[te], np.ones((te.size, 1))], axis=1)
                G = A.T @ A + hlam * np.eye(A.shape[1])
                wv = np.linalg.solve(G, A.T @ t[tr])
                s = B @ wv
                accs.append(float(np.mean((s > 0).astype(np.int64) == y[te])))
            a = float(np.mean(accs))
            if head_best is None or a > head_best[0]:
                head_best = (a, hname, cols, hlam)
    hacc, hname, hcols, hlam = head_best
    Fm = feats[:, hcols]
    A = np.concatenate([Fm, np.ones((N, 1))], axis=1)
    G = A.T @ A + hlam * np.eye(A.shape[1])
    wv = np.linalg.solve(G, A.T @ t)
    # map to head.weight [2,32] / head.bias [2]
    hw = np.zeros((2, D), dtype=np.float64)
    hb = np.zeros(2, dtype=np.float64)
    for j, c in enumerate(hcols):
        hw[1, c] = wv[j]
        hw[0, c] = -wv[j]
    hb[1] = wv[-1]
    hb[0] = -wv[-1]
    Wnew["head.weight"] = hw.astype(np.float32)
    Wnew["head.bias"] = hb.astype(np.float32)

    # ----------------------------------------------------------------------------
    # 6. verify with the realised forward pass
    # ----------------------------------------------------------------------------
    _, logits = forward(x, Wnew, Wnew["head.weight"], Wnew["head.bias"])
    pred = logits.argmax(1)
    acc = float(np.mean(pred == y))
    p1 = np.exp(logits[:, 1] - logits.max(1))
    p1 = p1 / (p1 + np.exp(logits[:, 0] - logits.max(1)))
    p1 = np.clip(p1, 1e-7, 1 - 1e-7)
    ce = float(-np.mean(y * np.log(p1) + (1 - y) * np.log(1 - p1)))
    # baseline (parent) accuracy from the cached logits
    base_logits = np.asarray(f["logits"], dtype=np.float64)
    base_acc = float(np.mean(base_logits.argmax(1) == y))
    base_ce = float(-np.mean(y * np.log(np.clip(np.asarray(f["probs"])[:, 1], 1e-7, 1)) +
                            (1 - y) * np.log(np.clip(1 - np.asarray(f["probs"])[:, 1], 1e-7, 1))))

    # ----------------------------------------------------------------------------
    # 7. safety: finite, float32, |.| <= 1000, shapes preserved
    # ----------------------------------------------------------------------------
    for k in w:
        arr = Wnew[k]
        if not np.all(np.isfinite(arr)):
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, -1000.0, 1000.0).astype(np.float32)
        assert arr.shape == np.asarray(w[k]).shape, (k, arr.shape, np.asarray(w[k]).shape)
        Wnew[k] = arr

    diagnostics = {
        "chosen_feature_set": kind,
        "chosen_ridge_lambda": lam,
        "cv_acc_by_feature_set": {k: v for k, v in cv_table.items()},
        "best_cv_acc": round(cv_acc, 4),
        "head_channels": hname,
        "head_ridge_lambda": hlam,
        "head_cv_acc": round(hacc, 4),
        "parent_pool_acc": round(base_acc, 4),
        "parent_pool_ce": round(base_ce, 4),
        "new_pool_acc": round(acc, 4),
        "new_pool_ce": round(ce, 4),
        "n_negators_found": int(neg_ids.size),
        "negator_ids": neg_ids[:40].tolist(),
        "negator_df": dfreq[neg_ids][:40].round(1).tolist(),
        "negator_share_neg": share_neg[neg_ids][:40].round(3).tolist(),
        "uni_w_absmax": round(float(np.abs(uni_raw).max()), 4),
        "neg_w_absmax": round(float(np.abs(neg_raw).max()), 4),
        "embed_row_std_mean": round(float(Wnew["embed.weight"].std(1).mean()), 4),
        "embed_row_mean_absmean": round(float(np.abs(Wnew["embed.weight"].mean(1)).mean()), 6),
        "head_weight_absmax": round(float(np.abs(Wnew["head.weight"]).max()), 4),
        "feature_channel_means": [round(float(feats[:, c].mean()), 4) for c in range(6)],
        "feature_channel_stds": [round(float(feats[:, c].std()), 4) for c in range(6)],
        "corr_feat0_label": round(float(np.corrcoef(feats[:, 0], t)[0, 1]), 4),
        "corr_feat1_label": round(float(np.corrcoef(feats[:, 1], t)[0, 1]), 4),
        "delta_acc_vs_parent": round(acc - base_acc, 4),
    }
    return {"weights": Wnew, "diagnostics": diagnostics}
