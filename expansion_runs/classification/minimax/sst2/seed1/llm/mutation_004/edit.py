import numpy as np

def edit(w, f):
    x = np.asarray(f["x"], dtype=np.int64)
    y = np.asarray(f["y"], dtype=np.int64)
    N, T = x.shape
    V = w["embed.weight"].shape[0]
    D = w["embed.weight"].shape[1]

    # ---------------- 1. token statistics from the permitted pool ----------------
    mask = (x != 0)
    # counts per class (documents containing token, and token occurrences)
    cls = (y == 1)
    npos = int(cls.sum()); nneg = N - npos
    doc_pos = np.zeros(V, dtype=np.float64); doc_neg = np.zeros(V, dtype=np.float64)
    occ_pos = np.zeros(V, dtype=np.float64); occ_neg = np.zeros(V, dtype=np.float64)
    for c in range(2):
        sel = x[cls if c == 1 else ~cls]
        m = sel != 0
        rows = np.repeat(np.arange(sel.shape[0]), m.sum(axis=1))
        cols = sel[m]
        np.add.at(doc_pos if c == 1 else doc_neg, cols, 1.0)
        np.add.at(occ_pos if c == 1 else occ_neg, cols, 1.0)

    # smoothed log-odds (Jeffreys-ish smoothing, alpha=1 on doc counts)
    alpha = 1.0
    p1 = (doc_pos + alpha) / (npos + 2 * alpha)
    p0 = (doc_neg + alpha) / (nneg + 2 * alpha)
    s = np.log(p1) - np.log(p0)
    s[0] = 0.0
    # shrink tokens with little evidence toward 0 (variance-based shrinkage)
    ev = doc_pos + doc_neg
    shrink = ev / (ev + 8.0)
    s = s * shrink

    # ---------------- 2. negation-aware token score ----------------
    # vocabulary is not given as ids here, so detect negators by *behaviour*:
    # tokens that most often precede a strongly negative token are treated as negators.
    neg_strength = np.maximum(-s, 0.0)
    # score of "token j preceded by token i": use co-occurrence weighted by neg_strength
    # build predecessor map cheaply on the pool
    pred = np.zeros((V, V), dtype=np.float32)  # too big? V=2000 -> 4M floats = 16MB ok
    for i in range(N):
        row = x[i]
        m = row != 0
        ids = row[m]
        if ids.size < 2:
            continue
        prev = ids[:-1]; cur = ids[1:]
        np.add.at(pred, (prev, cur), 1.0)
    # negator score: how often token i precedes a negative-word token, relative to base rate
    base = neg_strength / max(neg_strength.sum(), 1e-9)
    cooc_neg = pred @ neg_strength           # [V] total negative mass after token i
    tot = pred.sum(axis=1) + 1.0
    lift = (cooc_neg / tot) / (base.mean() + 1e-9)
    # a token is a negator if it strongly precedes negative words
    negator = np.zeros(V, dtype=np.float64)
    cand = np.where((tot > 20) & (lift > 1.6))[0]
    for t in cand:
        # fraction of its occurrences followed by a negative word
        frac = cooc_neg[t] / (tot[t] - 1.0 + 1e-9)
        if frac > 0.35:
            negator[t] = 1.0
    # apply negation: a token preceded by a negator gets its score flipped
    s_neg = s.copy()
    if negator.any():
        # for each (i,j) pair where i is a negator, j's score flips
        flip_pairs = np.argwhere(pred > 0)
        # vectorised: build adjacency of negator -> token
        neg_idx = np.where(negator > 0)[0]
        for i in neg_idx:
            js = np.where(pred[i] > 0)[0]
            s_neg[js] = -s_neg[js]
    # ---------------- 3. features per example (closed form) ----------------
    # feature map: signed counts of positive/negative tokens (as the network computes)
    def signed_counts(scores):
        pos = (scores > 0).astype(np.float64)
        neg = (scores < 0).astype(np.float64)
        F = np.zeros((N, 4), dtype=np.float64)
        for i in range(N):
            ids = x[i][mask[i]]
            if ids.size == 0:
                continue
            v = scores[ids]
            F[i, 0] = (v > 0).sum()
            F[i, 1] = (v < 0).sum()
            F[i, 2] = v.sum()
            F[i, 3] = ids.size
        return F

    Fraw = signed_counts(s)
    Fneg = signed_counts(s_neg)

    # ---------------- 4. ridge closed-form fit on engineered features ----------------
    def ridge_fit(F, target, lam):
        Fb = np.hstack([F, np.ones((F.shape[0], 1))])
        A = Fb.T @ Fb + lam * np.eye(Fb.shape[1])
        b = Fb.T @ target
        return np.linalg.solve(A, b)

    def acc(F, wts, target):
        s_ = F @ wts[:-1] + wts[-1]
        return float(((s_ > 0).astype(np.int64) == (target > 0).astype(np.int64)).mean())

    t = np.where(y == 1, 1.0, -1.0)
    # honest split: even/odd indices (deterministic, no search over training)
    ev = np.arange(N) % 2 == 0
    tr, va = ~ev, ev
    best = None
    # a small, fixed set of closed-form candidates (each solved in closed form)
    for F in (Fraw, Fneg, np.hstack([Fraw, Fneg])):
        for lam in (1.0, 10.0, 100.0, 1000.0):
            wtr = ridge_fit(F[tr], t[tr], lam)
            a = acc(F[va], wtr, t[va])
            if best is None or a > best[0]:
                best = (a, F, lam, wtr)
    val_acc, Fbest, lam, wfull = best
    wfull = ridge_fit(Fbest, t, lam)

    # ---------------- 5. realise the classifier inside the fixed architecture ----------------
    # The network can only carry a *direction* per token (LayerNorm per position).
    # Use the token score's SIGN as the per-token direction, and let the head
    # (linear on the pooled 32-dim vector) realise the weighted combination.
    # Pooled vector layout (channels):
    #   ch0 : +1 for positive-score tokens, -1 for negative-score tokens (signed count)
    #   ch1 : number of tokens (length)
    #   ch2 : +1 for positive-score tokens (positive count)
    #   ch3 : +1 for negative-score tokens (negative count)
    #   ch4 : sign-weighted sum (sum of signs)  -> equals ch0 for sign tokens
    #   ch5.. : negated-sentiment signed count / counts
    # To realise multiple channels we need the per-token vector to carry them and
    # the final LayerNorm to preserve them. LayerNorm divides by per-token std, so
    # we instead realise each channel as a *separate* direction and use the head
    # to combine: since LN(v) = 4*sign(s)*e for v = s*e, only one direction family
    # survives per token. Therefore we use TWO independent "directions" by placing
    # the sign pattern in channels (0,1) for the raw score and (2,3) for the
    # negated score, and let the *head* read the pooled channels.
    #   embed[t] = [sgn(s_t), -sgn(s_t), sgn(s_neg_t), -sgn(s_neg_t), 0...]
    # After final LN (identity weight, zero bias) each token contributes
    #   [4*sgn, -4*sgn, 4*sgn_neg, -4*sgn_neg, 0...] / (per-token std factor)
    # but LN normalises the WHOLE 32-vector, so we must keep the ratio fixed:
    # use equal-magnitude entries so the std is the same for every token.
    e = np.zeros((V, D), dtype=np.float64)
    e[:, 0] = 1.0; e[:, 1] = -1.0
    e[:, 2] = 1.0; e[:, 3] = -1.0
    e[:, 4] = 1.0; e[:, 5] = -1.0
    e[:, 6] = 1.0; e[:, 7] = -1.0
    # token sign patterns (raw and negation-aware)
    sg = np.sign(s); sg[0] = 0.0
    sgn_ = np.sign(s_neg); sgn_[0] = 0.0
    # channel groups: (0,1) raw sign, (2,3) negated sign, (4,5) raw magnitude sign,
    # (6,7) negated magnitude sign  -> 4 independent "counters" per token
    embed_new = np.zeros((V, D), dtype=np.float64)
    embed_new[:, 0] = sg; embed_new[:, 1] = -sg
    embed_new[:, 2] = sgn_; embed_new[:, 3] = -sgn_
    embed_new[:, 4] = sg; embed_new[:, 5] = -sg
    embed_new[:, 6] = sgn_; embed_new[:, 7] = -sgn_

    # ---------------- 6. build blocks so the per-token vector survives ----------
    # block0: identity pass-through (norm1 = identity, qkv -> v = input, proj -> identity)
    # block1: identity pass-through as well.
    # With LayerNorm inside the blocks, the *direction* is preserved but the
    # per-token magnitude is normalised; since we only need signs, that is fine.
    # However the final `norm` must not destroy the sign pattern: it must be the
    # identity (weight=1, bias=0) and the per-token vector must already have
    # zero mean and equal scale across the used channels.
    # We therefore make the embed vector zero-mean over ALL 32 channels:
    embed_new -= embed_new.mean(axis=1, keepdims=True)

    # blocks: make them exact identity maps.
    for b in ("block0", "block1"):
        w[b + ".norm1.weight"] = np.ones(D, dtype=np.float32)
        w[b + ".norm1.bias"] = np.zeros(D, dtype=np.float32)
        w[b + ".norm2.weight"] = np.ones(D, dtype=np.float32)
        w[b + ".norm2.bias"] = np.zeros(D, dtype=np.float32)
        # qkv: v-part = identity, q/k = zeros
        qkvw = np.zeros((3 * D, D), dtype=np.float32)
        qkvw[2 * D:3 * D, :] = np.eye(D, dtype=np.float32)
        w[b + ".qkv.weight"] = qkvw
        w[b + ".qkv.bias"] = np.zeros(3 * D, dtype=np.float32)
        # proj: identity
        w[b + ".proj.weight"] = np.eye(D, dtype=np.float32)
        w[b + ".proj.bias"] = np.zeros(D, dtype=np.float32)
        # ff1/ff2: identity (ff1 width 64 -> embed 32 dims in first 32 units, relu ok)
        ff1w = np.zeros((64, D), dtype=np.float32)
        ff1w[:, :] = np.eye(D, dtype=np.float32)
        w[b + ".ff1.weight"] = ff1w
        w[b + ".ff1.bias"] = np.zeros(64, dtype=np.float32)
        ff2w = np.zeros((D, 64), dtype=np.float32)
        ff2w[:, :D] = np.eye(D, dtype=np.float32)
        w[b + ".ff2.weight"] = ff2w
        w[b + ".ff2.bias"] = np.zeros(D, dtype=np.float32)

    w["pos.weight"] = np.zeros((64, D), dtype=np.float32)
    w["norm.weight"] = np.ones(D, dtype=np.float32)
    w["norm.bias"] = np.zeros(D, dtype=np.float32)
    w["embed.weight"] = embed_new.astype(np.float32)

    # ---------------- 7. head: closed-form ridge on the pooled features ---------
    # pooled channels (after final LN, per-token vector has 8 non-zero channels
    # with equal magnitude, so LN(v) = 4*signpattern/... constant scale):
    #   c0 = sum_t sgn(s_t)      (raw signed count)
    #   c1 = -c0
    #   c2 = sum_t sgn(s_neg_t)  (negation-aware signed count)
    #   c3 = -c2
    #   c4 = c0, c5 = -c0, c6 = c2, c7 = -c2
    # plus length information is NOT available (mean pooling removes it), so
    # the head sees only these 8 channels. Fit head on the matching features.
    def pooled_features(scores):
        P = np.zeros((N, 8), dtype=np.float64)
        for i in range(N):
            ids = x[i][mask[i]]
            if ids.size == 0:
                continue
            g = np.sign(scores[ids])
            P[i, 0] = g.sum(); P[i, 1] = -g.sum()
            P[i, 2] = g.sum(); P[i, 3] = -g.sum()
            P[i, 4] = g.sum(); P[i, 5] = -g.sum()
            P[i, 6] = g.sum(); P[i, 7] = -g.sum()
        return P

    # Build the actual pooled features using raw and negated sign vectors
    P = np.zeros((N, 8), dtype=np.float64)
    for i in range(N):
        ids = x[i][mask[i]]
        if ids.size == 0:
            continue
        g = sg[ids]; gn = sgn_[ids]
        P[i, 0] = g.sum();  P[i, 1] = -g.sum()
        P[i, 2] = gn.sum(); P[i, 3] = -gn.sum()
        P[i, 4] = g.sum();  P[i, 5] = -g.sum()
        P[i, 6] = gn.sum(); P[i, 7] = -gn.sum()
    # ridge on pooled features (with intercept)
    Pb = np.hstack([P, np.ones((N, 1))])
    lam_h = 1.0
    Ah = Pb.T @ Pb + lam_h * np.eye(Pb.shape[1])
    bh = Pb.T @ t
    hw = np.linalg.solve(Ah, bh)
    w["head.weight"] = hw[:-1].astype(np.float32).reshape(2, D)
    w["head.bias"] = np.array([-hw[-1], hw[-1]], dtype=np.float32)

    # ---------------- 8. verify with our own forward pass ----------------
    def forward_acc():
        emb = w["embed.weight"]; posw = w["pos.weight"]
        # replicate: h = embed(ids)+pos ; blocks identity-ish ; final norm ; mean pool
        # because blocks are identity (with LayerNorm inside, they normalise each
        # token vector), replicate exactly:
        def ln(xx, g, b):
            m = xx.mean(axis=-1, keepdims=True)
            v = xx.var(axis=-1, keepdims=True)
            return (xx - m) / np.sqrt(v + 1e-5) * g + b
        H = emb[x] + posw[None, :, :]
        for bname in ("block0", "block1"):
            z = ln(H, w[bname + ".norm1.weight"], w[bname + ".norm1.bias"])
            qkv = z @ w[bname + ".qkv.weight"].T + w[bname + ".qkv.bias"]
            q, k, v = np.split(qkv, 3, axis=-1)
            # 2 heads of width 16
            att = np.zeros_like(v)
            for hh in range(2):
                sl = slice(hh * 16, (hh + 1) * 16)
                sc = q[..., sl] @ np.swapaxes(k[..., sl], -1, -2) / 4.0
                sc = np.where(mask[:, None, :], sc, -1e9)
                a = np.exp(sc - sc.max(axis=-1, keepdims=True))
                a = a / a.sum(axis=-1, keepdims=True)
                att[..., sl] = a @ v[..., sl]
            H = H + att @ w[bname + ".proj.weight"].T + w[bname + ".proj.bias"]
            z2 = ln(H, w[bname + ".norm2.weight"], w[bname + ".norm2.bias"])
            f = np.maximum(z2 @ w[bname + ".ff1.weight"].T + w[bname + ".ff1.bias"], 0.0)
            H = H + f @ w[bname + ".ff2.weight"].T + w[bname + ".ff2.bias"]
        Hn = ln(H, w["norm.weight"], w["norm.bias"])
        denom = np.maximum(mask.sum(axis=1, keepdims=True), 1)
        pooled = (Hn * mask[:, :, None]).sum(axis=1) / denom
        logits = pooled @ w["head.weight"].T + w["head.bias"]
        pred = logits.argmax(axis=1)
        return float((pred == y).mean()), logits

    acc_fwd, logits_fwd = forward_acc()

    # ---------------- 9. diagnostics ----------------
    diag = {
        "n_examples": int(N),
        "n_pos": npos, "n_neg": nneg,
        "tokens_used": int((ev > 0).sum()),
        "negators_found": int(negator.sum()),
        "negator_ids": [int(t) for t in np.where(negator > 0)[0][:40]],
        "s_absmax": float(np.abs(s).max()),
        "s_absmean_nonzero": float(np.abs(s[s != 0]).mean()),
        "feature_val_acc_evenodd": float(val_acc),
        "feature_lam": float(lam),
        "forward_acc_full_pool": acc_fwd,
        "head_weight_abs_max": float(np.abs(hw[:-1]).max()),
        "embed_abs_max": float(np.abs(embed_new).max()),
        "logit_abs_max": float(np.abs(logits_fwd).max()),
    }
    for k in list(w.keys()):
        arr = np.asarray(w[k], dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            arr = np.nan_to_num(arr, nan=0.0, posinf=1000.0, neginf=-1000.0)
        arr = np.clip(arr, -1000.0, 1000.0).astype(np.float32)
        w[k] = arr
    return {"weights": w, "diagnostics": diag}
