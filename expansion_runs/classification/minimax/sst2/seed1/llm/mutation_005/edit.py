import numpy as np

def edit(w, f):
    # ---------------- data ----------------
    x = np.asarray(f["x"], dtype=np.int64)          # [N,64] token ids
    y = np.asarray(f["y"], dtype=np.int64)          # [N]
    N, L = x.shape
    V = int(w["embed.weight"].shape[0])
    D = int(w["embed.weight"].shape[1])

    mask = (x != 0)
    lens = mask.sum(1).astype(np.float64)
    lens[lens == 0] = 1.0

    # ---------------- token statistics (permitted pool only) ----------------
    flat = x.reshape(-1)
    mflat = mask.reshape(-1)
    tok = flat[mflat]
    lab = np.repeat(y, L)[mflat]

    cnt = np.bincount(tok, minlength=V).astype(np.float64)
    c0 = np.bincount(tok[lab == 0], minlength=V).astype(np.float64)
    c1 = np.bincount(tok[lab == 1], minlength=V).astype(np.float64)

    # ---------------- negation handling ----------------
    NEG = np.zeros(V, dtype=bool)
    for t in ("n't", "not", "no", "never", "without", "hardly", "barely"):
        NEG[t] = True
    neg_tok = NEG[tok]

    # build per-position "negated" flag: a token is negated if a negation word
    # occurred earlier in the same sequence (within a 3-token window)
    negpos = np.zeros((N, L), dtype=bool)
    for i in range(N):
        row = x[i]
        act = False
        for p in range(L):
            if row[p] == 0:
                break
            if neg_tok[np.searchsorted(tok, 0)] if False else False:
                pass
            negpos[i, p] = act
            if NEG[row[p]]:
                act = True
    # (window: reset after 3 tokens)
    for i in range(N):
        row = x[i]
        last = -99
        for p in range(L):
            if row[p] == 0:
                break
            if NEG[row[p]]:
                last = p
            negpos[i, p] = (p - last) <= 3 and last >= 0

    # ---------------- NB log-count-ratio with Laplace smoothing ----------------
    alpha = 1.0
    tot0 = c0.sum() + alpha * V
    tot1 = c1.sum() + alpha * V
    p0 = (c0 + alpha) / tot0
    p1 = (c1 + alpha) / tot1
    s = np.log(p1) - np.log(p0)          # per-token log odds
    s[0] = 0.0                            # pad token contributes nothing

    # document score with negation flip
    def doc_scores(sv, negflip):
        sc = np.zeros(N, dtype=np.float64)
        for i in range(N):
            row = x[i]
            n = int(lens[i])
            if n == 0:
                continue
            tt = row[:n]
            val = sv[tt]
            if negflip:
                nf = negpos[i, :n]
                val = np.where(nf, -val, val)
            sc[i] = val.sum()
        return sc

    s_plain = doc_scores(s, False)
    s_neg = doc_scores(s, True)

    # ---------------- ridge on binary bag-of-words (closed form) ----------------
    # design matrix: binary presence of each token (plus negated-presence channel)
    Xb = np.zeros((N, V), dtype=np.float64)
    for i in range(N):
        row = x[i]
        n = int(lens[i])
        if n:
            Xb[i, row[:n]] = 1.0
    Xn = np.zeros((N, V), dtype=np.float64)
    for i in range(N):
        row = x[i]
        n = int(lens[i])
        if n:
            Xn[i, row[:n][negpos[i, :n]]] = 1.0
    X = np.hstack([Xb, Xn]) if Xn.any() else Xb
    t = np.where(y == 1, 1.0, -1.0)

    def ridge_fit(Xd, td, lam):
        A = Xd.T @ Xd
        A[np.diag_indices_from(A)] += lam
        b = Xd.T @ td
        return np.linalg.solve(A, b)

    def acc_of(score):
        return float(np.mean((score > 0).astype(np.int64) == y))

    # 2-fold CV for the ridge model (deterministic split)
    idx = np.arange(N)
    fold = idx % 2
    best_lam, best_cv = None, -1.0
    for lam in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0):
        sc = np.zeros(N)
        for fo in (0, 1):
            tr = fold != fo
            te = ~tr
            wv = ridge_fit(X[tr], t[tr], lam)
            sc[te] = X[te] @ wv
        cv = acc_of(sc)
        if cv > best_cv:
            best_cv, best_lam = cv, lam
    ridge_w = ridge_fit(X, t, best_lam)
    ridge_sc = X @ ridge_w
    ridge_acc = acc_of(ridge_sc)

    # ---------------- pick the better classifier ----------------
    cand = {
        "nb_plain": s_plain,
        "nb_neg": s_neg,
        "ridge": ridge_sc,
    }
    # in-sample accuracy of each; also 2-fold CV accuracy for the NB models
    def cv_nb(sv):
        sc = np.zeros(N)
        for fo in (0, 1):
            tr = fold != fo
            te = ~tr
            # refit NB log-odds on training half
            tt = np.repeat(np.arange(N), L)[mflat]
            ltr = np.repeat(y, L)[mflat]
            sel = tr[tt]
            cc0 = np.bincount(tok[sel & (ltr == 0)], minlength=V).astype(np.float64)
            cc1 = np.bincount(tok[sel & (ltr == 1)], minlength=V).astype(np.float64)
            aa = 1.0
            pp0 = (cc0 + aa) / (cc0.sum() + aa * V)
            pp1 = (cc1 + aa) / (cc1.sum() + aa * V)
            ss = np.log(pp1) - np.log(pp0)
            ss[0] = 0.0
            sc[te] = doc_scores_sub(ss, x, negpos, lens, te)
        return acc_of(sc)

    def doc_scores_sub(sv, xx, npg, ln, sel):
        out = np.zeros(N)
        for i in np.nonzero(sel)[0]:
            n = int(ln[i])
            if n == 0:
                continue
            tt = xx[i, :n]
            val = sv[tt]
            val = np.where(npg[i, :n], -val, val)
            out[i] = val.sum()
        return out

    cv_nb_plain = cv_nb(s)
    cv_nb_neg = cv_nb(s)

    scores = {
        "nb_plain": (s_plain, acc_of(s_plain), cv_nb_plain),
        "nb_neg": (s_neg, acc_of(s_neg), cv_nb_neg),
        "ridge": (ridge_sc, ridge_acc, best_cv),
    }
    # choose by CV accuracy (generalisation proxy), tie-break by in-sample
    best = max(scores.items(), key=lambda kv: (kv[1][2], kv[1][1]))
    name = best[0]
    sc = best[1][0]

    # ---------------- build the network that realises this score ----------------
    # per-token channel value v_t (signed), pooled = mean over tokens of v_t
    # final LayerNorm turns (v,-v,0..) into 4*sign(v)*(1,-1,0..) -> loses magnitude,
    # so instead we make the *pre-norm* vector direction-fixed per token type:
    #   embed[t] = (a_t, -a_t, 0...) with a_t = |v_t| + eps  -> LN gives sign only.
    # To keep magnitude we use the FF block: relu is monotone, so we can encode
    # v_t through a *pair* of channels whose ratio survives LayerNorm.
    # Simpler & exact: use the FF nonlinearity to build a per-token vector whose
    # LayerNorm-normalised direction is fixed and whose scale is carried by a
    # second channel pair.  We instead use the trick below:
    #   embed[t] = (v_t, v_t, 0...)  -> LN -> (d, d, 0...) with d = sqrt(32/2)*sign(v)
    # still sign-only.  => use two-token-type trick:
    #   encode v_t in a 3-channel pattern (v_t, |v_t|, 0...) so that after LN the
    #   direction depends on sign(v_t) only.  Therefore we must carry the score
    # through the *pooled* channel, not through per-token LN.
    # Practical exact construction:
    #   embed[t] = (v_t, -v_t, 0...0)  (32 dims, dims 0,1 used)
    #   blocks: proj/ff zeroed  -> h unchanged
    #   final norm: gamma=1, beta=0 -> LN(embed[t]) = 4*sign(v_t)*(1,-1,0..)
    #   => pooled channel = 4*mean(sign(v_t)) -> sign vote (what we have now).
    # To realise the *sum* we instead make the head read a channel that is
    # proportional to sum_t v_t by using the attention block: set qkv so that
    # attention output at each position = mean of v over the sequence, and then
    # head reads it.  (pooled mean over positions of the sequence-mean = same)
    # => instead we exploit: pooled = mean_t c_t ; choose c_t = v_t * e  and
    #    multiply by a per-document factor via the *bias* of the final norm is
    #    impossible, but the 1/L factor is positive and does not change argmax.
    # So: implement c_t = v_t * e  with e=(1,-1,0..) using the FF block so that
    # the final LayerNorm does not destroy the magnitude:
    #   FF: ff1 maps the token's own channel to (v_t, |v_t|) pattern, relu keeps
    #   sign info, ff2 maps to (v_t, -v_t) direction; then final LN is applied to
    #   (embed + ffout) whose direction is fixed per sign -> magnitude lost again.
    # FINAL: we bypass LN magnitude loss by making the *pooled* statistic a sum
    # of per-token vectors that are all positive multiples of a common direction:
    #   embed[t] = (|v_t|, |v_t|, 0..)  and let the head weight encode sign via
    #   two separate channels is impossible (one vector per token).
    # => Use the sign-vote channel PLUS a magnitude channel obtained by making
    #    the token vector proportional to (v_t, 0, 0..) and the *final* norm's
    #    per-dimension gamma/beta chosen so that the normalised output equals
    #    (v_t, 0, 0..) exactly.  This is possible because we can pick gamma,beta
    #    and the embedding scale jointly.
    #
    # Construction that is exact:
    #   embed[t] = (v_t, v_t, 0...0)  (dims 0,1)
    #   final norm: z = LN(embed[t]) = (s, s, 0..)/... with s = sign(v_t)*sqrt(16)
    #   -> we want output (v_t, 0, 0..).  Choose gamma_d and beta_d:
    #      out_d = gamma_d * z_d + beta_d
    #      z = (4*sign(v_t), 4*sign(v_t), 0..)  (since var over 32 dims = 2v^2/32,
    #      std = |v|/4, so (v-mean)/std = 4*sign(v))
    #      magnitude is gone -> cannot recover v_t.
    # => Therefore use the FF block to add a *magnitude* channel that survives:
    #      ff1/ff2 are applied AFTER the first LayerNorm, and the residual
    #      h = h + ff2(relu(ff1(LN(h)))) is added to h, so the FF output can carry
    #      the true signed magnitude in a channel that the final LN then sees
    #      together with a fixed direction.  We make the FF output = (v_t, -v_t,0..)
    #      and the embedding = (1, -1, 0..)*eps so that the *sum* h = embed+ffout
    #      has direction (v_t, -v_t) whose LN is again sign-only.
    #      => Instead make FF output = (v_t, 0, 0..) and embed = (0, 0, ..) so
    #      h = (v_t,0,0..) -> LN -> sign-only again.
    # Conclusion: a per-token LayerNorm always destroys per-token magnitude.
    # The only magnitude-preserving route is to make the pooled statistic a
    # *sum* over tokens of a fixed-direction vector, i.e. use positions:
    #   pos.weight[p] can carry the document-position structure, but the token
    #   identity enters only through embed.
    # PRACTICAL SOLUTION: use the *attention* block to compute a sequence-level
    # (sum-like) statistic: with q=0 (uniform attention over non-pad keys) and
    # v = (v_t, -v_t, 0..), the attention output at every position equals the
    # mean over the sequence of v_t; then the FF block can add a *second* copy
    # with a different scaling, and the final LayerNorm sees a vector whose
    # direction is FIXED (independent of the document) and whose magnitude
    # encodes the sum -> after LN we lose it again...
    # => The clean, LN-safe way: put the score in the *pooled* vector via the
    #    head, and make the pooled vector's informative direction be the mean of
    #    per-token *fixed-direction* vectors: pooled = (1/L) sum_t c_t.  Since
    #    argmax is invariant to positive per-document scaling, we only need
    #    pooled_d to be a positive multiple of the desired score.  A sign-vote
    #    gives sign only, but we can use *magnitude buckets*: assign each token
    #    a fixed direction vector whose magnitude encodes |s_t| in one of a few
    #    channels, and combine several channels in the head.
    #    We use K=8 channels: channel k gets +1 if s_t > k*step, -1 if s_t < -k*step.
    #    Then pooled channel k = (1/L)*sum_t sign(s_t - k*step) which is a
    #    monotone-in-|s| surrogate of the log-odds sum, and the head combines
    #    them with fitted weights -> recovers a smooth score.
    #
    # (This is the construction implemented below.)

    # ---- magnitude-bucket encoding ----
    step = float(np.max(np.abs(s)) / 8.0 + 1e-6)
    K = 8
    # per-token channel values: for token t, channel k value = sign(s_t - k*step)
    # we build embed rows as (val_k, -val_k) pairs in dims 0..2K-1 (K<=16)
    # but LayerNorm per token destroys magnitude again -> use SIGN-VOTE per
    # channel (each channel is a separate sign), which survives LN because the
    # whole token vector is normalised jointly: we need each channel to keep its
    # own sign.  Trick: put each bucket's sign in its own *pair* of dims and
    # accept that LN rescales all dims by the same factor (a per-token positive
    # factor) -> the *relative* signs across channels are preserved, and the
    # head (linear) then sums them with fitted weights.  The per-token factor
    # is positive -> pooled channels keep their sign pattern.
    # => embed[t] = concat_k ( b_k(t), -b_k(t) ) where b_k(t)=sign(s_t-k*step)
    #    plus a magnitude-carrying pair: (|s_t| scaled, -|s_t| scaled) is lost.
    #    Use instead a *rank* encoding: b_k(t)=sign(s_t - k*step) for K=8 thresholds
    #    -> the vector (b_1..b_8) determines |s_t| up to the bucket, and the head
    #    weights (fitted by least squares on the pool) reconstruct the score.
    #    We fit head weights by closed-form least squares on the realised
    #    pooled features -> this is a linear fit, allowed.

    # Build the realised per-token channel vectors
    thr = np.linspace(-np.max(np.abs(s)), np.max(np.abs(s)), K + 1)[1:-1]
    B = np.zeros((V, K), dtype=np.float64)     # bucket signs per token
    for k in range(K):
        B[:, k] = np.sign(s - thr[k])
    B[0, :] = 0.0

    # per-token embedding rows: pairs (b_k, -b_k) in dims 0..2K-1
    E = np.zeros((V, D), dtype=np.float64)
    for k in range(K):
        E[:, 2 * k] = B[:, k]
        E[:, 2 * k + 1] = -B[:, k]
    # scale so that per-token vector has zero mean & unit-ish variance handled by LN
    # (embeddings for PAD/UNK rows are zero -> LN of zero vector is zero; fine)

    # blocks: zero out everything so h passes through unchanged
    for b in ("block0", "block1"):
        for kk in ("qkv.weight", "qkv.bias", "proj.weight", "proj.bias",
                   "ff1.weight", "ff1.bias", "ff2.weight", "ff2.bias"):
            key = b + "." + kk
            if key in w:
                w[key] = np.zeros_like(w[key])
        for kk in ("norm1.weight", "norm2.weight"):
            key = b + "." + kk
            if key in w:
                w[key] = np.ones_like(w[key])
        for kk in ("norm1.bias", "norm2.bias"):
            key = b + "." + kk
            if key in w:
                w[key] = np.zeros_like(w[key])
    w["pos.weight"] = np.zeros_like(w["pos.weight"])

    # ---- realised pooled features (analytic, matches forward pass) ----
    # per token t: LN(embed[t]) -> we must reproduce exactly.
    # embed[t] has 2K non-zero dims (values +-1 or 0), rest 0.
    # mean over dims = 0, var = (sum of squares)/32 = (2*sum_k b_k^2)/32
    # LN -> z_d = e_d * sqrt(32/(2*sum_k b_k^2)) for e_d in {+1,-1}
    # final norm: gamma=1,beta=0 -> out = z
    # pooled = mean over non-pad tokens of out
    # head: logits = W @ pooled + b
    def pooled_features(Ew, xx, ln):
        Nn = xx.shape[0]
        P = np.zeros((Nn, D), dtype=np.float64)
        for i in range(Nn):
            n = int(ln[i])
            if n == 0:
                continue
            rows = Ew[xx[i, :n]]
            ss = (rows * rows).sum(1)
            scale = np.zeros(n)
            nz = ss > 0
            scale[nz] = np.sqrt(32.0 / ss[nz])
            P[i] = (rows * scale[:, None]).sum(0) / n
        return P

    P = pooled_features(E, x, lens)

    # fit head by closed-form least squares on the realised pooled features
    # (target = +/-1 ; also try target = sign of the chosen score)
    def fit_head(Pd, td, lam):
        A = Pd.T @ Pd
        A[np.diag_indices_from(A)] += lam
        b = Pd.T @ td
        return np.linalg.solve(A, b)

    tgt = np.where(sc > 0, 1.0, -1.0)
    best = None
    for lam in (1e-6, 1e-4, 1e-3, 1e-2, 1e-1):
        Wl = fit_head(P, tgt, lam)
        sc_fit = P @ Wl
        a = acc_of(sc_fit)
        if best is None or a > best[0]:
            best = (a, lam, Wl)
    head_acc, head_lam, Wl = best

    # also try fitting directly on the raw score (regression), then sign
    sc_raw = sc / (np.abs(sc).max() + 1e-9)
    Wl2 = fit_head(P, sc_raw, 1e-4)
    a2 = acc_of(P @ Wl2)
    if a2 > head_acc:
        head_acc, Wl = a2, Wl2

    # ---- write weights ----
    w["embed.weight"] = E.astype(np.float32)
    w["norm.weight"] = np.ones_like(w["norm.weight"])
    w["norm.bias"] = np.zeros_like(w["norm.bias"])
    w["head.weight"] = Wl.astype(np.float32)
    w["head.bias"] = np.zeros_like(w["head.bias"])

    # ---- diagnostics ----
    def j(a):
        return [float(v) for v in np.asarray(a).ravel()[:64]]

    diag = {
        "design": "bucket-sign BoW log-odds encoded in embed dims (2k,2k+1); blocks zeroed; head = closed-form LS on realised pooled features",
        "chosen_classifier": name,
        "pool_acc_nb_plain": acc_of(s_plain),
        "pool_acc_nb_neg": acc_of(s_neg),
        "pool_acc_ridge": ridge_acc,
        "cv_nb_plain": cv_nb_plain,
        "cv_nb_neg": cv_nb_neg,
        "cv_ridge": best_cv,
        "ridge_lam": float(best_lam),
        "realised_head_acc": head_acc,
        "head_lam": float(head_lam),
        "K": K,
        "step": step,
        "s_absmax": float(np.max(np.abs(s))),
        "neg_token_frac": float(negpos.sum() / max(1, mask.sum())),
        "mean_len": float(lens.mean()),
        "embed_abs_max": float(np.abs(E).max()),
        "head_abs_max": float(np.abs(Wl).max()),
        "head_w": j(Wl),
        "top_tokens_by_abs_s": [
            [int(t), round(float(s[t]), 4), int(cnt[t])]
            for t in np.argsort(-np.abs(s))[:20]
        ],
    }
    return {"weights": w, "diagnostics": diag}
