import numpy as np

def edit(w, f):
    # ------------------------------------------------------------------
    # 0. helpers
    # ------------------------------------------------------------------
    def L(x):
        x = np.asarray(x, dtype=np.float64)
        m = x.mean(-1, keepdims=True)
        v = x.var(-1, keepdims=True)
        return (x - m) / np.sqrt(v + 1e-5)

    def lin(x, W, b):
        return x @ np.asarray(W, dtype=np.float64).T + np.asarray(b, dtype=np.float64)

    def softmax(a, axis=-1):
        a = a - a.max(axis=axis, keepdims=True)
        e = np.exp(a)
        return e / e.sum(axis=axis, keepdims=True)

    def forward(ids, P, ret_h=False):
        """Re-implementation of the documented forward pass."""
        B, Lp = ids.shape
        E = P["embed.weight"]; POS = P["pos.weight"]
        h = E[ids] + POS[None, :, :]                       # [B,L,32]
        mask = (ids != 0)
        for blk in ("block0", "block1"):
            z = L(h) * P[blk + ".norm1.weight"] + P[blk + ".norm1.bias"]
            qkv = lin(z, P[blk + ".qkv.weight"], P[blk + ".qkv.bias"])
            q, k, v = np.split(qkv, 3, axis=-1)
            B_, L_, D = q.shape
            nh, hd = 2, D // 2
            out = np.zeros_like(v)
            for hh in range(nh):
                qs = q[:, :, hh * hd:(hh + 1) * hd]
                ks = k[:, :, hh * hd:(hh + 1) * hd]
                vs = v[:, :, hh * hd:(hh + 1) * hd]
                sc = qs @ np.swapaxes(ks, 1, 2) / 4.0
                sc = np.where(mask[:, None, :], sc, -1e30)
                out[:, :, hh * hd:(hh + 1) * hd] = softmax(sc, -1) @ vs
            h = h + lin(out, P[blk + ".proj.weight"], P[blk + ".proj.bias"])
            z2 = L(h) * P[blk + ".norm2.weight"] + P[blk + ".norm2.bias"]
            a1 = np.maximum(lin(z2, P[blk + ".ff1.weight"], P[blk + ".ff1.bias"]), 0.0)
            h = h + lin(a1, P[blk + ".ff2.weight"], P[blk + ".ff2.bias"])
        hn = L(h) * P["norm.weight"] + P["norm.bias"]
        m = mask[:, :, None].astype(np.float64)
        pooled = (hn * m).sum(1) / np.maximum(m.sum(1), 1e-9)
        logits = lin(pooled, P["head.weight"], P["head.bias"])
        if ret_h:
            return logits, pooled, hn, h
        return logits

    # ------------------------------------------------------------------
    # 1. read current weights
    # ------------------------------------------------------------------
    P = {k: np.asarray(v, dtype=np.float64) for k, v in w.items()}
    x = np.asarray(f["x"], dtype=np.int64)
    y = np.asarray(f["y"], dtype=np.int64)
    N, Lp = x.shape
    V = P["embed.weight"].shape[0]

    # sanity: does my forward match the provided activations?
    chk = forward(x[:8], P)
    ref = np.asarray(f["logits"][:8], dtype=np.float64)
    fwd_err = float(np.max(np.abs(chk - ref)))

    # ------------------------------------------------------------------
    # 2. bag-of-words design matrix (counts) + token statistics
    # ------------------------------------------------------------------
    rows = np.repeat(np.arange(N), Lp)
    cols = x.reshape(-1)
    keep = cols > 0
    rows = rows[keep]; cols = cols[keep]
    Xc = np.zeros((N, V), dtype=np.float64)
    np.add.at(Xc, (rows, cols), 1.0)
    Xb = (Xc > 0).astype(np.float64)
    lens = Xc.sum(1)
    y1 = (y == 1).astype(np.float64)
    pos_cnt = Xc[y1 == 1].sum(0)
    neg_cnt = Xc[y1 == 0].sum(0)
    pos_doc = Xb[y1 == 1].sum(0)
    neg_doc = Xb[y1 == 0].sum(0)

    # ------------------------------------------------------------------
    # 3. closed-form ridge on BoW (counts), lambda by out-of-fold accuracy
    # ------------------------------------------------------------------
    def ridge_fit(Xd, t, lam):
        Xc_ = Xd - Xd.mean(0)
        A = Xc_.T @ Xc_ + lam * np.eye(Xd.shape[1])
        b = Xc_.T @ (t - t.mean())
        beta = np.linalg.solve(A, b)
        return beta, float(t.mean())

    def acc_of(Xd, t, beta, b0):
        s = Xd @ beta + b0
        return float(((s > 0).astype(np.float64) == t).mean())

    # 5-fold out-of-fold accuracy for a small lambda grid (closed-form solves)
    rng = np.random.default_rng(0)
    fold = rng.permutation(N) % 5
    lams = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    oof = {}
    for lam in lams:
        pred = np.zeros(N)
        for kf in range(5):
            tr = fold != kf; te = ~tr
            beta, b0 = ridge_fit(Xc[tr], y1[tr], lam)
            pred[te] = Xc[te] @ beta + b0
        oof[lam] = float(((pred > 0).astype(np.float64) == y1).mean())
    best_lam = max(oof, key=lambda L: oof[L])

    beta_c, b0_c = ridge_fit(Xc, y1, best_lam)
    beta_b, b0_b = ridge_fit(Xb, y1, best_lam)
    in_acc_c = acc_of(Xc, y1, beta_c, b0_c)
    in_acc_b = acc_of(Xb, y1, beta_b, b0_b)

    # NB log-odds reference (what the current model effectively uses)
    alpha_nb = 1.0
    p_pos = (pos_cnt + alpha_nb) / (pos_cnt.sum() + alpha_nb * V)
    p_neg = (neg_cnt + alpha_nb) / (neg_cnt.sum() + alpha_nb * V)
    r_nb = np.log(p_pos / p_neg)
    in_acc_nb = float((((Xc @ r_nb) > 0).astype(np.float64) == y1).mean())

    # ------------------------------------------------------------------
    # 4. per-token indicators: negation, intensifier, and neighbour score
    # ------------------------------------------------------------------
    NEG_WORDS = ["not", "n't", "no", "never", "without", "hardly", "barely",
                 "nothing", "neither", "nor", "cannot", "lacks", "lacking",
                 "fails", "failed", "unfunny", "unwatchable", "unoriginal",
                 "uninspired", "unconvincing", "unforgettable", "unfaithful",
                 "unsettling", "unpleasant", "unhappy", "uncomfortable",
                 "unbelievable", "unpredictable", "unremarkable", "unsympathetic",
                 "unappealing", "unattractive", "uninspired", "unnecessary",
                 "unnatural", "unrealistic", "unwarranted", "unproductive",
                 "uninteresting", "uninvolved", "uninhabited", "unreliable",
                 "unrewarding", "unruly", "unsolved", "unspoiled", "unsteady",
                 "unstoppable", "unsupported", "untenable", "unverified",
                 "unworthy", "avoid", "avoided", "lacks", "refuses", "refused",
                 "denies", "denied", "reject", "rejects", "rejected"]
    INT_WORDS = ["very", "too", "really", "quite", "so", "extremely", "highly",
                 "remarkably", "especially", "incredibly", "absurdly", "utterly",
                 "totally", "completely", "utterly", "deeply", "terribly",
                 "awfully", "insanely", "ridiculously", "unbelievably",
                 "exceedingly", "extraordinarily", "hugely", "immensely",
                 "incredibly", "rather", "fairly", "somewhat", "slightly",
                 "barely", "hardly", "scarcely", "mildly", "notably"]
    neg_ids = np.zeros(V, dtype=bool)
    int_ids = np.zeros(V, dtype=bool)
    # token id -> word is not available as an array; derive from the vocabulary
    # list embedded in the feedback examples is not accessible, so instead use
    # a data-driven proxy: tokens whose class-conditional log-odds are extreme
    # AND which frequently occur immediately before a high-|score| token.
    # (see step 4b)

    # 4b. data-driven "negator" discovery: tokens that frequently precede
    # strongly-opinionated tokens, and whose own unigram score is weak.
    prev_tok = np.zeros((N, Lp), dtype=np.int64)
    prev_tok[:, 1:] = x[:, :-1]
    big = np.zeros((V, V), dtype=np.float64)
    np.add.at(big, (prev_tok.reshape(-1), x.reshape(-1)), 1.0)
    prev_cnt = big.sum(1)
    prev_cnt[prev_cnt == 0] = 1.0
    # score of the following token, aggregated over occurrences of each token
    follow_score = (big @ r_nb) / prev_cnt
    # tokens that are "negators": their own unigram score is near zero but the
    # tokens they precede are strongly opinionated, and they appear often.
    own = np.abs(r_nb)
    cand = (own < 0.6) & (np.abs(follow_score) > 0.8) & (prev_cnt >= 8)
    # keep the strongest such tokens
    order = np.argsort(-(np.abs(follow_score) * np.minimum(prev_cnt, 50)))
    neg_ids = np.zeros(V, dtype=bool)
    neg_ids[order[:60]] = cand[order[:60]]
    # intensifiers: tokens that precede strongly opinionated tokens and are
    # themselves frequent, near-neutral words
    cand2 = (own < 1.0) & (np.abs(follow_score) > 0.8) & (prev_cnt >= 15) & (~neg_ids)
    order2 = np.argsort(-(np.abs(follow_score) * np.minimum(prev_cnt, 80)))
    int_ids = np.zeros(V, dtype=bool)
    int_ids[order2[:40]] = cand2[order2[:40]]

    # per-token "intensity" indicator: how often token t is followed by a
    # strongly opinionated token (used as a soft intensifier channel)
    strong = (np.abs(r_nb) > 1.0).astype(np.float64)
    follows_strong = (big @ strong) / prev_cnt
    int_ids = int_ids | (follows_strong > 0.6)

    # ------------------------------------------------------------------
    # 5. build the *simulated* pooled feature channels and fit the head
    #    channels:
    #      ch0 : sum of per-token score a_t            (plain BoW)
    #      ch1 : sum of a_t * neg_t                   (negation flip)
    #      ch2 : sum of a_t * int_t                   (intensifier)
    #      ch3 : sum of a_t * neg_{t-1}  (negator applies to next token)
    #      ch4 : sum of a_t * int_{t-1}
    #      ch5 : length (bias-like)
    # ------------------------------------------------------------------
    # per-token score used inside the network: the ridge BoW weight (signed)
    a_tok = beta_c.copy()                      # signed, per-token contribution
    a_tok = a_tok / (np.abs(a_tok).max() + 1e-9) * 2.5   # keep carrier valid

    def build_channels(a_vec, neg_vec, int_vec):
        """Return [N,6] simulated pooled-feature channels."""
        A = a_vec[x]                                   # [N,L]
        NEG = neg_vec[x].astype(np.float64)
        INT = int_vec[x].astype(np.float64)
        prevA = np.zeros_like(A); prevA[:, 1:] = A[:, :-1]
        prevNEG = np.zeros_like(NEG); prevNEG[:, 1:] = NEG[:, :-1]
        prevINT = np.zeros_like(INT); prevINT[:, 1:] = INT[:, :-1]
        m = (x != 0).astype(np.float64)
        s = m.sum(1)
        ch = np.stack([
            (A * m).sum(1),
            (A * NEG * m).sum(1),
            (A * INT * m).sum(1),
            (prevA * prevNEG * m).sum(1),
            (prevA * prevINT * m).sum(1),
            s,
        ], axis=1)
        return ch

    CH = build_channels(a_tok, neg_ids, int_ids)
    # standardize channels for the ridge fit
    mu = CH.mean(0); sd = CH.std(0); sd[sd < 1e-9] = 1.0
    CHs = (CH - mu) / sd
    beta_h, b0_h = ridge_fit(CHs, y1, best_lam)
    in_acc_full = acc_of(CHs, y1, beta_h, b0_h)

    # out-of-fold accuracy of the full channel model (same folds)
    pred_full = np.zeros(N)
    for kf in range(5):
        tr = fold != kf; te = ~tr
        bh, b0 = ridge_fit(CHs[tr], y1[tr], best_lam)
        pred_full[te] = CHs[te] @ bh + b0
    oof_full = float(((pred_full > 0).astype(np.float64) == y1).mean())

    # ------------------------------------------------------------------
    # 6. realise the channels inside the fixed architecture
    # ------------------------------------------------------------------
    # Embedding rows: zero-mean, norm exactly 3, dim0 = a_t (score channel).
    #   dims 1..31 carry a constant carrier so that LayerNorm keeps dim0
    #   proportional to a_t (this is what the current model already does).
    #   dim 1 also carries the negation indicator, dim 2 the intensifier.
    def carrier(a):
        # solve a + b + 30c = 0 and a^2 + b^2 + 30c^2 = 9 for (b, c)
        disc = 33480.0 - 3840.0 * a * a
        disc = np.maximum(disc, 1e-6)
        b = (-2.0 * a + np.sqrt(disc)) / 62.0
        c = (-a - b) / 30.0
        return b, c

    E_new = np.zeros((V, 32), dtype=np.float64)
    a_all = a_tok.copy()
    b_all, c_all = carrier(a_all)
    E_new[:, 0] = a_all
    E_new[:, 1] = b_all
    E_new[:, 2:] = c_all[:, None]

    # --- block0: attention delivers the PREVIOUS token's score -------------
    # qkv weight [96,32]: rows 0..31 q, 32..63 k, 64..95 v
    # q_i = e0 (dim0) ; k_j = e0 ; v_j = [a_j, neg_j, int_j, 0...]
    # -> attention output at i ~ value of position i-1 (peaked, since
    #    q_i.k_j = a_j is large in magnitude for opinionated tokens).
    # To keep it a *selection* of the previous token we instead use a
    # dedicated "previous" channel: v dim0 = a_j, and we let the softmax
    # pick the position with the largest |a|; we then only use the
    # *sign-carrying* combination via the FFN products.
    # Simpler & robust: use attention to move a_j to position i+1 by
    # making the attention essentially identity (q,k ~ one-hot on dim0 of a
    # *position-independent* key).  We realise "previous" exactly with an
    # identity-like attention: q = k = e0 * BIG would be constant.
    # Instead we use the FFN-free approach: the previous-token product is
    # built from the *current* token's own neg/int flag times the *current*
    # token's score, and additionally the attention provides the neighbour
    # score through a separate additive channel.
    # Concretely:
    #   embed dim0 = a_t              (score)
    #   embed dim1 = neg_t (0/1)      (negator flag)
    #   embed dim2 = int_t (0/1)      (intensifier flag)
    #   attention (block0) output dim0 = a_{prev}  (delivered by a peaked
    #      attention that selects the previous position)
    #   block0 FFN forms products: neg_t * a_{prev}  and int_t * a_{prev}
    #   -> these are added to the residual stream, so the final pooled
    #      vector contains sum_t a_t, sum_t neg_t a_t, sum_t int_t a_t,
    #      sum_t neg_t a_{prev}, sum_t int_t a_{prev}.
    #
    # Realising "previous position" with attention:
    #   q_i = e0 * s_i where s_i = a_i (dim0 of embed) -> no.
    #   Use: k_j = e0 * BIGONE (constant), q_i = e0 * a_i  => logits_j = a_i
    #   -> softmax over j is independent of j => uniform. Not useful.
    #   Use: q_i = e0, k_j = e0 * a_j -> logits_j = a_j/4 -> softmax picks
    #   the globally most extreme token, same for every i. Not "previous".
    #   => attention cannot express "previous" with a linear q/k map.
    #   Therefore we realise the neighbour channel with a *positional* trick:
    #   the "previous" score is delivered by attention as a *soft* aggregate
    #   and the exact neighbour product is realised by the FFN using the
    #   current token's flags and the *current* token's score.
    #   We therefore use channels:
    #     ch0 = sum a_t
    #     ch1 = sum neg_t * a_t
    #     ch2 = sum int_t * a_t
    #     ch3 = sum neg_t * a_{prev}   -> replaced by sum neg_t*a_t^2 style?
    #   To keep everything exact and closed-form, we restrict the realised
    #   channels to {sum a_t, sum neg_t a_t, sum int_t a_t, sum a_t^2-ish}
    #   and refit the head on exactly those.
    # ------------------------------------------------------------------
    # Re-derive the channel set to exactly what the architecture can express:
    #   C0 = sum_t a_t
    #   C1 = sum_t neg_t * a_t
    #   C2 = sum_t int_t * a_t
    #   C3 = sum_t a_t^2 * |neg_t|  (magnitude channel, sign-free)
    #   C4 = length
    # The FFN can realise C1, C2, C3 as position-wise products of the
    # per-token quantities (a_t, neg_t, int_t) which are all available in
    # the residual stream after the embedding.
    def build_channels2(a_vec, neg_vec, int_vec):
        A = a_vec[x]
        NEG = neg_vec[x].astype(np.float64)
        INT = int_vec[x].astype(np.float64)
        m = (x != 0).astype(np.float64)
        ch = np.stack([
            (A * m).sum(1),
            (A * NEG * m).sum(1),
            (A * INT * m).sum(1),
            (A * A * m).sum(1),
            m.sum(1),
        ], axis=1)
        return ch

    CH2 = build_channels2(a_tok, neg_ids, int_ids)
    mu2 = CH2.mean(0); sd2 = CH2.std(0); sd2[sd2 < 1e-9] = 1.0
    CH2s = (CH2 - mu2) / sd2
    bh2, b02 = ridge_fit(CH2s, y1, best_lam)
    in_acc2 = acc_of(CH2s, y1, bh2, b02)
    pred2 = np.zeros(N)
    for kf in range(5):
        tr = fold != kf; te = ~tr
        b_, b0_ = ridge_fit(CH2s[tr], y1[tr], best_lam)
        pred2[te] = CH2s[te] @ b_ + b0_
    oof2 = float(((pred2 > 0).astype(np.float64) == y1).mean())

    # choose the channel set with the better out-of-fold accuracy
    use_full = oof_full >= oof2
    if use_full:
        CHf, muf, sdf, bhf, b0f, oof_chosen = CHs, mu, sd, beta_h, b0_h, oof_full
    else:
        CHf, muf, sdf, bhf, b0f, oof_chosen = CH2s, mu2, sd2, bh2, b02, oof2

    # ------------------------------------------------------------------
    # 7. realise the chosen channels with the architecture
    # ------------------------------------------------------------------
    # We realise, per token t, the residual-stream vector
    #   v_t = [ a_t , neg_t , int_t , 0 ... ]  (dims 0,1,2)
    # and the FFN of block0 produces products:
    #   p_t = [ neg_t*a_t , int_t*a_t , a_t^2 ]  written into dims 3,4,5
    # The pooled (pre-LayerNorm) sum over tokens therefore contains exactly
    #   C0 = sum a_t, C1 = sum neg*a, C2 = sum int*a, C3 = sum a^2, plus
    #   the raw sums of neg and int (used as extra channels).
    # The head then reads the pooled vector linearly.
    #
    # IMPORTANT: the final LayerNorm is applied per token, so we must keep a
    # constant-norm, zero-mean carrier in the residual stream so that the
    # informative components survive.  We therefore keep the embed carrier
    # (dims 3..31 constant) and let the FFN write its products into dims
    # 3..5 while keeping dims 6..31 as the carrier continuation.
    # To keep the per-token row exactly zero-mean and norm-3 after the FFN
    # we instead give the FFN output a *compensating* structure: we design
    # the FFN to output a vector that is zero-mean and norm-preserving is
    # impossible in general, so we instead rely on the fact that LayerNorm
    # only needs the *ratio* of informative component to row scale to be
    # stable.  We therefore (i) keep the embed carrier large (norm 3) and
    # (ii) keep the FFN products small relative to it, so the row scale is
    # dominated by the carrier and the informative dims keep their sign and
    # relative magnitude.
    # The head is then fitted on the *simulated* pooled features obtained by
    # running the real forward pass (below), so any distortion is absorbed
    # by the closed-form fit.

    # ---- embed: dim0 = a_t, dim1 = neg_t, dim2 = int_t, dims 3..31 carrier
    E2 = np.zeros((V, 32), dtype=np.float64)
    a_s = a_tok * 0.25                     # keep a small vs carrier
    b_s, c_s = carrier(a_s)
    E2[:, 0] = a_s
    E2[:, 1] = neg_ids.astype(np.float64) * 0.05
    E2[:, 2] = int_ids.astype(np.float64) * 0.05
    # carrier occupies dims 3..31 (29 dims) -> recompute for 29-dim carrier
    def carrier29(a):
        # a + b + 29c = 0 ; a^2 + b^2 + 29c^2 = 9
        disc = 9.0 * 30.0 - (1.0 + 29.0) * a * a
        disc = np.maximum(disc, 1e-6)
        b = (-2.0 * a + np.sqrt(disc)) / 30.0
        c = (-a - b) / 29.0
        return b, c
    b2s, c2s = carrier29(a_s)
    E2[:, 3] = b2s
    E2[:, 4:] = c2s[:, None]
    # NOTE dims 1,2 carry the flags; the carrier is dims 3..31 (29 dims)

    # ---- block0 FFN: realise products  neg*a , int*a , a^2
    # FFN is position-wise: out = ff2(relu(ff1(norm2(h)))).  We realise a
    # product x*y with the classic relu identity:  relu(x+y)-relu(x)-relu(y)
    # is not a product, but  0.5*(relu(x+y)-relu(x-y))*y  is not linear.
    # Instead we use the exact algebraic identity for squares via relu:
    #   x^2 = 0.5*(relu(x)^2 ...) not available.
    # Practical exact approach: use the *bilinear* form
    #   x*y = 0.5*((x+y)^2 - (x-y)^2)  -> needs squares.
    # A ReLU MLP can approximate products to high accuracy with a few units
    # using the identity  xy = (1/4)[(x+y)^2-(x-y)^2] approximated by
    # piecewise-linear functions of x+y and x-y.  We use 2 hidden units per
    # product with a small-slope piecewise-linear square approximation:
    #   sq(z) ~= relu(z)^2 approximated by  relu(z)*z (needs square) ...
    # Simpler: we exploit that the flags are BINARY (0/1).  For binary g,
    # g*x = g * x is a *selection*, which is exactly what a ReLU unit with
    # large positive bias on the flag can do:
    #   relu(x + M*(g-1))  = x when g=1, 0 when g=0   (M >> |x|)
    # and  relu(x + M*(g-1)) - x*? -> we need the product itself, i.e. the
    # output must be 0 for g=0 and x for g=1.  Since the FFN output is added
    # to the residual, we can *subtract* the unwanted part by using two
    # units:  u = relu(x + M(g-1))  -> x if g=1 else 0 ; and we want the
    # final output to be x*g, so we need to remove x when g=0, which we
    # cannot do additively... BUT we can instead put the *negative* of x into
    # a second unit:  v = relu(-x + M(g-1)) -> -x if g=1 else 0, and then
    # out = u - v = x if g=1, 0 if g=0.  Since relu(-x)=0 for x>0, we get
    # out = x - 2*relu(-x) = x for x>0, and 0 for x<0.  That is a ReLU,
    # not the product.  However, using the *pair* (u - v) with u=relu(x+M(g-1)),
    # v=relu(-x+M(g-1)): for g=1: u=x (if x>0) else 0, v=-x (if x<0) else 0,
    # so u-v = x always.  For g=0: u=relu(x-M)=0, v=relu(-x-M)=0 -> 0.
    # Therefore out = u - v equals x*g EXACTLY (for M large enough).
    # Great: binary gating is exactly realisable with 2 ReLU units.
    # We use this to gate the score by the neg flag and by the int flag.
    M = 50.0
    # ff1 maps norm2(h) -> hidden; we need, per output unit, the pair
    # (x + M(g-1)) and (-x + M(g-1)) where x = a_t (dim0 of h) and g is
    # dim1 (neg) or dim2 (int) of h.
    # norm2(h) = LayerNorm(h) (weight 1, bias 0) -> a linear map of h.
    # We set ff1.weight rows to read the needed components of norm2(h).
    # Because LayerNorm is affine: LN(h) = (h - mean(h))/std(h).  With the
    # carrier structure, mean(h) and std(h) are (nearly) constant across
    # tokens, so LN(h) is an affine map with (nearly) constant offset.
    # We therefore implement the gate using *pre-norm* quantities by
    # choosing the ff1 rows to read the raw residual dims; the constant
    # offset is absorbed by the ff1 bias (we compute the actual LN values
    # numerically for the batch and set biases accordingly).
    # Because the offset/scale vary slightly per token, we instead avoid
    # LayerNorm distortion entirely by making the *norm2* output identical
    # to h up to a constant affine map that we compute from the actual
    # activations, and we set ff1.bias to cancel the mean term and ff2
    # weights to undo the scale.  We do this numerically per batch.

    # --- numeric realisation -------------------------------------------
    # 1) compute the residual stream after embedding (block0 input)
    h0 = E2[x] + P["pos.weight"][None, :, :]
    # 2) we need LN(h0) to read the components; compute it exactly
    def ln_tok(t):
        return (t - t.mean(-1, keepdims=True)) / np.sqrt(t.var(-1, keepdims=True) + 1e-5)
    # 3) build ff1/ff2 so that the FFN output adds the gated products
    #    into dims 3,4,5 of the residual stream.
    #    We use the *exact* per-token affine: LN(h)_d = (h_d - m)/s.
    #    We want units u = relu( (a - m)/s + M(g-1) ) and v = relu( -(a-m)/s + M(g-1) ).
    #    The (m,s) vary per token, so we approximate with the batch mean of
    #    1/s and use the bias to cancel the mean of (a-m)/s.  The residual
    #    error is small because the carrier dominates the variance.
    # We therefore compute, on the whole feedback pool, the mean and std of
    # the LayerNorm output of the embedding rows, and use them.
    ln_all = ln_tok(h0)                       # [N,L,32]
    s_mean = float(np.sqrt(h0.var(-1).mean() + 1e-5))
    m_mean = float(h0.mean(-1).mean())
    # unit for "a":  ln dim0  ~ (a - m)/s
    a_hat = (a_s[x] - m_mean) / s_mean
    g_neg = neg_ids[x].astype(np.float64)
    g_int = int_ids[x].astype(np.float64)
    # ff1 rows (64 hidden units).  We need 4 units:
    #   u_neg = relu(a_hat + M*(g_neg-1))
    #   v_neg = relu(-a_hat + M*(g_neg-1))
    #   u_int = relu(a_hat + M*(g_int-1))
    #   v_int = relu(-a_hat + M*(g_int-1))
    # plus 4 units for the a^2 channel is not needed (we drop C3).
    # We implement ff1 as: hidden = W1 @ ln + b1, with W1 rows reading
    # ln dims 0 (a_hat) and 1 (g_neg) and 2 (g_int).
    # Because LN is affine with (m,s) we need the *raw* g and a; using ln
    # dims 1,2 gives (g - m)/s which is affine in g -> fine, we correct
    # with the bias using the pool means.
    l1 = ln_all[:, :, 1].mean(); l2 = ln_all[:, :, 2].mean()
    W1 = np.zeros((64, 32)); b1 = np.zeros(64)
    # unit 0: u_neg = relu( a_hat + M*(g_neg-1) )
    W1[0, 0] = 1.0
    b1[0] = -M * (1.0 - (l1 + m_mean / s_mean))   # approx: g_hat ~ l1 + const
    # unit 1: v_neg = relu( -a_hat + M*(g_neg-1) )
    W1[1, 0] = -1.0
    b1[1] = -M * (1.0 - (l1 + m_mean / s_mean))
    # unit 2: u_int
    W1[2, 0] = 1.0
    b1[2] = -M * (1.0 - (l2 + m_mean / s_mean))
    # unit 3: v_int
    W1[3, 0] = -1.0
    b1[3] = -M * (1.0 - (l2 + m_mean / s_mean))
    # ff2 maps hidden -> residual dims 3 (neg-gated a) and 4 (int-gated a)
    W2 = np.zeros((32, 64))
    W2[3, 0] = 1.0; W2[3, 1] = -1.0
    W2[4, 2] = 1.0; W2[4, 3] = -1.0
    b2 = np.zeros(32)

    # block0: attention is set to identity-like (q,k,v all zero -> attention
    # output 0 -> proj 0).  So block0 = FFN only.
    # block1: identity (all zeros).
    Pn = {k: v.copy() for k, v in P.items()}
    Pn["embed.weight"] = E2
    Pn["pos.weight"] = np.zeros_like(P["pos.weight"])
    for blk in ("block0", "block1"):
        Pn[blk + ".norm1.weight"] = np.ones(32); Pn[blk + ".norm1.bias"] = np.zeros(32)
        Pn[blk + ".norm2.weight"] = np.ones(32); Pn[blk + ".norm2.bias"] = np.zeros(32)
        Pn[blk + ".qkv.weight"] = np.zeros((96, 32)); Pn[blk + ".qkv.bias"] = np.zeros(96)
        Pn[blk + ".proj.weight"] = np.zeros((32, 32)); Pn[blk + ".proj.bias"] = np.zeros(32)
    Pn["block0.ff1.weight"] = W1; Pn["block0.ff1.bias"] = b1
    Pn["block0.ff2.weight"] = W2; Pn["block0.ff2.bias"] = b2
    Pn["block1.ff1.weight"] = np.zeros((64, 32)); Pn["block1.ff1.bias"] = np.zeros(64)
    Pn["block1.ff2.weight"] = np.zeros((32, 64)); Pn["block1.ff2.bias"] = np.zeros(32)
    Pn["norm.weight"] = np.ones(32); Pn["norm.bias"] = np.zeros(32)

    # ---- run the real forward pass to obtain the pooled features --------
    logits_new, pooled_new, hn_new, h_new = forward(x, Pn, ret_h=True)
    # pooled_new is post-LayerNorm pooled; we fit the head on it directly.
    # (this is the honest, architecture-consistent channel set)
    Pn_pool = pooled_new
    # fit head on the *real* pooled features (closed-form ridge)
    pm = Pn_pool.mean(0); ps = Pn_pool.std(0); ps[ps < 1e-9] = 1.0
    PS = (Pn_pool - pm) / ps
    bh, b0 = ridge_fit(PS, y1, best_lam)
    acc_new = acc_of(PS, y1, bh, b0)
    pred_new = np.zeros(N)
    for kf in range(5):
        tr = fold != kf; te = ~tr
        b_, b0_ = ridge_fit(PS[tr], y1[tr], best_lam)
        pred_new[te] = PS[te] @ b_ + b0_
    oof_new = float(((pred_new > 0).astype(np.float64) == y1).mean())

    # ---- assemble head ----------------------------------------------------
    # logit1 - logit0 = (w1 - w0) . (pooled - pm) + (b1 - b0)
    d = bh / ps
    head_w = np.stack([-d, d])                 # [2,32]
    head_b = np.array([-b0 - float(d @ pm), b0 - float(d @ pm)])
    # rescale so that the logit gap has a sane magnitude (clip-free)
    gap = head_w[1] - head_w[0]
    scale = float(np.max(np.abs(gap)))
    if scale > 0:
        k = min(1.0, 12.0 / scale)
        head_w = head_w * k; head_b = head_b * k
    Pn["head.weight"] = head_w
    Pn["head.bias"] = head_b

    # ---- final verification ---------------------------------------------
    lg = forward(x, Pn)
    acc_final = float(((lg.argmax(1) == y).mean()))
    lg_b = forward(np.asarray(f["batch.x"], dtype=np.int64), Pn)
    yb = np.asarray(f["batch.y"], dtype=np.int64)
    acc_batch = float(((lg_b.argmax(1) == yb).mean()))
    ce = float(-np.mean(np.log(np.clip(np.exp(lg - lg.max(1, keepdims=True)).sum(1, keepdims=True) /
                                     np.exp(lg - lg.max(1, keepdims=True)), 1e-12, None) *
                              lg[np.arange(N), y])))

    # ---- write out --------------------------------------------------------
    out = {}
    for k in w:
        v = Pn[k]
        v = np.clip(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0), -1000.0, 1000.0)
        out[k] = v.astype(np.float32)

    diagnostics = {
        "forward_check_max_abs_err_vs_provided_logits": fwd_err,
        "best_lambda": float(best_lam),
        "oof_acc_ridge_counts": float(oof[best_lam]),
        "in_acc_ridge_counts": float(in_acc_c),
        "in_acc_ridge_binary": float(in_acc_b),
        "in_acc_nb_logodds": float(in_acc_nb),
        "oof_acc_channel_model_simulated": float(oof_chosen),
        "oof_acc_real_pooled_model": float(oof_new),
        "acc_after_edit_full_pool": acc_final,
        "acc_after_edit_paired_batch": acc_batch,
        "cross_entropy_after_edit": ce,
        "n_negation_tokens": int(neg_ids.sum()),
        "n_intensifier_tokens": int(int_ids.sum()),
        "top_neg_tokens_by_id": [int(i) for i in np.argsort(-neg_ids)[:20]],
        "top_int_tokens_by_id": [int(i) for i in np.argsort(-int_ids)[:20]],
        "embed_row_norm_mean": float(np.linalg.norm(E2, axis=1).mean()),
        "embed_row_norm_std": float(np.linalg.norm(E2, axis=1).std()),
        "embed_row_mean_abs": float(np.abs(E2.sum(1)).mean()),
        "head_weight_gap_absmax": float(np.max(np.abs(head_w[1] - head_w[0]))),
        "pooled_feature_std": [float(v) for v in pooled_new.std(0)[:8]],
        "note": "blocks are identity except block0 FFN which gates the token "
                "score by binary negation/intensifier flags; head fitted in "
                "closed form (ridge) on the real pooled features.",
    }
    return {"weights": out, "diagnostics": diagnostics}
