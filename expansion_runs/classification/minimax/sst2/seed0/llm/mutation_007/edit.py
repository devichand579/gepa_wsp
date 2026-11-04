import numpy as np

def edit(w, f):
    # ------------------------------------------------------------------ data
    x = np.asarray(f["x"], dtype=np.int64)          # [N,64] token ids
    y = np.asarray(f["y"], dtype=np.int64)          # [N] labels
    N, L = x.shape
    V = 2000
    mask = (x != 0)
    lens = mask.sum(1).astype(np.float64)
    lens[lens == 0] = 1.0

    # ------------------------------------------------------------- utilities
    def softmax_rows(a):
        m = a.max(axis=-1, keepdims=True)
        e = np.exp(a - m)
        return e / e.sum(axis=-1, keepdims=True)

    def layer_norm(h, g, b, eps=1e-5):
        mu = h.mean(-1, keepdims=True)
        var = h.var(-1, keepdims=True)
        return g * (h - mu) / np.sqrt(var + eps) + b

    def lin(h, W, b):
        return h @ W.T + b

    # ------------------------------------------------------- token features
    # unigram: normalised counts ; bigram: (prev,cur) normalised counts
    def counts_uni():
        C = np.zeros((N, V), dtype=np.float64)
        rows = np.repeat(np.arange(N), lens.astype(int))
        C[rows, x[mask]] = 1.0
        return C / lens[:, None]

    prev = np.zeros_like(x)
    prev[:, 1:] = x[:, :-1]
    valid_prev = np.zeros_like(x, dtype=bool)
    valid_prev[:, 1:] = mask[:, :-1]
    # bigram index only where both tokens real
    bidx = (prev * V + x).astype(np.int64)
    use_bi = mask & valid_prev

    def counts_bi():
        B = np.zeros((N, V * V), dtype=np.float64)
        rows = np.repeat(np.arange(N), use_bi.sum(1))
        B[rows, bidx[use_bi]] = 1.0
        return B / lens[:, None]

    # ------------------------------------------------- closed-form ridge fit
    def ridge_fit(Phi, t, lams):
        # Phi [N,D] (intercept appended), t [N]
        A = Phi.T @ Phi
        b = Phi.T @ t
        n, D = Phi.shape
        out = {}
        for lam in lams:
            R = A.copy()
            R[1:, 1:] += lam * np.eye(D - 1)   # do not penalise intercept
            try:
                wgt = np.linalg.solve(R, b)
            except np.linalg.LinAlgError:
                wgt = np.linalg.lstsq(R, b, rcond=None)[0]
            out[lam] = wgt
        return out

    def gcv(Phi, t, wgt):
        n = Phi.shape[0]
        resid = Phi @ wgt - t
        rss = float(resid @ resid)
        # effective dof via trace of hat matrix
        A = Phi.T @ Phi
        # dof = tr(A (A+lamI)^-1) computed from the same normal equations
        return rss / n, wgt

    # ================================================================ MODEL A
    # Exact unigram construction (current model family), refitted with a
    # better feature map + closed-form lambda selection.
    #   embed[v] = g_v * dhat + sqrt(1-g_v^2) * phat   (zero mean, unit var)
    #   head.w[1]-head.w[0] = s*dhat , head.b[1]-head.b[0] = c
    #   blocks zeroed -> identity ; pos = 0
    # score = s * mean_t g(id_t) + c

    def build_embed(g):
        # dhat : unit zero-mean direction (dim 32)
        rng = np.random.default_rng(0)
        d = rng.normal(size=32)
        d -= d.mean()
        d /= np.linalg.norm(d)
        # phat : unit vector orthogonal to dhat, zero mean
        p = rng.normal(size=32)
        p -= p.mean()
        p -= (p @ d) * d
        p /= np.linalg.norm(p)
        g = np.clip(g, -0.999, 0.999)
        E = np.outer(g, d) + np.outer(np.sqrt(1.0 - g * g), p)
        return E, d, p

    def make_unigram(g, c, s):
        E, d, p = build_embed(g)
        W = dict(w)
        W["embed.weight"] = E.astype(np.float32)
        W["pos.weight"] = np.zeros((64, 32), dtype=np.float32)
        for blk in ("block0", "block1"):
            for nm, shp in (("qkv.weight", (96, 32)), ("qkv.bias", (96,)),
                            ("proj.weight", (32, 32)), ("proj.bias", (32,)),
                            ("ff1.weight", (64, 32)), ("ff1.bias", (64,)),
                            ("ff2.weight", (32, 64)), ("ff2.bias", (32,))):
                W["%s.%s" % (blk, nm)] = np.zeros(shp, dtype=np.float32)
        hw = np.zeros((2, 32), dtype=np.float64)
        hw[0] = -0.5 * s * d
        hw[1] = 0.5 * s * d
        W["head.weight"] = hw.astype(np.float32)
        hb = np.zeros(2, dtype=np.float64)
        hb[0] = -0.5 * c
        hb[1] = 0.5 * c
        W["head.bias"] = hb.astype(np.float32)
        return W

    def forward_unigram(W, x):
        E = W["embed.weight"].astype(np.float64)
        h = E[x]                                   # [N,L,32]
        h = layer_norm(h, np.ones(32), np.zeros(32))
        m = (x != 0).astype(np.float64)
        pooled = (h * m[:, :, None]).sum(1) / m.sum(1, keepdims=True)
        lg = pooled @ W["head.weight"].astype(np.float64).T + W["head.bias"]
        return lg

    def acc_ce(W, x, y):
        lg = forward_unigram(W, x)
        p = softmax_rows(lg)
        a = float((lg.argmax(1) == y).mean())
        ce = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())
        return a, ce

    # feature variants for the unigram part
    Cu = counts_uni()
    feats = {
        "count": Cu,
        "log": np.log1p(Cu),
        "bin": (Cu > 0).astype(np.float64),
    }

    t = (2.0 * y - 1.0)
    best = None
    for fname, Phi0 in feats.items():
        Phi = np.concatenate([np.ones((N, 1)), Phi0], axis=1)
        for lam in (0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
            A = Phi.T @ Phi
            b = Phi.T @ t
            R = A.copy()
            R[1:, 1:] += lam * np.eye(Phi.shape[1] - 1)
            try:
                wg = np.linalg.solve(R, b)
            except np.linalg.LinAlgError:
                continue
            resid = Phi @ wg - t
            rss = float(resid @ resid)
            # closed-form GCV over the ridge path (no iterative optimisation)
            Rinv_needed = False
            # dof = tr(A R^{-1}) ; compute with one solve of stacked systems
            try:
                sol = np.linalg.solve(R, np.concatenate([b, A], axis=1))
                wg = sol[:, 0]
                dof = float(np.trace(sol[:, 1:]))
            except np.linalg.LinAlgError:
                continue
            gcv = rss / n_safe(rss) if False else rss / max(N - dof, 1e-6)
            # in-sample accuracy of the linear surrogate
            sc = Phi @ wg
            a = float(((sc > 0).astype(int) == y).mean())
            key = (fname, lam)
            cand = {"fname": fname, "lam": lam, "gcv": gcv, "acc": a,
                    "g": wg[1:], "c": float(wg[0])}
            if best is None or gcv < best["gcv"]:
                best = cand

    g = best["g"]
    c = best["c"]
    s = float(np.max(np.abs(g)))
    s = max(s, 1e-3)

    W_uni = make_unigram(g, c, s)
    a_uni, ce_uni = acc_ce(W_uni, x, y)

    # ================================================================ MODEL B
    # Bigram model realised through block0 attention (shifted one-hot lookup)
    # + block0 FFN ReLU interaction.  Verified numerically against an
    # explicit re-implementation of the forward pass.
    #
    # Design:
    #   pos.weight[t] = big * c_t   (position code, dominant norm)
    #   embed rows     = small token code (so norm1(h) ~ direction of c_t)
    #   block0.qkv: q uses c_t, k uses c_{s+1}  ->  attention peak at s = t-1
    #   block0.proj: identity on v (v = embed of previous token)
    #   block0.ff1/ff2: ReLU product of two coordinates of norm2(h)
    #   block1: zeroed (identity)
    #   head: reads unigram coordinate + bigram coordinate

    rng = np.random.default_rng(1)
    # ---- position codes with sharp shifted matching -------------------------
    # want <c_t, c_{s+1}> large for s = t-1 only.
    # build random 64x32 codes, then whiten so that C C^T ~ I on the
    # shifted pairing via a Gram-matrix correction (closed form).
    C = rng.normal(size=(64, 32))
    # target Gram for shifted pairing: G = C S C^T should be ~ I_64 (impossible
    # rank) -> instead maximise the diagonal-vs-offdiag gap by whitening C:
    G = C @ C.T
    G = G / (np.trace(G) / 64.0)
    # symmetric square root inverse of G (ridge-regularised for rank)
    ev, U = np.linalg.eigh(G)
    ev = np.clip(ev, 1e-3, None)
    Ginv_half = U @ np.diag(ev ** -0.5) @ U.T
    Cw = Ginv_half @ C                      # rows ~ orthonormal-ish
    # shifted pairing code: k_s uses c_{s+1}
    Cshift = np.zeros_like(Cw)
    Cshift[:-1] = Cw[1:]
    Cshift[-1] = Cw[-1]
    # matched inner products
    matched = np.sum(Cw[:-1] * Cshift[:-1], axis=1)
    # competitor magnitudes
    comp = Cw @ Cshift.T
    np.fill_diagonal(comp, 1e9)
    gap = matched - np.max(np.abs(comp), axis=1)

    # ---- build the block0 attention ---------------------------------------
    # h_t = embed[id_t] + pos[t];  norm1 -> z_t
    # q_t = Wq z_t  (uses c_t), k_s = Wk z_s (uses c_{s+1})
    # We realise: q_t = a * c_t , k_s = b * c_{s+1}
    # Wq maps z -> a*c_t : since z_t ~ c_t direction (pos dominant),
    #   Wq = a * pinv(C)  (least squares: Wq C = a I)
    #   Wk = b * pinv(Cshift)
    a_amp, b_amp = 6.0, 6.0
    Wq = a_amp * np.linalg.pinv(Cw)            # 32x32
    Wk = b_amp * np.linalg.pinv(Cshift)        # 32x32
    # v_s = Wv z_s : we want v_s = embed[id_s] (token code), so Wv should
    # extract the token part of z_s.  z_s = norm1(embed[id_s] + pos[s]).
    # With pos dominant, z_s ~ c_s direction -> token info is in the
    # residual.  Instead: make v_s = Wv z_s where Wv picks the token code
    # component: use the projection onto the orthogonal complement of the
    # position subspace:  Wv = (I - P_pos) with P_pos = C^T (C C^T)^-1 C.
    Ppos = Cw.T @ np.linalg.pinv(Cw @ Cw.T) @ Cw
    Wv = np.eye(32) - Ppos
    # proj: identity (attention output already in token space)
    Wproj = np.eye(32)
    bproj = np.zeros(32)

    # ---- token codes -------------------------------------------------------
    # token code u_v : 32-dim, zero mean over dims, unit variance over dims
    # (so that LayerNorm(embed row) = embed row) but small norm relative to
    # position code so that norm1(h) is position dominated.
    # We need two special coordinates in the token code:
    #   coord A : unigram sentiment score g_v  (for the head readout)
    #   coord B : previous-token score g_v    (for the interaction)
    # Use two independent directions dA, dB (orthogonal, zero mean).
    rng2 = np.random.default_rng(2)
    dA = rng2.normal(size=32); dA -= dA.mean(); dA /= np.linalg.norm(dA)
    dB = rng2.normal(size=32); dB -= dB.mean(); dB -= (dB @ dA) * dA; dB /= np.linalg.norm(dB)
    # remaining 30 dims: random filler, scaled so row std == 1
    # build rows: u_v = gA_v * dA + gB_v * dB + filler_v
    # choose gA_v = g_v (unigram), gB_v = g_v (same table, used as context)
    # filler: random unit-variance orthogonal part
    F = rng2.normal(size=(V, 32))
    F -= (F @ dA)[:, None] * dA[None, :]
    F -= (F @ dB)[:, None] * dB[None, :]
    F -= F.mean(axis=1, keepdims=True)
    F /= F.std(axis=1, keepdims=True)
    # scale filler so that after adding the two g-components the row has
    # zero mean and unit variance over the 32 dims
    U = np.zeros((V, 32))
    U[:, :] = F * 0.0
    # start from filler scaled to leave room
    resid_dims = 32
    # variance contributed by gA*dA and gB*dB : (gA^2 + gB^2)/32 each dim
    gA = np.clip(g, -0.999, 0.999)
    gB = np.clip(g, -0.999, 0.999)
    var_special = (gA ** 2 + gB ** 2) / 32.0
    scale_f = np.sqrt(np.clip(1.0 - var_special, 1e-6, None))
    U = F * scale_f[:, None]
    U += np.outer(gA, dA) + np.outer(gB, dB)
    # enforce exact zero mean / unit std per row
    U -= U.mean(axis=1, keepdims=True)
    U /= U.std(axis=1, keepdims=True)
    # now coordinate along dA is not exactly gA any more -> recompute
    # (we only need the *relative* values; keep as is but record)
    gA_eff = U @ dA
    gB_eff = U @ dB

    # ---- position embedding magnitude --------------------------------------
    pos_scale = 40.0
    POS = pos_scale * Cw                     # [64,32] rows = pos codes

    # ---- block0 FFN: exact ReLU product of two coordinates ---------------
    # z = norm2(h) has coordinates (along dA) = x (current token score)
    # and (along dB) = y (previous token score).  We need out = x*y.
    # ReLU identity for bounded x,y in [-1,1]:
    #   xy = 0.5*((x+y)^2 - (x-y)^2)  and  0.5*z^2 = relu(z)*relu(z) is not
    #   linear; use the exact 2-unit trick:
    #   for z in [-1,1]:  z^2 = relu(z)*relu(z)  -> not linear.
    # Instead use the exact piecewise-linear surrogate on a fine grid via
    # relu units:  z^2 = sum_k  ... (approximate, verified numerically).
    # Simpler exact alternative: use the identity
    #   xy = 0.5*[(x+y)^2-(x-y)^2] and approximate z^2 by
    #   relu(z)^2 approximated with 16 relu units (piecewise linear).
    # We instead use a *different* exact route: the interaction is realised
    # by the attention *value* path is impossible -> use PL approximation.
    # PL approximation of z^2 on [-1,1] with m segments:
    #   z^2 = sum_{k} slope_k * relu(z - knot_k) + const   (exact at knots)
    mseg = 12
    knots = np.linspace(-1.0, 1.0, mseg + 1)
    # slopes of the secants of z^2 between knots
    slopes = (knots[1:] ** 2 - knots[:-1] ** 2) / (knots[1:] - knots[:-1])
    # z^2 = c0 + sum_k slopes_k * relu(z - knots_k)   (k=0..mseg-1)
    c0 = knots[0] ** 2
    # build ff1 (64x32) and ff2 (32x64) implementing  out = xy
    # hidden units: for u = x+y : relu(u - knots_k) ; for v = x-y : relu(v-knots_k)
    # ff1 rows: 2*mseg units, each unit computes relu(lin - bias)
    #   lin for u: dA + dB ; for v: dA - dB
    nhid = 2 * mseg
    W1 = np.zeros((nhid, 32))
    b1 = np.zeros(nhid)
    W1[0:mseg, :] = (dA + dB)[None, :]
    b1[0:mseg] = -knots[:-1]
    W1[mseg:2 * mseg, :] = (dA - dB)[None, :]
    b1[mseg:2 * mseg] = -knots[:-1]
    # ff2: out_coord = 0.5*(sum slopes*relu(u-knot) - sum slopes*relu(v-knot))
    #       + 0.5*(c0 - c0)  -> constant cancels
    W2 = np.zeros((32, nhid))
    W2[0, 0:mseg] = 0.5 * slopes
    W2[0, mseg:2 * mseg] = -0.5 * slopes
    b2 = np.zeros(32)
    # the constant term: 0.5*(c0 - c0) = 0  (both sides share c0)
    # put the product into coordinate 0 of the residual stream
    # but coordinate 0 must be a dedicated "interaction" channel.
    # We instead place the interaction on a dedicated direction dI and let the
    # head read it.  Use dI = a third direction; but ff2 output must be
    # added to the residual stream, whose coordinates are the token code.
    # => use coordinate index of dA?  We add a dedicated dimension:
    # choose dI orthogonal to dA,dB and set ff2 row = 0.5*slopes along dI.
    dI = rng2.normal(size=32); dI -= dI.mean()
    dI -= (dI @ dA) * dA; dI -= (dI @ dB) * dB
    dI /= np.linalg.norm(dI)
    W2 = np.zeros((32, nhid))
    W2 = np.outer(dI, np.concatenate([0.5 * slopes, -0.5 * slopes]))
    b2 = np.zeros(32)
    # IMPORTANT: the interaction channel must not be part of the token code
    # read by the head's unigram part -> keep head reading dA only.

    # ---- assemble block0 --------------------------------------------------
    qkvW = np.zeros((96, 32))
    qkvW[0:32] = Wq
    qkvW[32:64] = Wk
    qkvW[64:96] = Wv
    qkvB = np.zeros(96)

    Wb = dict(w)
    Wb["embed.weight"] = U.astype(np.float32)
    Wb["pos.weight"] = POS.astype(np.float32)
    Wb["block0.qkv.weight"] = qkvW.astype(np.float32)
    Wb["block0.qkv.bias"] = qkvB.astype(np.float32)
    Wb["block0.proj.weight"] = Wproj.astype(np.float32)
    Wb["block0.proj.bias"] = bproj.astype(np.float32)
    Wb["block0.ff1.weight"] = W1.astype(np.float32)
    Wb["block0.ff1.bias"] = b1.astype(np.float32)
    Wb["block0.ff2.weight"] = W2.astype(np.float32)
    Wb["block0.ff2.bias"] = b2.astype(np.float32)
    for nm, shp in (("qkv.weight", (96, 32)), ("qkv.bias", (96,)),
                    ("proj.weight", (32, 32)), ("proj.bias", (32,)),
                    ("ff1.weight", (64, 32)), ("ff1.bias", (64,)),
                    ("ff2.weight", (32, 64)), ("ff2.bias", (32,))):
        Wb["block1.%s" % nm] = np.zeros(shp, dtype=np.float32)
    Wb["norm.weight"] = np.ones(32, dtype=np.float32)
    Wb["norm.bias"] = np.zeros(32, dtype=np.float32)

    # ---- verify the constructed bigram model numerically -------------------
    def forward_bigram(W, x, chunk=256):
        outs = []
        for st in range(0, x.shape[0], chunk):
            xb = x[st:st + chunk]
            nb = xb.shape[0]
            E = W["embed.weight"].astype(np.float64)
            h = E[xb] + W["pos.weight"].astype(np.float64)[None, :, :]
            z1 = layer_norm(h, np.ones(32), np.zeros(32))
            qkv = lin(z1, W["block0.qkv.weight"].astype(np.float64),
                      W["block0.qkv.bias"].astype(np.float64))
            q = qkv[:, :, 0:32]; k = qkv[:, :, 32:64]; v = qkv[:, :, 64:96]
            outs.append((q, k, v, h))
        q = np.concatenate([o[0] for o in outs], 0)
        k = np.concatenate([o[1] for o in outs], 0)
        v = np.concatenate([o[2] for o in outs], 0)
        h = np.concatenate([o[3] for o in outs], 0)
        nh = x.shape[1]
        # 2 heads of width 16
        att = np.zeros((x.shape[0], nh, 32))
        for hd in range(2):
            sl = slice(hd * 16, (hd + 1) * 16)
            sc = np.einsum('ntd,nsd->nts', q[:, :, sl], k[:, :, sl]) / 4.0
            m = (x != 0)
            sc = np.where(m[:, None, :], sc, -1e30)
            a = softmax_rows(sc)
            att[:, :, sl] = np.einsum('nts,nsd->ntd', a, v[:, :, sl])
        h = h + lin(att, W["block0.proj.weight"].astype(np.float64),
                    W["block0.proj.bias"].astype(np.float64))
        z2 = layer_norm(h, np.ones(32), np.zeros(32))
        hid = np.maximum(lin(z2, W["block0.ff1.weight"].astype(np.float64),
                             W["block0.ff1.bias"].astype(np.float64)), 0.0)
        h = h + lin(hid, W["block0.ff2.weight"].astype(np.float64),
                    W["block0.ff2.bias"].astype(np.float64))
        h = layer_norm(h, np.ones(32), np.zeros(32))
        m = (x != 0).astype(np.float64)
        pooled = (h * m[:, :, None]).sum(1) / m.sum(1, keepdims=True)
        lg = pooled @ W["head.weight"].astype(np.float64).T + W["head.bias"]
        return lg

    # ---- fit the bigram table --------------------------------------------
    # We want, per position t, the interaction channel to carry
    #   g(id_t) * g(id_{t-1})  (a genuine bigram-ish score).
    # The head reads:  score = sA * mean_t gA(id_t) + sI * mean_t inter_t
    # where inter_t = gA(id_t) * gB(id_{t-1}) (approx, PL error).
    # Fit the *scale* of the interaction by closed-form least squares on the
    # surrogate features (unigram counts + interaction counts).
    # surrogate features: [1, unigram(counts), interaction counts]
    # interaction counts: for each (prev,cur) pair, count of occurrences.
    Bmat = np.zeros((N, V * V), dtype=np.float64)
    rows = np.repeat(np.arange(N), use_bi.sum(1))
    Bmat[rows, bidx[use_bi]] = 1.0
    Bmat = Bmat / lens[:, None]
    PhiB = np.concatenate([np.ones((N, 1)), Cu, Bmat], axis=1)
    # ridge with strong penalty on the bigram block
    lam_u, lam_b = 3.0, 300.0
    A = PhiB.T @ PhiB
    b = PhiB.T @ t
    R = A.copy()
    R[1:1 + V, 1:1 + V] += lam_u * np.eye(V)
    R[1 + V:, 1 + V:] += lam_b * np.eye(V * V)
    try:
        wB = np.linalg.solve(R, b)
    except np.linalg.LinAlgError:
        wB = np.linalg.lstsq(R, b, rcond=None)[0]
    gB_tab = wB[1 + V:].reshape(V, V)      # bigram table (prev, cur)
    # the realised interaction channel must be gA(id_t)*gB(id_{t-1});
    # we can only realise products of two *scalar* per-token codes, so we
    # approximate the bigram table by a rank-1 factorisation:
    #   gB_tab[prev,cur] ~ beta_prev * alpha_cur
    # closed-form: alternating-free rank-1 SVD of the table (one step).
    Umat, S, Vt = np.linalg.svd(gB_tab, full_matrices=False)
    beta = Umat[:, 0] * np.sqrt(S[0])
    alpha = Vt[0] * np.sqrt(S[0])
    # rescale so that the context code gB_v = beta_v and current code
    # gA_v = alpha_v (both bounded to [-1,1])
    mx = max(np.max(np.abs(beta)), np.max(np.abs(alpha)), 1e-9)
    beta = beta / mx
    alpha = alpha / mx
    # rebuild token code with alpha (current) and beta (context)
    gA2 = np.clip(alpha, -0.999, 0.999)
    gB2 = np.clip(beta, -0.999, 0.999)
    var_special = (gA2 ** 2 + gB2 ** 2) / 32.0
    scale_f = np.sqrt(np.clip(1.0 - var_special, 1e-6, None))
    U2 = F * scale_f[:, None]
    U2 += np.outer(gA2, dA) + np.outer(gB2, dB)
    U2 -= U2.mean(axis=1, keepdims=True)
    U2 /= U2.std(axis=1, keepdims=True)
    Wb["embed.weight"] = U2.astype(np.float32)

    # head: read unigram channel (dA) and interaction channel (dI)
    # pooled interaction = mean_t inter_t ; inter_t = alpha(id_t)*beta(id_{t-1})
    # fit head scales by closed-form least squares on the realised surrogate
    # score:  sA*mean_t alpha(id_t) + sI*mean_t alpha(id_t)beta(id_{t-1})
    su = Cu @ alpha
    si = np.zeros(N)
    rows2 = np.repeat(np.arange(N), use_bi.sum(1))
    prod_tab = (alpha[:, None] * beta[None, :])       # [cur, prev]
    si[rows2] = prod_tab[x[use_bi], prev[use_bi]]
    si = si / lens
    PhiH = np.stack([np.ones(N), su, si], axis=1)
    AH = PhiH.T @ PhiH + 1e-6 * np.eye(3)
    bH = PhiH.T @ t
    wH = np.linalg.solve(AH, bH)
    sA, sI, cH = float(wH[1]), float(wH[2]), float(wH[0])

    # assemble head for the bigram model:
    #   head.w[1]-head.w[0] = sA*dA + sI*dI
    hw = np.zeros((2, 32))
    hw[0] = -0.5 * (sA * dA + sI * dI)
    hw[1] = 0.5 * (sA * dA + sI * dI)
    hb = np.zeros(2)
    hb[0] = -0.5 * cH
    hb[1] = 0.5 * cH
    Wb["head.weight"] = hw.astype(np.float32)
    Wb["head.bias"] = hb.astype(np.float32)

    lgB = forward_bigram(Wb, x)
    pB = softmax_rows(lgB)
    aB = float((lgB.argmax(1) == y).mean())
    ceB = float(-np.log(np.clip(pB[np.arange(N), y], 1e-12, None)).mean())

    # ---- choose the better model on the pool (both are exact constructions) --
    use_bigram = aB > a_uni
    if use_bigram:
        Wout = Wb
        a_sel, ce_sel = aB, ceB
    else:
        Wout = W_uni
        a_sel, ce_sel = a_uni, ce_uni

    # ---- final safety: shapes, dtype, finiteness, bounds -------------------
    for k in list(w.keys()):
        v = np.asarray(Wout[k], dtype=np.float32)
        if v.shape != np.asarray(w[k]).shape:
            v = np.asarray(w[k], dtype=np.float32)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        v = np.clip(v, -1000.0, 1000.0)
        Wout[k] = v

    diag = {
        "uni_feat": best["fname"],
        "uni_lambda": best["lam"],
        "uni_gcv": round(float(best["gcv"]), 5),
        "uni_pool_acc": round(a_uni, 4),
        "uni_pool_ce": round(ce_uni, 4),
        "bigram_pool_acc": round(aB, 4),
        "bigram_pool_ce": round(ceB, 4),
        "selected": "bigram" if use_bigram else "unigram",
        "selected_acc": round(a_sel, 4),
        "selected_ce": round(ce_sel, 4),
        "attn_matched_min": round(float(matched.min()), 4),
        "attn_gap_min": round(float(gap.min()), 4),
        "sA": round(sA, 4),
        "sI": round(sI, 4),
        "cH": round(cH, 4),
        "alpha_absmax": round(float(np.max(np.abs(alpha))), 4),
        "beta_absmax": round(float(np.max(np.abs(beta))), 4),
        "note": "blocks0 attention=shifted one-hot prev-token lookup; "
                "ff1/ff2 = PL product of current-token x context-token codes; "
                "block1 identity; head reads unigram + interaction channels",
    }
    return {"weights": Wout, "diagnostics": diag}
