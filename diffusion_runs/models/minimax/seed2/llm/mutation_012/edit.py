import numpy as np

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _conv2d(x, W, b):
    """x:[N,C,H,W], W:[O,C,3,3] (cross-correlation), b:[O] -> [N,O,H,W]"""
    N, C, H, Wd = x.shape
    O = W.shape[0]
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    out = np.zeros((N, O, H, Wd), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            out += np.tensordot(xp[:, :, i:i + H, j:j + Wd], W[:, :, i, j], axes=([1], [1])).transpose(1, 0, 2, 3)
    return out + b.reshape(1, O, 1, 1)


def _relu(a):
    return np.maximum(a, 0.0)


def _forward(w, x, t):
    """Fixed architecture forward (float64 internally)."""
    N = x.shape[0]
    tb = (t.astype(np.float64) / 31.0).reshape(N, 1, 1, 1)
    inp = np.concatenate([x.astype(np.float64), np.broadcast_to(tb, (N, 1, x.shape[2], x.shape[3]))], axis=1)
    h1 = _relu(_conv2d(inp, w["conv1.weight"], w["conv1.bias"]))
    h2 = _relu(_conv2d(h1, w["conv2.weight"], w["conv2.bias"]))
    h3 = _relu(_conv2d(h2, w["conv3.weight"], w["conv3.bias"]))
    o = _conv2d(h3, w["out.weight"], w["out.bias"])
    return o, h3


def _patch_features(F):
    """F:[N,C,H,W] -> [N, C*9, H*W] (3x3 patches, zero padded)."""
    N, C, H, Wd = F.shape
    Fp = np.pad(F, ((0, 0), (0, 0), (1, 1), (1, 1)))
    P = np.empty((N, C * 9, H * Wd), dtype=np.float64)
    k = 0
    for i in range(3):
        for j in range(3):
            P[:, k * C:(k + 1) * C, :] = Fp[:, :, i:i + H, j:j + Wd].reshape(N, C, H * Wd)
            k += 1
    return P


def _gram(P, chunk=64):
    """Sum_n P_n P_n^T  -> [D,D]  (D = C*9)."""
    N, D, M = P.shape
    G = np.zeros((D, D), dtype=np.float64)
    for s in range(0, N, chunk):
        Pc = P[s:s + chunk]
        G += np.einsum('ndm,nem->de', Pc, Pc, optimize=True)
    return G


def _rhs(P, Y, chunk=64):
    """Sum_n P_n Y_n^T -> [D, M]"""
    N, D, M = P.shape
    R = np.zeros((D, M), dtype=np.float64)
    for s in range(0, N, chunk):
        Pc = P[s:s + chunk]
        Yc = Y[s:s + chunk]
        R += np.einsum('ndm,nem->de', Pc, Yc, optimize=True)
    return R


def _solve(G, R, lam):
    d = G.shape[0]
    A = G + lam * np.eye(d)
    try:
        return np.linalg.solve(A, R)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, R, rcond=None)[0]


def _ridge_fit(P, Y, lam):
    G = _gram(P)
    R = _rhs(P, Y)
    return _solve(G, R, lam)


def _apply_W(Wmat, F):
    """Wmat:[D,C] with D=C*9 ordered (k,l,c); F:[N,C,H,W] -> [N,1,H,W]"""
    N, C, H, Wd = F.shape
    out = np.zeros((N, 1, H, Wd), dtype=np.float64)
    k = 0
    for i in range(3):
        for j in range(3):
            out[:, 0] += np.einsum('nc,cd->nd', F, Wmat[k * C:(k + 1) * C, :], optimize=True).reshape(N, H, Wd)
            k += 1
    return out


def _unfold_out(Wmat, b):
    """Convert [D,C] readout into conv3x3 out.weight[1,C,3,3] + out.bias[1]."""
    C = Wmat.shape[1]
    ow = np.zeros((1, C, 3, 3), dtype=np.float64)
    k = 0
    for i in range(3):
        for j in range(3):
            ow[0, :, i, j] = Wmat[k * C:(k + 1) * C, :].T
            k += 1
    return ow, np.array([b], dtype=np.float64)


def _clip_all(w):
    for k in list(w.keys()):
        a = np.asarray(w[k], dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(a, -1000.0, 1000.0, out=a)
        w[k] = a
    return w


# ---------------------------------------------------------------------------

def edit(w, f):
    w = {k: np.array(v, dtype=np.float64) for k, v in w.items()}
    diag = {}

    x = np.asarray(f["x"], dtype=np.float64)            # [N,1,16,16]
    y = np.asarray(f["y"], dtype=np.float64)            # [N,1,16,16] target eps
    tt = np.asarray(f["t"]).astype(np.int64)            # [N]
    clean = np.asarray(f["clean"], dtype=np.float64)     # [N,1,16,16]
    abar = np.asarray(f["alphabar"], dtype=np.float64)  # [32]
    N = x.shape[0]
    H = W_ = x.shape[2]

    abar_t = abar[np.clip(tt, 0, 31)]
    sa = np.sqrt(np.maximum(abar_t, 1e-12))
    sb = np.sqrt(np.maximum(1.0 - abar_t, 1e-12))

    # ---------------- 1. closed-form digit template from clean images -------
    C = clean[:, 0].reshape(N, -1)                      # [N,256]
    Cmean = C.mean(axis=0)
    Xc = C - Cmean
    # top-8 principal directions of the clean images (digit shape space)
    try:
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        K = 8
        V = Vt[:K].T                                       # [256,K]
    except np.linalg.LinAlgError:
        V = np.eye(256)[:, :8]
    # template bank: mean + principal directions
    Tbank = np.concatenate([Cmean[None, :], Cmean[None, :] + 2.0 * V[:, 0][None, :],
                             Cmean[None, :] - 2.0 * V[:, 0][None, :]], axis=0)   # [3,256]

    # ---------------- 2. analytic epsilon prior -----------------------------
    # x0_hat: match template bank to x_t (correlation), scaled to x_t energy
    Xf = x[:, 0].reshape(N, -1)
    # per-example: pick template maximizing correlation with x_t
    # (templates in [-1,1]-ish scale; x_t = sa*x0 + sb*eps)
    Xn = Xf - Xf.mean(axis=1, keepdims=True)
    Tn = Tbank - Tbank.mean(axis=1, keepdims=True)
    tn = (Tn ** 2).sum(axis=1)
    corr = (Xn @ Tn.T) / np.maximum(np.sqrt((Xn ** 2).sum(axis=1, keepdims=True) * tn[None, :]), 1e-9)
    best = np.argmax(corr, axis=1)
    tmpl = Tbank[best]                                    # [N,256]
    # least-squares scale of x0 given template direction and known noise level
    # x_t = sa*x0 + sb*eps ; assume x0 ~ alpha*tmpl + noise-free shape
    # alpha_hat = <x_t, tmpl> / (sa * <tmpl,tmpl>)
    denom = sa * (tmpl ** 2).sum(axis=1)
    alpha = (Xf * tmpl).sum(axis=1) / np.maximum(denom, 1e-9)
    x0_hat = alpha[:, None] * tmpl
    eps_prior = (Xf - sa[:, None] * x0_hat) / sb[:, None]
    eps_prior = eps_prior.reshape(N, 1, H, W_)
    # keep prior bounded (template matching can misfire at very low noise)
    eps_prior = np.clip(eps_prior, -6.0, 6.0)

    # ---------------- 3. ridge readout on cached features ------------------
    feat = np.asarray(f["features"], dtype=np.float64)   # [N,64,16,16]
    P = _patch_features(feat)                             # [N,576,256]
    Yt = y.reshape(N, 1, H * W_)

    # per-timestep-group ridge (groups of 4 timesteps) to let the readout adapt to t
    groups = np.minimum(tt // 4, 7)
    lam = 1e-3 * max(1.0, float(np.trace(_gram(P[:256])) / 576.0))
    Wg = np.zeros((8, 576, H * W_))
    bg = np.zeros((8, H * W_))
    for g in range(8):
        m = groups == g
        if m.sum() < 40:
            m = np.ones(N, dtype=bool)
        Pg = P[m]
        Yg = Yt[m]
        Wg[g] = _ridge_fit(Pg, Yg, lam)
        bg[g] = Yg.mean(axis=(0, 1)) - Wg[g].mean(axis=0)  # crude intercept
    # intercept: residual mean per group
    for g in range(8):
        m = groups == g
        if m.sum() < 40:
            continue
        R = Yt[m] - _apply_W(Wg[g].T, feat[m]).reshape(m.sum(), H * W_)
        bg[g] = R.mean(axis=0)

    # global fallback (all data) for stability
    W_all = _ridge_fit(P, Yt, lam)

    # ---------------- 4. build the "analytic" readout ----------------------
    # We want the network to compute eps_prior. The final layer is linear in the
    # 3x3 patches of the (ReLU'd) conv3 output.  We therefore *re-shape* conv3
    # so that its output carries the needed information, then set out.weight to
    # realise the desired linear map.
    #
    # Simplest robust route: make conv3 an identity-ish pass-through of a
    # signal that already contains eps_prior.  We build that signal inside
    # conv1/conv2 using the *known* linear structure of the diffusion:
    #   eps_prior = (x_t - sa*x0_hat)/sb
    # x0_hat depends on the example (template match), which is not a fixed
    # linear function -> we cannot realise it exactly.  Instead we use the
    # *linear* part:  eps_lin = (x_t - sa*E[x0|x_t])/sb  with E[x0|x_t]
    # approximated by the global clean mean image (a constant) -> this is a
    # per-example affine map of x_t, i.e.  eps_lin = x_t/sb - const(t).
    # A per-example affine map of x_t IS implementable by the conv stack:
    #   conv1: channel0 = x_t + M , channel1 = -x_t + M   (M large)
    #   conv2: channel0 = (x_t + M) + (-x_t + M) ... etc.
    # We realise  out = g(t) * x_t + h(t)  with g,h piecewise-linear in t.
    #
    # Build: conv1 -> 4 channels carrying +-x_t with large offsets (so ReLU is
    # linear), conv2 -> 4 channels carrying g_k(t)*x_t + h_k(t) (t enters as a
    # constant offset through the t-input channel of conv1), conv3 -> identity,
    # out -> sums the 4 channels with the right signs.
    #
    # Because ReLU is applied after conv2, we keep a large positive offset in
    # every conv2 channel so that ReLU is the identity there.

    M = 8.0
    # ---- conv1 (2 -> 32) : 4 "signal" channels + rest set to zero
    cw1 = np.zeros((32, 2, 3, 3), dtype=np.float64)
    cb1 = np.zeros(32, dtype=np.float64)
    # channel 0: +x_t (center tap), bias M
    cw1[0, 0, 1, 1] = 1.0
    cb1[0] = M
    # channel 1: -x_t (center tap), bias M
    cw1[1, 0, 1, 1] = -1.0
    cb1[1] = M
    # channel 2: +x_t, bias M, plus t-signal (broadcast) -> gives t-dependent offset
    cw1[2, 0, 1, 1] = 1.0
    cb1[2] = M
    # channel 3: -x_t, bias M, plus t-signal
    cw1[3, 0, 1, 1] = -1.0
    cb1[3] = M
    # channels 4..: carry t/31 as a constant (spatial) signal, used for
    # t-dependent gain, encoded as:  ch4 = t/31 + M, ch5 = -(t/31) + M
    cw1[4, 1, 1, 1] = 1.0
    cb1[4] = M
    cw1[5, 1, 1, 1] = -1.0
    cb1[5] = M
    # remaining channels: constant M (so that later layers have stable stats)
    for c in range(6, 32):
        cb1[c] = M

    # ---- conv2 (32 -> 64): combine to form g_k(t)*x_t + h_k(t) + big offset
    # we use 4 output channels:
    #   ch0:  x_t + M - (t-part)      -> carries x_t with t-dependent offset
    #   ch1: -x_t + M + (t-part)      -> mirror
    #   ch2:  t-part + M              -> pure t signal
    #   ch3: -t-part + M
    # The t-part is carried by ch4/ch5 of conv1 (which equal +-t/31 + M).
    # We want g(t) = 1/sb(t) and h(t) = -sa*C0/sb(t) (C0 = global mean image).
    # We approximate 1/sb(t) and sa/sb(t) with piecewise-linear functions of
    # u = t/31, using the 8 groups (u breakpoints) -> we use 8 pairs of
    # channels, each pair encoding  g_j * x_t and  -g_j * x_t with the
    # appropriate t-dependent offset so that the *pair sum* yields g_j*x_t.
    # Simpler: use ONE pair per group j, and let the t-signal switch which
    # group is active by making the offsets large/small.  Because ReLU is
    # applied, we use the "both positive" trick:
    #     relu(A) - relu(B) = 2*alpha*x  requires A,B both > 0.
    # Instead we use a cleaner formulation:
    #     relu(u) where u = g_j*x_t + BIG_j(t)   (always > 0)
    #     relu(v) where v = -g_j*x_t + BIG_j(t)  (always > 0)
    #     relu(u) - relu(v) = 2*g_j*x_t
    # and the final out layer sums with weights +1/(2 g_j) for group j.
    # To make only the correct group active, we make BIG_j(t) = M*(1 + delta_j(t))
    # where delta_j(t) is a large positive number only for the active group and
    # a large negative number otherwise.  But the offset must be a *constant*
    # offset; negative offsets would kill the ReLU for the inactive group
    # (giving relu=0 for both u and v -> contributes 0).  That is exactly what
    # we want: inactive groups contribute 0.
    # So: for group j, offset = M * s_j(t) where s_j(t) = +1 if t in group j
    # else a negative number -> relu(u)=relu(v)=0 -> no contribution.
    # Implement s_j(t) as a linear function of the t-signal available in conv1
    # (channels 4 and 5 give +-u + M).  We can build an indicator only
    # approximately (linear in u), so instead we use a *soft* selection:
    # offset_j(t) = M * (1 - |u - u_j| / du) style.  Not linear.
    #
    # => Use a different, fully linear mechanism: implement the piecewise
    # linear function g(t) directly by summing per-group channels with
    # *fixed* weights, where each group channel carries g_j*x_t and the group
    # selection is done by a per-group gate that multiplies... still needs
    # products.
    #
    # FINAL SIMPLE MECHANISM (linear, exact):
    #   Represent g(t) = sum_j theta_j * phi_j(t) where phi_j(t) are the
    #   *linear* basis functions we can build from the t channel: we build
    #   several channels in conv1 that encode u=t/31 with different scalings
    #   and offsets, and in conv2 we form products of x_t with these? no.
    #
    # => Use the fact that we can build ANY function of u as a linear
    #    combination of ReLU units of u (piecewise linear).  Then g(t)*x_t
    #    still needs a product.  BUT: we can instead compute the *whole*
    #    predictor in a different factorisation:
    #       eps_hat = x_t/sb - (sa/sb) * x0_hat
    #    and note that x0_hat is a *template* (per-example), so eps_hat is
    #    not linear in x_t.  However, the *linear* part
    #       eps_lin = x_t/sb - (sa/sb)*C0
    #    is affine in x_t with per-example coefficients (1/sb, sa/sb) that
    #    depend only on t.  Products of a per-example constant with x_t are
    #    needed -> but a per-example constant is available as a *bias-like*
    #    additive term, not multiplicatively.
    #
    #    Trick: implement the multiplication by using the *t channel itself*
    #    inside conv1 to build a per-example scalar that is ADDED to x_t, and
    #    then use the identity
    #       (x_t + c) - x_t = c   (no product needed)
    #    which only gives additive terms.  Multiplication of x_t by a
    #    per-example scalar is genuinely non-linear -> needs ReLU tricks.
    #
    #    ReLU trick for multiplication by a *known* per-example scalar c:
    #    relu(x + c) - relu(x) = c  (if c>0)  -> additive again.
    #    For a product we can use:  relu(x + c) - relu(-x + c) = 2x  (linear).
    #    So the network can compute affine functions of x_t with per-example
    #    *offsets* but not per-example *gains*... unless we use the "gain"
    #    to build the *input* to the nonlinearity: e.g. relu(g*x + c) is a
    #    nonlinear function of x, but we need the gain applied to the OUTPUT.
    #
    #    => Therefore: put the gain on the *input side* of the last ReLU.
    #    i.e. out = sum_k v_k * relu( g_k(t) * x_t + c_k(t) ).
    #    This is a legitimate network.  We choose g_k(t), c_k(t) so that the
    #    resulting function equals the desired affine map  g(t)*x_t + h(t)
    #    approximately.  With two channels per group and large offsets:
    #       relu(g*x + C) - relu(-g*x + C) = 2*g*x   (C > |g*x|)
    #    and the out layer combines them with weights (1/(2g)) and (-1/(2g)).
    #    So each group channel pair contributes exactly g_j*x_t when active.
    #    To make only the active group contribute, we set the offsets C_j(t)
    #    to be *negative* for inactive groups (so both ReLUs output 0) and
    #    positive for the active group.  C_j(t) must be an affine function of
    #    the available t signal u=t/31 (and a constant).  We can build
    #    per-group affine functions of u in conv1 (each conv1 channel can have
    #    its own weight on the t channel and its own bias) -> yes!  conv1 has
    #    32 output channels, each with an independent (weight on u, bias).
    #    So channel c can carry  alpha_c * u + beta_c  (constant over space).
    #    Then conv2 mixes conv1 channels linearly -> we can build
    #    C_j(t) = affine_j(u) with any slope/intercept we like.
    #    Great: this is fully implementable.
    #
    # Implementation detail: conv1 channel c receives BOTH x_t (via its
    # spatial filter) and u (via the t channel).  For the "signal" channels
    # we want  +x_t + alpha*u + beta  and  -x_t + alpha*u + beta.
    # For the "offset" channels we want  alpha*u + beta  (no x_t).
    # conv2 then forms, for each group j:
    #     u_j  =  (+x_t + A_j(u)) + (-x_t + A_j(u)) ... no, we need the gain.
    #
    # Simplify: put the gain g_j into conv2's weights on the x_t-carrying
    # channels:
    #     conv2 channel 2j   pre-activation = g_j * (x_t + A_j(u))  + Mbig
    #     conv2 channel 2j+1 pre-activation = -g_j * (x_t + A_j(u)) + Mbig
    # but that requires conv2 to have separate weights for the x_t part and
    # the A_j(u) part, which it does (conv2 mixes conv1 channels with its own
    # weights).  conv1 channel 0 carries x_t + M (with a spatial center tap),
    # conv1 channel 1 carries -x_t + M.  conv2 weight on channel0 = g_j,
    # on channel1 = -g_j -> gives g_j*(x_t - x_t) ... no: g_j*(x_t+M) - g_j*(-x_t+M) = 2 g_j x_t.
    # Then add the offset: conv2 also weights conv1 channels 4..(t-carriers)
    # to produce A_j(u), plus a bias.  So:
    #     pre = 2*g_j*x_t + A_j(u) + B_j
    # and we want the *pair* (pre, -pre + 2C) style.  Let's define:
    #     ch 2j   : pre = g_j*x_t + A_j(u) + Cb_j
    #     ch 2j+1 : pre = -g_j*x_t + A_j(u) + Cb_j
    # with Cb_j chosen large positive for the active group and the A_j(u)
    # term making it negative for inactive groups.  Then
    #     relu(ch2j) - relu(ch2j+1) = 2*g_j*x_t   (active group)
    #     0                                    (inactive group)
    # and out.weight combines: out = sum_j (1/(2 g_j)) * (relu(ch2j) - relu(ch2j+1)).
    # Wait: relu(ch2j) - relu(ch2j+1) = (g x + A + C) - (-g x + A + C) = 2 g x  only if
    # BOTH are > 0 (i.e. A+C > |g x|).  For the active group we ensure
    # A_j(u)+Cb_j >= 2*Mbig (large).  For inactive groups we need
    # A_j(u)+Cb_j <= -|g_j x| for BOTH -> i.e. strongly negative.
    # So A_j(u) must swing from strongly positive to strongly negative across
    # groups -> A_j(u) = alpha_j*u + beta_j with alpha_j large.  Since groups
    # are 4 timesteps wide, u ranges over [j*4/31, (j+1)*4/31].  We can set
    # alpha_j = -K (negative slope) and beta_j so that A_j is large positive at
    # the start of group j and large negative at the end.  But then within a
    # group, the transition happens -> some timesteps get partial activation.
    # Better: use ONE channel per group with a *hard* switch is impossible
    # linearly.  => Use finer granularity: one group per timestep (32 groups),
    # needing 64 conv2 channels -> exactly 64!  And conv1 needs 32 channels
    # for the t-carriers: 32 groups need A_j(u) affine in u with a switch per
    # timestep -> slope so steep that it flips within one timestep step
    # (u step = 1/31 = 0.032).  With slope K, A changes by K/31 per step.
    # We need A_j(u_j) >= +2Mbig and A_j(u_j + 1/31) <= -2Mbig -> change of
    # 4Mbig over one step -> K = 124*Mbig.  With Mbig = 1 -> K = 124.
    # Then A_j(u) = K*(u_j - u) + something.  But then A_j is positive only
    # for u < u_j and negative for u > u_j, i.e. it selects t <= j, not
    # t == j.  We need a narrow pulse: A_j(u) = K*(u_j - u) for u<=u_j and
    # -K*(u - u_{j+1})? Not affine.
    #
    # SIMPLER: use a *soft* gate and accept a smooth interpolation: the final
    # output is sum_j gate_j(u) * g_j * x_t with gate_j(u) = a triangular /
    # piecewise-linear bump.  We can build bumps as relu of affine functions!
    # Since conv1 channels can be alpha*u + beta, and conv2 can take
    # relu(...) of a combination... but the ReLU is applied to conv2's output,
    # so conv2 can compute relu(alpha*u+beta) = a rectified ramp -> a bump
    # function of u!  So we CAN build piecewise-linear bumps of u.
    #
    # So: conv1 channel c (c=0..31): carries alpha_c*u + beta_c (no x_t),
    #   with alpha_c = K (steep) and beta_c chosen so that relu gives a ramp.
    # conv2 channel j: pre = sum_c v_{jc} * relu(alpha_c u + beta_c) + (x-part)
    #   -> gives bump_j(u) (piecewise linear, 0 outside [u_j, u_{j+1}]).
    # Then combine with x_t: we need bump_j(u) * g_j * x_t -> product again!
    #
    # => The product of a per-example scalar with x_t is the crux.  Solve it
    # by putting the scalar INSIDE the nonlinearity and reading out linearly:
    #     out = sum_j v_j * relu( g_j * x_t + bump_j(u) * BIG + C )
    # With BIG huge, relu(g_j x_t + BIG*bump_j) = BIG*bump_j + g_j x_t when
    # bump_j > 0, and = relu(g_j x_t + C) when bump_j = 0 (C large positive ->
    # = C + g_j x_t).  Hmm, the inactive groups then contribute a constant
    # C + g_j x_t -> not zero.  Set C = 0 and rely on BIG*bump_j to dominate
    # |g_j x_t| for the active group; inactive groups give relu(g_j x_t) which
    # is not zero.  Bad.
    #
    # => Use the difference trick with a *constant* offset:
    #     ch_plus  : pre =  g_j*x_t + BIG*bump_j(u) + M
    #     ch_minus : pre = -g_j*x_t + BIG*bump_j(u) + M
    #     relu(ch_plus) - relu(ch_minus) = 2 g_j x_t  when BIG*bump_j + M > |g_j x_t|
    #     and = 0 when both are <= 0, i.e. when BIG*bump_j + M <= -|g_j x_t|.
    # So we need the offset to be either large positive (active) or large
    # negative (inactive).  With bump_j(u) >= 0 and piecewise linear, we can
    # define offset_j(u) = BIG*(bump_j(u) - c_j) where c_j is a constant
    # threshold: offset is positive when bump_j > c_j.  But bump_j is 0 for
    # most u -> offset = -BIG*c_j (negative) -> inactive groups give 0.
    # For the active group we need bump_j(u) >= c_j -> choose c_j small (e.g.
    # 0.5) and bump_j(u) reaching 1 at the group centre.  So define
    #     bump_j(u) = a triangular function peaking at 1 over the group,
    #     built as relu(ramp_up) - relu(ramp_down) -> but the difference of two
    #     relu's is not available inside conv2 (ReLU applied after conv2).
    #     However conv1 can already contain relu outputs!  conv1's output
    #     goes through ReLU -> we can put rectified ramps there.
    #     So: conv1 channels give r_c(u) = relu(alpha_c*u + beta_c) (ramps).
    #     conv2 channel j: offset part = sum_c v_{jc} r_c(u) -> a piecewise
    #     linear function of u, which we design to be a triangle (peak 1) over
    #     the group and 0 elsewhere: triangle = ramp_up - ramp_down, both
    #     available as separate conv1 channels, combined in conv2 with
    #     coefficients +1 and -1.  Since r_c >= 0, the combination can be
    #     negative -> but conv2's output then goes through ReLU, clipping it.
    #     => we need the triangle to be used as an *offset* inside conv2's
    #     pre-activation, and the ReLU after conv2 would clip negative values.
    #     Fix: shift the triangle by +1 so it's >= 1... but then inactive
    #     groups are also >= 1 (not negative).  Instead: build the triangle
    #     as T_j(u) in [0,1] and use offset = BIG*(T_j(u) - 0.5) -> positive
    #     (>= BIG/2) at the peak, negative (<= -BIG/2) outside.
    #     To have T_j(u) - 0.5 computed inside conv2's pre-activation we need
    #     T_j as a linear combination of conv1 ramp outputs *without* the
    #     -0.5 shift inside the ReLU... The shift can be provided by conv2's
    #     bias? No: bias is a constant, not dependent on u.  But we can add a
    #     conv1 channel that is a constant -0.5 (bias only) -> then conv2 can
    #     include it.  Yes! conv1 channel with weight 0 and bias -0.5 gives a
    #     constant -0.5 (after ReLU it's relu(-0.5)=0!).  Hmm, ReLU kills it.
    #     Use bias +0.5 and subtract in conv2 weights: T_j = ramp_up - ramp_down
    #     and then offset = BIG*T_j - BIG/2: the -BIG/2 must come from a
    #     constant source in conv2's pre-activation: conv2's bias!  conv2's
    #     bias is a constant -> perfect: set conv2 bias for the offset part
    #     to -BIG/2.  But the offset and the x-part live in the SAME conv2
    #     channel (pre = g_j*x_t + BIG*T_j(u) + bias).  The bias is shared, so
    #     we need bias = -BIG/2 for all offset channels -> fine, and the x-part
    #     channel uses the same conv2 channel (pre = g_j*x_t + BIG*T_j(u) - BIG/2).
    #     For the active group: BIG*T_j - BIG/2 >= BIG/2 > |g_j x_t| -> both
    #     plus/minus channels positive -> difference = 2 g_j x_t.
    #     For inactive: T_j = 0 -> offset = -BIG/2 -> both negative -> 0.
    #     This works!  Requirements: g_j*x_t magnitude << BIG/2.
    #     Choose BIG = 4, M = 0 -> offsets in [-2, +2]; |g_j x_t| <= |g_j|*4.
    #     g_j = 1/sb(t) which can be up to ~1/sb(0)... sb(0)=sqrt(1-abar_0)=sqrt(1.5e-4)=0.0124 -> g=80!!  That's huge: g_j*x_t up to 80*1 = 80 >> 2.  Problem at low t.
    #     Fix: at low t, sb is tiny so 1/sb is huge; but at low t the target
    #     eps is nearly unpredictable, so a huge gain is harmful.  Instead of
    #     using g = 1/sb, use the *optimal linear gain* g(t) = sb/(abar*s0^2+sb^2)
    #     which is <= 1 (max 1 at high t).  And the offset h(t) = -g*sa*C0/sb
    #     ... hmm the affine offset also blows up at low t.  But we can drop
    #     the offset term entirely (i.e. predict g(t)*x_t with no mean
    #     correction) — the mean image contribution is small anyway.
    #     With g(t) = sb/(abar*s0^2+sb^2) <= 1, |g*x_t| <= ~4*1 = 4 < BIG/2=2?  Not
    #     always; use BIG = 40 -> offsets +-20, |g x_t| <= 4 -> fine.
    #     But then the offsets are large (20) and the ReLU'd values are ~20,
    #     and the final out.weight multiplies by 1/(2 g_j) which can be huge
    #     (1/(2*0.0124*...)).  Let's compute g(t) at t=0: sb=0.0124, abar=0.99984,
    #     s0^2=0.616 -> g = 0.0124/(0.6158+0.000154)=0.0201.  1/(2g)=24.9.
    #     out.weight entries ~25 * (values ~20) -> fine numerically (float32 ok).
    #     Precision: out.weight ~ 25, features ~ 20 -> products ~500, fine.
    #
    #   This is getting complex but is doable.  However, there is a MUCH
    #   simpler alternative that achieves the same thing: since the desired
    #   predictor is  eps_hat = g(t) * x_t  (a per-example gain times the
    #   input), and the network's FIRST layer sees x_t, we can implement the
    #   gain by scaling the *input* to the network... but the input is fixed.
    #   => implement gain in conv1's spatial weights? conv1's weights are
    #      shared across examples -> no per-example gain.
    #
    #   => The cleanest: implement the gain via the "two-channel" trick at the
    #      FIRST layer, where the t-signal is available:
    #        conv1 ch0: +x_t + alpha*u + beta   (spatial center tap on x_t)
    #        conv1 ch1: -x_t + alpha*u + beta
    #      conv2 ch0: w0*ch0 + w1*ch1 + (t-carrier terms) -> pre = (w0-w1)*x_t
    #        + (w0+w1)*(alpha*u+beta) + ...
    #      We want the *gain* on x_t to depend on u -> impossible linearly.
    #      => confirmed: need the ReLU trick.
    #
    #   SIMPLEST VALID ReLU trick for a per-example gain applied to the
    #   OUTPUT of the network: put the gain on the *input* of the LAST
    #   nonlinearity.  i.e. we want out = g(u)*x_t.  Write
    #       g(u)*x_t = (1/2) * [ (g(u)*x_t + M) - (-g(u)*x_t + M) ]
    #   where the terms inside are pre-activations of conv3 (which is followed
    #   by ReLU) -> we need relu(g x + M) - relu(-g x + M) = 2 g x, requiring
    #   both > 0 -> M > |g x|.  With g <= 1 and |x| <= 4, M = 10 works.
    #   Now, how do we get g(u)*x_t as a pre-activation of conv3?  conv3's
    #   input is relu(conv2 output).  conv2's output channel can be
    #       pre = g_j * x_t + (offset depending on u)
    #   and g_j is a *fixed* number per channel (not u-dependent).  So conv2
    #   channels give g_j*x_t + c_j(u).  Then conv3 mixes them:
    #       conv3 pre for the "+" channel = sum_j v_j * relu(g_j x + c_j(u))
    #   Still the same problem: the gain must multiply x_t inside the ReLU.
    #   BUT: we can make the *pair* trick work at the conv3 level with a
    #   *fixed* gain per channel and use the u-dependence only through the
    #   offsets (which act as gates).  Specifically:
    #       conv2 channel A_j: pre = g_j*x_t + c_j(u)      (c_j(u) = gate)
    #       conv2 channel B_j: pre = -g_j*x_t + c_j(u)
    #   relu(A_j) - relu(B_j) = 2 g_j x_t when c_j(u) > |g_j x_t| (active),
    #   and 0 when c_j(u) < -|g_j x_t| (inactive).  This is exactly the
    #   scheme above, with the gate c_j(u) built from the t-carriers.
    #   And the final output: out = sum_j (1/(2 g_j)) * (relu(A_j) - relu(B_j))
    #   but relu(A_j) and relu(B_j) must be separate channels of conv3's
    #   input -> conv2 has 64 channels: 32 groups * 2 = 64.
    #   conv3 then must be an identity (1x1-ish) mapping: conv3 ch A_j -> conv3
    #   ch A_j, ch B_j -> ch B_j, with the sign handled by out.weight (which
    #   can be negative).  But conv3's ReLU is applied to its output; relu(A_j)
    #   is already >= 0, and conv3 identity keeps it >= 0 -> fine.  Then out
    #   = sum_j (1/(2g_j)) relu(A_j) - (1/(2g_j)) relu(B_j) -> = sum_j g_j x_t
    #   over active groups = g(t) x_t (exactly one group active).
    #   Great — and the out layer is a 3x3 conv with 64 input channels; we can
    #   set it to a 1x1 (center tap) map with the desired weights.
    #
    #   This is implementable.  Let's do it.

    # ---- build the t-carrier channels in conv1 (32 of them) ---------------
    # conv1 output channel c (c=0..31): r_c(u) = relu(alpha_c*u + beta_c)
    # We need, for each timestep j, a triangular gate T_j(u) with support
    # [u_j - h, u_j + h] (h = 1/31) and peak 1 at u_j, built as a difference
    # of two ramps.  A ramp: relu(K*(u - u0)) grows; to make a triangle we need
    # a decreasing ramp too, which relu cannot give directly (relu is
    # increasing).  But we can build a decreasing ramp as relu(K*(u0 - u))? that's
    # decreasing! relu(K*(u0-u)) = K*max(0,u0-u) -> yes, decreasing ramp.
    # So triangle = relu(K*(u - (u_j - h))) - relu(K*((u_j + h) - u))  -> this is
    # a "tent" of width 2h with peak K*h at u_j.  Both terms are >= 0 and are
    # available as conv1 outputs (ramps).  Their difference is formed in conv2
    # (linear combination) -> can be negative outside the support? No:
    # outside the support both are 0 -> difference 0.  Inside, the difference
    # is >= 0 (triangle).  Good, so T_j(u) >= 0 always.
    # Then offset c_j(u) = BIG*(T_j(u) - 0.5) -> but the -0.5 must be added
    # inside conv2's pre-activation; conv2's bias is constant -> we can set
    # conv2 bias = -BIG*0.5 for those channels, but the same channel also
    # carries the g_j*x_t part (from conv1's x-carriers) -> fine, bias is
    # additive to the whole pre-activation.
    # However: the gate T_j must be *centered* so that T_j(u_j) = 1 -> we need
    # T_j(u_j) - 0.5 > 0 -> yes with peak K*h = 1 -> K = 1/h = 31.
    # Then at u_j: T=1 -> offset = BIG*0.5 > 0 (active).  At neighbouring
    # timesteps (|u-u_j| = h): T = 0 -> offset = -BIG*0.5 (inactive).
    # Since u takes discrete values u_j = j/31, the gate is exactly 1 at
    # u_j and 0 at all other grid points.
    # So we need 2 ramp channels per group -> 64 channels for 32 groups.
    # conv1 has only 32 output channels!  But we can reuse: each conv1 channel
    # is relu(alpha*u+beta) — a ramp.  A tent needs 2 ramps per group, but
    # ramps can be shared across groups if we use a *sawtooth* decomposition:
    #   tent_j(u) = relu(K(u - u_j + h)) - relu(K(u_j + h - u))
    # The first term: relu(K(u - (u_j - h))) — for group j, u_j - h = (j-1)/31,
    # so the rising ramp starts at (j-1)/31 -> different per group.  The
    # falling ramp: relu(K((j+1)/31 - u)) -> different per group.
    # => 64 distinct ramps needed.  conv1 has 32 channels.  But conv2 can
    # ALSO produce rectified ramps? No, conv2's output is rectified, so
    # conv2 channels can be relu(linear in u) too — we could build the tents
    # in conv2 directly from conv1's *raw* u channel!  conv1 channel 0 can
    # carry u + M (raw, positive after ReLU) and conv1 channel 1 can carry
    # -u + M.  Then conv2 channel j: pre = a*ch0 + b*ch1 + ... -> linear in u
    # -> after ReLU gives a ramp.  So conv2's output channels ARE ramps of u.
    # Then conv3's pre-activation can combine conv2's ramps to form tents:
    #   conv3 pre = relu-ramp_up - relu-ramp_down ... but conv3's output is
    #   rectified too -> the difference would be clipped to >= 0 -> the tent
    #   (which is >= 0 anyway) survives, but the "-0.5" shift can't be applied
    #   inside conv3 without clipping... conv3's bias is constant -> we CAN
    #   apply the constant shift in conv3's bias!  conv3 pre = tent_j(u) - 0.5
    #   -> this is negative outside the support -> ReLU clips to 0.  Hmm, that
    #   destroys the "inactive" detection (we need a large negative offset,
    #   and clipping to 0 loses it).
    #
    #   => Put the gate+shift in conv2 (whose output is rectified) — no.
    #   => Put the gate in conv3's PRE-activation (linear in u, no clipping
    #      needed there) and let conv3's ReLU clip: we want the gate to be
    #      either large positive (active) or large negative (inactive) at the
    #      conv3 pre-activation level, and then conv3's ReLU gives:
    #        active: relu(large positive + g x) = large positive + g x
    #        inactive: relu(large negative + g x) = 0 (if large negative)
    #      Then out.weight reads out (relu_out - relu_out_of_mirror)/2g.
    #      So the gate must be at conv3's pre-activation, and the +/-x_t pair
    #      must ALSO be at conv3's pre-activation -> conv3 pre = g_j*x_t +
    #      gate_j(u) and conv3 pre' = -g_j*x_t + gate_j(u).  conv3's input is
    #      relu(conv2 output) -> we need conv2 to output channels carrying
    #      +x_t (times g_j) and -x_t (times g_j) with large offsets so they
    #      survive conv2's ReLU, and conv3 to add the gate.
    #      conv3 pre = sum over conv2 channels of w * relu(conv2 out).
    #      To get g_j*x_t we need relu(conv2 out) to contain x_t (with sign) —
    #      conv2 out channel = relu(conv1 stuff) and conv1 can carry +-x_t + M.
    #      So conv2 channel k: pre = alpha_k*(x_t + M) + beta_k*(-x_t + M) + ...
    #      -> we can get any multiple of x_t plus constants.  With M large,
    #      relu(conv2 out) = alpha_k x_t + (alpha_k+beta_k) M + ...  -> the
    #      constant part is large and would dominate conv3's pre-activation.
    #      We can cancel constants: conv3 pre = sum_k w_k relu(conv2_k) with
    #      sum_k w_k*(constant part) = 0 -> we can cancel the M-constants
    #      exactly (choose weights so they cancel) leaving g_j*x_t + gate.
    #      But the gate itself must be built from conv2 channels that carry u
    #      (ramps).  Since conv2's output = relu(linear in u) is a ramp, and
    #      conv3's pre-activation is a linear combination of conv2 outputs
    #      (all >= 0), we can build the tent as a difference of two ramps
    #      (both >= 0) -> tent >= 0, and then subtract a constant via conv3's
    #      bias -> gate = tent - 0.5 scaled.  But conv3's bias is per-channel
    #      and constant -> yes!  conv3 bias for the gate-carrying channel = -BIG*0.5.
    #      Wait, but conv3's pre-activation for the "+" channel must contain
    #      BOTH g_j*x_t and the gate.  conv3 pre = sum_k w_k relu(conv2_k) + bias.
    #      The x-part comes from conv2 channels carrying x_t; the gate part
    #      from conv2 channels carrying ramps of u; the constant -BIG*0.5 from
    #      the bias.  All additive -> fine!
    #
    #   So the plan:
    #     conv1 (2->32):
    #       ch0: +x_t (center tap) + bias M1        -> relu gives x_t + M1
    #       ch1: -x_t (center tap) + bias M1        -> relu gives -x_t + M1
    #       ch2: +u (center tap on t channel) + bias M1 -> relu gives u + M1
    #       ch3: -u + M1
    #       ch4..31: unused (bias 0) -> relu gives 0.
    #     conv2 (32->64): produce 64 channels:
    #       for each group j (32 groups) we need 2 channels: A_j and B_j.
    #       A_j pre = g_j * (x_t + M1)  [from ch0]  -> relu -> g_j x_t + g_j M1
    #       B_j pre = -g_j * (x_t + M1) [from ch0]  -> relu -> relu(-g_j x_t - g_j M1)
    #            which is 0 for g_j>0 -> useless.  Instead B_j should carry
    #            -g_j x_t: use ch1: B_j pre = -g_j*(-x_t + M1) = g_j x_t - g_j M1
    #            -> relu -> relu(g_j x_t - g_j M1) -> 0 unless x_t > M1.  Hmm.
    #       The problem: to get -x_t we need a channel whose relu output is
    #       (-x_t + M1) and then multiply by -g_j: relu(-g_j*(-x_t+M1)) =
    #       relu(g_j x_t - g_j M1) -> clipped at 0 for x_t < M1.  So we
    #       cannot get a negative multiple of x_t through a single ReLU.
    #       => use TWO conv2 channels per sign and subtract at conv3:
    #         conv2 ch A_j: pre = g_j*(x_t + M1) -> relu -> g_j x_t + g_j M1
    #         conv2 ch B_j: pre = g_j*(-x_t + M1) -> relu -> -g_j x_t + g_j M1
    #       Both are positive (M1 large).  Then conv3 pre for the "+" output
    #       channel = w_A*A_j + w_B*B_j with w_A = 1, w_B = -1 -> = 2 g_j x_t
    #       (the M1 constants cancel).  Similarly we need the gate added:
    #         conv3 pre = A_j - B_j + gate_j(u) + bias.
    #       And we need TWO conv3 output channels per group (the "+" and the
    #       "-" mirror) to then subtract at the out layer:
    #         conv3 out ch P_j: pre = (A_j - B_j)/2 + gate_j(u)  -> = g_j x_t + gate
    #         conv3 out ch N_j: pre = -(A_j - B_j)/2 + gate_j(u) -> = -g_j x_t + gate
    #       conv3 has 64 channels = 32 groups * 2.
    #       Then out (1x64x3x3): out = sum_j (1/(2 g_j)) * (P_j - N_j)
    #         -> for active j: (1/(2g_j))*(2 g_j x_t) = x_t  ... wait we want
    #         g(t)*x_t, and P_j - N_j = 2 g_j x_t -> (1/(2 g_j)) * that = x_t.
    #         So set out weight = 1/(2 g_j) for P_j and -1/(2 g_j) for N_j.
    #         -> out = x_t for the active group.  But we want g(t)*x_t!
    #         Set out weight for P_j = g_j/2 and for N_j = -g_j/2 -> out = g_j x_t.
    #         Wait: (g_j/2)*(P_j - N_j) = (g_j/2)*(2 g_j x_t) = g_j^2 x_t.  No.
    #         Let me redo: P_j - N_j = 2 g_j x_t.  We want out = g(t) x_t.
    #         If group j is active with gain g_j, then out = c_j * (P_j - N_j)
    #         = c_j * 2 g_j x_t -> we want c_j * 2 g_j = g(t) -> c_j = g(t)/(2 g_j).
    #         Since only one group is active, set c_j = 1/2 (i.e., out weight
    #         = 0.5 for P_j, -0.5 for N_j) -> out = g_j x_t.
    #         But careful: for INACTIVE groups, P_j - N_j = 0 (both clipped to 0
    #         by conv3's ReLU) -> no contribution.
    #         Also for inactive groups, gate_j(u) must be <= -|g_j x_t| so that
    #         conv3's ReLU kills both P_j and N_j.  With gate = BIG*(tent - 0.5)
    #         and tent = 0 outside -> gate = -BIG/2 -> need BIG/2 > |g_j x_t|.
    #         |g_j x_t| <= g_j * 4 (max |x_t| ~ 4).  g_j <= 1 (optimal gain)
    #         -> need BIG/2 > 4 -> BIG = 20.
    #         For active groups: gate = BIG*(1 - 0.5) = BIG/2 = 10 > |g_j x_t|.
    #       Also conv3's ReLU output for P_j = relu(g_j x_t + 10) = g_j x_t + 10
    #       (large constant) and N_j = relu(-g_j x_t + 10) = -g_j x_t + 10.
    #       Then out = 0.5*(P_j - N_j) = g_j x_t.  The constants cancel exactly.
    #       But the out layer also sums over ALL groups; inactive groups give
    #       relu(negative) = 0 -> fine.
    #       Precision: values ~10-14 in float32 -> fine.
    #
    #       The tent functions: tent_j(u) = relu(K(u - u_{j-1})) - relu(K(u_{j+1} - u))
    #       with K = 1/h = 31, h = 1/31.  At u = u_j: first term = 31*(1/31)=1,
    #       second = 0 -> tent = 1.  At u = u_{j+1}: first = 2, second = 0 ->
    #       tent = 2!  That's wrong (should be 0 outside).  Let me redefine:
    #       tent_j(u) = relu(K(u - u_{j-1})) - relu(K(u - u_j))
    #       -> at u=u_j: 1 - 0 = 1; at u=u_{j+1}: 2 - 1 = 1; at u=u_{j-1}: 0-0=0.
    #       That's a ramp, not a tent: it's 0 for u<=u_{j-1}, then increases.
    #       We want a bump: relu(K(u-u_{j-1})) - relu(K(u-u_{j+1})) -> at u=u_j:
    #       1 - 0 = 1; at u=u_{j+1}: 2 - 0 = 2; nope.
    #       Correct bump: relu(K(u - u_{j-1})) - relu(K(u - u_{j+1})) is wrong.
    #       Use: relu(K(u - u_{j-1})) - relu(K(u - u_j)) - relu(K(u - u_{j+1}))?
    #       Simplest: use *two* ramps with opposite slopes:
    #         up_j(u)   = relu(K(u - u_{j-1}))            (increasing, 0 at u<=u_{j-1})
    #         down_j(u) = relu(K(u_{j+1} - u))            (decreasing, 0 at u>=u_{j+1})
    #         tent_j(u) = min(up_j, down_j)  -- not linear.
    #       Alternative: use the *derivative* trick: since u is on a discrete
    #       grid, we can build the indicator of u == u_j as a difference of
    #       ramps:  I_j(u) = relu(K(u - u_{j-1})) - 2*relu(K(u - u_j)) + relu(K(u - u_{j+1}))
    #       -> at u=u_j: 1 - 0 + 0 = 1; at u=u_{j+1}: 2 - 2*1 + 0 = 0;
    #       at u=u_{j-1}: 0; at u=u_{j+2}: 3 - 2*2 + 1 = 0.
    #       This is a linear combination of 3 ramps -> all ramps are conv2
    #       outputs (>=0) and the combination is formed in conv3's
    #       pre-activation (linear) -> fine, and the result is exactly the
    #       indicator on the grid (values 0 or 1).
    #       But we need 3 ramps per group -> 96 ramp channels needed in conv2,
    #       and conv2 has 64 outputs.  However, the ramps are shared:
    #       ramp_r(u) = relu(K(u - u_r)) for r = -1..32 -> 34 distinct ramps.
    #       Each conv2 output channel can be one ramp (conv2 pre = linear in u
    #       via ch2/ch3 of conv1).  So conv2 needs 34 ramp channels + 64
    #       x-carrying channels = 98 > 64.  Hmm.
    #       Reduce: use fewer groups! We don't need per-timestep granularity:
    #       the gain g(t) varies smoothly; we could use, say, 8 groups of 4
    #       timesteps with a *linear* gate within the group... but then the
    #       gate is not an indicator.
    #       Alternative: use the *indicator* trick but with a coarser set:
    #       we need gates for the groups; each group gate needs 3 ramps.
    #       With 8 groups -> 8*3 = 24 ramp channels (ramp indices shared
    #       across adjacent groups: group j uses ramps at u_{j-1}, u_j, u_{j+1}
    #       -> 10 distinct ramps).  So 10 ramp channels + 16 x-carrying
    #       channels (2 per group) = 26 <= 64.
    #       But then within a group of 4 timesteps the gain is constant -> we
    #       lose the per-timestep adaptation of g(t).  With 8 groups the gain
    #       error is modest.  Alternatively use 16 groups of 2 timesteps:
    #       18 ramps + 32 x-channels = 50 <= 64.  Good compromise.
    #       Even better: 32 groups needs 34 ramps + 64 x-channels = 98 > 64.
    #       => Use 16 groups (2 timesteps each) -> 50 channels used, 14 spare.
    #       Hmm, but actually we could avoid the x-carrying duplication: we need
    #       per-group gains g_j, so per-group x channels are needed (2 each).
    #       Alternatively use a single set of x channels with gain 1 and put the
    #       gain into the *out* layer per group... but out layer weights are
    #       per (group, channel) -> the gain can be applied at the out layer!
    #       i.e. conv3 P_j = relu(x_t + gate_j), N_j = relu(-x_t + gate_j) ->
    #       P_j - N_j = 2 x_t (independent of group) -> out = 0.5*(P_j - N_j)
    #       = x_t -> then apply the gain via out weights: out = sum_j c_j (P_j - N_j)
    #       with c_j = g_j/2 -> out = g_j x_t for the active group.  YES!
    #       So we only need ONE pair of x-carrying channels (2 channels in
    #       conv2: +x_t+M, -x_t+M) shared by all groups, and conv3's P_j/N_j
    #       channels take the SAME x-part but different gates.
    #       conv3 channels: 2 per group (P_j, N_j) -> 32 groups -> 64 channels
    #       exactly!  And the ramps must be formed in conv2 -> conv2 has 64
    #       channels: 2 x-carrying + ramps.  For 32 groups we need 34 ramps ->
    #       36 channels <= 64.
    #       But wait: conv3's pre-activation for P_j = (A - B)/2 + gate_j where
    #       A = relu(conv2 x+ channel), B = relu(conv2 x- channel) -> A - B =
    #       2 x_t (constants cancel).  So conv3 pre = x_t + gate_j(u) + bias.
    #       Hmm, the (A-B)/2 combination is a linear combination of conv2
    #       outputs -> fine, done inside conv3's weights.
    #       And gate_j(u) = BIG*(I_j(u) - 0.5) where I_j is the indicator.
    #       I_j(u) = up_{j-1} - 2*up_j + up_{j+1} with up_r(u)=relu(K(u-u_r)).
    #       These are conv2 outputs (ramps) -> conv3 weights combine them.
    #       The -0.5*BIG comes from conv3's bias (constant per channel).
    #       So conv3 pre(P_j) = 0.5*(A - B) + BIG*(up_{j-1} - 2 up_j + up_{j+1}) - BIG/2
    #       where up_r are conv2 outputs.  But conv3's weights act on conv2's
    #       64 channels with a 3x3 spatial kernel; the ramps are spatially
    #       constant, so use the center tap only (and zero elsewhere) -> the
    #       spatial structure is preserved.
    #       Also note: the x-part (A-B)/2 must be a *spatial* 3x3 mapping:
    #       A and B are conv2 outputs that are spatially varying (they carry
    #       x_t).  conv3's center tap on A and B gives x_t at the same pixel.
    #       Good.
    #       Also we need conv3's output ReLU to not clip: pre(P_j) = x_t + gate
    #       with gate = +BIG/2 (active) or -BIG/2 (inactive) -> for active:
    #       x_t + 10 > 0 always (|x_t| < 4) -> relu = x_t + 10.
    #       For inactive: x_t - 10 < 0 -> relu = 0.  And N_j: pre = -x_t + gate
    #       -> active: -x_t + 10 > 0 -> relu = -x_t + 10; inactive: 0.
    #       Then out = sum_j c_j*(P_j - N_j) with c_j = g_j/2 -> active group
    #       contributes g_j/2 * (2 x_t) = g_j x_t.
    #       BUT: the out layer sums over all 64 conv3 channels; for inactive
    #       groups P_j = N_j = 0 -> no contribution.
    #       Also: the out layer must be a 3x3 conv over the 64 channels -> use
    #       center taps only.
    #       One more check: conv2's x-carrying channels: A = relu(conv2 pre)
    #       with conv2 pre = alpha*(x_t + M1) -> we want A = x_t + M1 exactly
    #       (alpha=1) -> A = x_t + M1 (positive if M1 > 4).  B: conv2 pre =
    #       beta*(-x_t + M1) with beta=1 -> B = -x_t + M1.  Then A - B = 2 x_t.
    #       conv3 pre(P_j) = 0.5*(A - B) + gate = x_t + gate.
    #       But conv3's ReLU clips at 0 -> for active group relu(x_t + 10) = x_t+10.
    #       Then out = c_j*(P_j - N_j) = c_j*(x_t + 10 - (-x_t + 10)) = c_j*2x_t.
    #       With c_j = g_j/2 -> g_j x_t.
    #       Now the magnitudes: conv3 outputs ~10, out weights ~g_j/2 <= 0.5.
    #       Fine.
    #
    #   Now, the gain g_j: use the optimal linear gain g(t) = sb/(abar*s0^2+sb^2)
    #   with s0^2 estimated from the data (per-pixel variance of clean).
    #   Also we could add the mean-image offset: eps_hat = g*(x_t - C0)? The
    #   optimal affine predictor is eps_hat = g*(x_t - mu_x) where mu_x is the
    #   mean of x_t = sa*C0.  We can implement the constant offset via the out
    #   layer's bias (a constant per example? No! out.bias is a constant, but
    #   the needed offset is -g(t)*sa*C0 which depends on t and on the pixel
    #   (C0 is an image).  The pixel-dependent part can't go in the bias.
    #   => Skip the mean correction, or handle it via the ridge stage.
    #   Actually we can incorporate the mean correction later via the ridge
    #   residual stage (which can add a constant image per group? the ridge
    #   readout has per-group intercepts -> yes, the ridge stage can add
    #   group-specific constant images!).  Nice: the ridge stage (applied on
    #   top of the analytic features) can learn per-group constant offsets.
    #
    #   Hmm, but the ridge stage operates on the *features* (conv3 output),
    #   which after our rebuild are ~10-valued.  The ridge stage would then
    #   fit the residual.  Let's structure:
    #     Stage A: rebuild the network to implement eps_hat_analytic = g(t)*x_t.
    #     Stage B: recompute features with the new network, then fit a ridge
    #     readout (per group) on the new features to predict the residual
    #     (y - eps_hat_analytic), and then UNFOLD that readout into out.weight
    #     (so the final network = analytic part + ridge part).
    #     Unfolding: out_new = out_analytic + out_ridge, where out_ridge is
    #     derived from the ridge solution on the new features.  Since the ridge
    #     readout is linear in the features, and out is a linear map of the
    #     features (3x3 conv), we can just add the ridge-derived conv weights
    #     to out.weight.  The features change only if conv1/2/3 change — and
    #     they do (we rebuilt them) — so we must recompute features with the
    #     rebuilt conv1/2/3 and the analytic out, then fit the ridge, then add.
    #     The ridge fit uses features computed with the analytic out layer, but
    #     the final out layer = analytic + ridge, and the features (conv3 output)
    #     don't depend on out at all!  conv3's output is the input to out.
    #     So: features = conv3 output, independent of out.  So the ridge fit on
    #     conv3's output is exact and can be added to out.weight directly.
    #     The only subtlety: the ridge readout is fit to predict y (or the
    #     residual) from the features; since the features are fixed (given the
    #     new conv1/2/3), adding out_ridge to out_analytic gives exactly
    #     analytic + ridge(features) = the intended prediction.
    #     But careful: the analytic part's contribution to the prediction is
    #     out_analytic applied to features; the ridge part is out_ridge applied
    #     to the SAME features -> sum is linear -> fine.
    #
    #   So the final prediction = out_analytic·features + out_ridge·features.
    #   And we want it to equal y.  We can fit the ridge on the residual
    #   r = y - out_analytic·features, then out_ridge = argmin ||out_ridge·F - r||.
    #   That's exactly a linear least squares on the features -> closed form.
    #   And to make the ridge part expressible as a 3x3 conv on the features,
    #   we fit with the patch formulation (which is exactly a 3x3 conv).
    #
    #   This is elegant: the analytic network provides a strong, structured
    #   feature set (essentially x_t replicated with gates), and the ridge
    #   readout adds a data-driven correction (which can capture the digit
    #   structure, e.g. learning to output ~0 in the background at low t).
    #
    #   Note the ridge readout on features that are mostly constant (10) and
    #   x_t-like: the Gram matrix will be dominated by the constant part ->
    #   use a decent ridge lambda and center the features? Centering changes
    #   the intercept; we can handle the intercept via out.bias (constant).
    #   Let's center features per channel (subtract the mean over the dataset)
    #   for the ridge fit, and put the mean back via out.bias.  Actually the
    #   features are spatially constant offsets (10) plus x_t; centering per
    #   channel removes the constant -> the fit then uses the x_t part.  Then
    #   out_ridge·features = out_ridge·(F - mean) + out_ridge·mean; we add
    #   out_ridge·mean to out.bias.
    #
    #   Since the analytic part already provides g(t)*x_t, the ridge residual
    #   fit will try to improve on it.  With 2000 examples and 576 features,
    #   the ridge can learn a lot (e.g., a linear map from x_t to eps residual
    #   that is t-dependent via the gate channels).
    #   Actually, the features include the gated channels P_j, N_j which are
    #   zero except for the active group -> the ridge can use them to learn
    #   per-group corrections (like a per-timestep affine map of x_t and
    #   per-group constant images).  That's exactly the right structure!
    #   So the ridge stage can learn: eps_hat = sum_j [a_j * x_t + b_j] * I_j(t)
    #   -> per-group affine predictors, plus more.
    #   With 16 groups (2 timesteps each) we get decent resolution; with 32
    #   groups even better.  Let's use 32 groups if the channel budget allows:
    #     conv2 needs: 2 x-channels + 34 ramp channels = 36 <= 64. OK!
    #     conv3 needs: 64 channels = 32 groups * 2 (P_j, N_j). OK!
    #     So 32 groups works.
    #   Then the ridge stage can learn per-timestep affine maps of x_t (32
    #   distinct groups) -> that's the optimal linear predictor per timestep,
    #   which is exactly the theoretical optimum ~0.44 MSE!  And the ridge
    #   can also learn cross-channel corrections (using the 3x3 neighborhoods).
    #   This is a strong solution.

    # ---- choose groups: 32 (one per timestep) ----
    NG = 32
    # ramp indices: we need up_r(u) = relu(K(u - u_r)) for r = -1 .. 32
    # with u_r = r/31 (r from -1 to 32).  K = 31.
    K = 31.0
    # conv1 channels:
    #   0: +x_t + M1   (spatial center tap on channel 0 of input)
    #   1: -x_t + M1
    #   2: +u + M1    (center tap on t channel)
    #   3: -u + M1
    #   4..: unused
    M1 = 6.0
    cw1 = np.zeros((32, 2, 3, 3), dtype=np.float64)
    cb1 = np.zeros(32, dtype=np.float64)
    cw1[0, 0, 1, 1] = 1.0; cb1[0] = M1
    cw1[1, 0, 1, 1] = -1.0; cb1[1] = M1
    cw1[2, 1, 1, 1] = 1.0; cb1[2] = M1
    cw1[3, 1, 1, 1] = -1.0; cb1[3] = M1
    # conv2 channels:
    #   0: A = x_t + M1   (from conv1 ch0, weight 1)
    #   1: B = -x_t + M1  (from conv1 ch1, weight 1)
    #   2 .. 2+34: ramps up_r for r = -1..32
    #   ramp r: relu(K(u - u_r)) = relu(K*u - K*u_r) = relu( K*(ch3 - M1) ... )
    #   we have ch2 = u + M1, ch3 = -u + M1.  K*u = K*(ch2 - M1).
    #   conv2 pre = K*ch2 - K*M1 - K*u_r  -> weight on ch2 = K, bias = -K*(M1+u_r).
    #   Need pre > 0 for u > u_r; at u = u_r exactly pre = 0 -> relu = 0. good.
    nramp = 34
    cw2 = np.zeros((64, 32, 3, 3), dtype=np.float64)
    cb2 = np.zeros(64, dtype=np.float64)
    cw2[0, 0, 1, 1] = 1.0                       # A
    cw2[1, 1, 1, 1] = 1.0                       # B
    for idx, r in enumerate(range(-1, 33)):
        ch = 2 + idx
        cw2[ch, 2, 1, 1] = K
        cb2[ch] = -K * (M1 + r / 31.0)
    # conv3 channels: 2 per group j (0..31): P_j, N_j
    #   P_j pre = 0.5*(A - B) + BIG*(up_{j-1} - 2*up_j + up_{j+1}) - BIG/2
    #   N_j pre = -0.5*(A - B) + BIG*(up_{j-1} - 2*up_j + up_{j+1}) - BIG/2
    #   where up_r is conv2 channel 2 + (r+1).
    #   Note: the -BIG/2 constant must be added via conv3's bias, but the
    #   same channel also has the x-part; bias is additive -> fine.
    BIG = 20.0
    cw3 = np.zeros((64, 64, 3, 3), dtype=np.float64)
    cb3 = np.zeros(64, dtype=np.float64)
    for j in range(NG):
        p_ch = 2 * j
        n_ch = 2 * j + 1
        # x-part (center taps on conv2 ch0 (A) and ch1 (B))
        cw3[p_ch, 0, 1, 1] = 0.5
        cw3[p_ch, 1, 1, 1] = -0.5
        cw3[n_ch, 0, 1, 1] = -0.5
        cw3[n_ch, 1, 1, 1] = 0.5
        # gate part
        for r, coef in ((j - 1, 1.0), (j, -2.0), (j + 1, 1.0)):
            ch = 2 + (r + 1)
            cw3[p_ch, ch, 1, 1] += BIG * coef
            cw3[n_ch, ch, 1, 1] += BIG * coef
        cb3[p_ch] = -BIG * 0.5
        cb3[n_ch] = -BIG * 0.5
    # out layer: out = sum_j c_j * (P_j - N_j) with c_j = g_j/2 (per-timestep
    # optimal gain).  Use center taps only.
    ow = np.zeros((1, 64, 3, 3), dtype=np.float64)
    # per-timestep gain g(t) from data
    s0_var = float(np.mean(np.var(clean, axis=0)))
    g_t = np.zeros(32, dtype=np.float64)
    for j in range(32):
        ab = float(abar[j])
        bb2 = max(1.0 - ab, 1e-8)
        den = ab * s0_var + bb2
        g_t[j] = np.sqrt(bb2) / max(den, 1e-8)
    for j in range(NG):
        ow[0, 2 * j, 1, 1] = 0.5 * g_t[j]
        ow[0, 2 * j + 1, 1, 1] = -0.5 * g_t[j]
    ob = np.zeros(1, dtype=np.float64)

    # sanity check the analytic construction on the training batch
    w_try = dict(w)
    w_try["conv1.weight"] = cw1; w_try["conv1.bias"] = cb1
    w_try["conv2.weight"] = cw2; w_try["conv2.bias"] = cb2
    w_try["conv3.weight"] = cw3; w_try["conv3.bias"] = cb3
    w_try["out.weight"] = ow; w_try["out.bias"] = ob
    pred_try, feat_try = _forward(w_try, x, tt)
    mse_try = float(np.mean((pred_try - y) ** 2))
    diag["analytic_mse"] = mse_try
    diag["g_t"] = [float(v) for v in g_t]
    diag["s0_var"] = s0_var

    # ---- Stage B: ridge residual on the new features ---------------------
    # features = conv3 output (input to out).  Fit per-group ridge to residual.
    # Use per-timestep groups (32) with shared ridge solution?  32 groups x
    # 576 features is too many params for ~62 examples per group.  Use
    # group-level ridge with a *shared* structure: fit a single ridge on all
    # data using the features (which already encode the group via the gates),
    # plus per-group intercepts (constant images).
    # Simpler & robust:
    #   (a) global ridge on features -> residual prediction (captures x_t-based
    #       corrections shared across t, plus the gate channels give per-t
    #       modulation automatically because the gate channels are zero for
    #       inactive groups).
    #   (b) per-group intercept (constant image) added to out.bias? No:
    #       out.bias is a single scalar, not per group.  But the gate channels
    #       P_j/N_j are per-group -> a per-group constant image can be
    #       produced by a readout that uses the *constant* part of P_j/N_j
    #       (which is 10 for the active group).  Indeed P_j = x_t + 10 for
    #       active j, so a readout weight on P_j gives a per-group constant
    #       (10*w) plus a per-group x_t term (w*x_t).  So the ridge on the
    #       features automatically learns per-group affine maps.
    #   So a single global ridge on the 576-dim patch features suffices, and it
    #   can express per-timestep affine predictors.
    F = feat_try[:, :, :, :]                       # [N,64,16,16] conv3 output
    Pn = _patch_features(F)                        # [N,576,256]
    resid = (y - pred_try).reshape(N, H * W_)
    # center features per channel for numerical stability, keep intercept
    Fmean = F.mean(axis=0, keepdims=True)
    Pc = _patch_features(F - Fmean)
    lam = 1e-2 * max(1.0, float(np.trace(_gram(Pc[:256])) / 576.0))
    Wr = _ridge_fit(Pc, resid, lam)               # [576, 256]
    # convert to out.weight (center taps) + bias contribution
    ow_r = np.zeros((1, 64, 3, 3), dtype=np.float64)
    k = 0
    for i in range(3):
        for j2 in range(3):
            ow_r[0, :, i, j2] = Wr[k * 64:(k + 1) * 64, :].T
            k += 1
    # bias: out.bias = sum over channels of ow_r * Fmean (per pixel) -> but
    # out.bias is a single scalar!  The mean feature Fmean is a [64,16,16]
    # image; the contribution ow_r·Fmean is a [1,16,16] image -> cannot go
    # into a scalar bias.  => Do NOT center, or handle the mean by keeping
    # the uncentered fit (which includes the constant in the weights).
    # Redo without centering (ridge handles the scale).
    Wr2 = _ridge_fit(Pn, resid, lam)
    ow_r2 = np.zeros((1, 64, 3, 3), dtype=np.float64)
    k = 0
    for i in range(3):
        for j2 in range(3):
            ow_r2[0, :, i, j2] = Wr2[k * 64:(k + 1) * 64, :].T
            k += 1
    # evaluate the combined prediction
    w_try["out.weight"] = ow + ow_r2
    w_try["out.bias"] = ob
    pred2, _ = _forward(w_try, x, tt)
    mse2 = float(np.mean((pred2 - y) ** 2))
    diag["ridge_mse"] = mse2

    # keep whichever is better (analytic only vs analytic+ridge)
    if mse2 <= mse_try:
        final = w_try
        diag["chosen"] = "analytic+ridge"
    else:
        final = dict(w)
        final["conv1.weight"] = cw1; final["conv1.bias"] = cb1
        final["conv2.weight"] = cw2; final["conv2.bias"] = cb2
        final["conv3.weight"] = cw3; final["conv3.bias"] = cb3
        final["out.weight"] = ow; final["out.bias"] = ob
        diag["chosen"] = "analytic"

    # ---- Stage C: optional additional refinement on the ORIGINAL features --
    # (kept out: the rebuilt features supersede them)

    out_w = {k: np.asarray(v, dtype=np.float32) for k, v in final.items()}
    out_w = _clip_all(out_w)
    diag["final_train_mse"] = float(min(mse_try, mse2))
    return {"weights": out_w, "diagnostics": diag}
