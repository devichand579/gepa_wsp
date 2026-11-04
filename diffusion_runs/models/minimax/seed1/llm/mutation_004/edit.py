import numpy as np

def edit(w, f):
    # ---------------------------------------------------------------
    # Closed-form construction of an epsilon-predictor inside the fixed
    # conv stack.  No autograd / iterative optimisation is used.
    # ---------------------------------------------------------------
    W = {k: np.array(v, dtype=np.float64) for k, v in w.items()}

    clean = np.asarray(f["clean"], dtype=np.float64)      # [N,1,16,16]
    x     = np.asarray(f["x"], dtype=np.float64)          # [N,1,16,16]
    t     = np.asarray(f["t"], dtype=np.int64).reshape(-1)
    y     = np.asarray(f["y"], dtype=np.float64)          # [N,1,16,16]
    abar  = np.asarray(f["alphabar"], dtype=np.float64).reshape(-1)

    N = clean.shape[0]
    H = W["clean"].shape[-1] if False else clean.shape[-2]
    Wd = clean.shape[-1]

    # ---- schedule -------------------------------------------------
    ab = abar[np.clip(t, 0, len(abar) - 1)]
    sa = np.sqrt(np.clip(ab, 0.0, 1.0))
    sb = np.sqrt(np.clip(1.0 - ab, 1e-8, 1.0))

    # ---- clean-image statistics (per-pixel mean, per-pixel variance)
    m  = clean.mean(axis=0)                                # [1,16,16]
    var = clean.var(axis=0)                                # [1,16,16]
    sig2 = float(var.mean())

    # ---- verify the forward identity:  x = sa*clean + sb*eps -----
    eps_hat_from_data = (x - sa[:, None, None, None] * clean) / sb[:, None, None, None]
    resid = float(np.mean((eps_hat_from_data - y) ** 2))

    # ---- optimal (per-pixel, per-t) linear gain & offset ----------
    # c(t) = sb / (ab*sigma2 + 1-ab)   ;  d(t) = -c(t)*sa*m
    den = np.clip(ab * sig2 + (1.0 - ab), 1e-6, None)
    c_t = sb / den
    d_t = -c_t * sa

    # ---- per-timestep empirical refinement ------------------------
    # For each t, regress y on x (and on the mean image) to get the
    # best achievable affine predictor  y ~ g*x + k  (per pixel field).
    g_tab = np.zeros(len(abar))
    k_tab = np.zeros(len(abar))
    for ti in range(len(abar)):
        sel = t == ti
        if sel.sum() < 8:
            g_tab[ti] = c_t[sel][0] if sel.any() else 0.0
            k_tab[ti] = d_t[sel][0] if sel.any() else 0.0
            continue
        xs = x[sel].reshape(sel.sum(), -1)
        ys = y[sel].reshape(sel.sum(), -1)
        ms = np.broadcast_to(m.reshape(1, -1), xs.shape)
        A = np.concatenate([xs, ms, np.ones((xs.shape[0], 1))], axis=1)
        sol, *_ = np.linalg.lstsq(A, ys, rcond=None)
        g_tab[ti] = float(sol[0].mean())
        k_tab[ti] = float(sol[-1].mean())

    # ---- fit the gain / offset as piecewise-linear functions of u=t/31
    # Use a hinge basis  phi_j(u) = relu(u - b_j)  (j = 0..K-1) plus 1.
    # Solve a small least-squares problem for the coefficients so that
    #   sum_j a_j phi_j(u) ~= g(t),   sum_j b_j phi_j(u) ~= k(t)
    K = 16
    u = t.astype(np.float64) / 31.0
    brk = np.linspace(0.0, 1.0, K + 1)[1:-1]
    Phi = np.concatenate([np.ones((N, 1)), np.maximum(u[:, None] - brk[None, :], 0.0)], axis=1)

    def fit_curve(target):
        # target: [N] values to reproduce with sum_j coef_j phi_j(u)
        G = Phi.T @ Phi + 1e-8 * np.eye(Phi.shape[1])
        return np.linalg.solve(G, Phi.T @ target)

    # empirical per-example targets: g(t) and k(t) (constant part only)
    g_target = np.array([g_tab[ti] for ti in t])
    k_target = np.array([k_tab[ti] for ti in t])

    cg = fit_curve(g_target)
    ck = fit_curve(k_target)

    # ---- build the network ----------------------------------------
    # Channel plan for conv1 (32 filters):
    #   ch 0      : +x  (identity tap on x, offset M)
    #   ch 1      : -x  (identity tap on x, negated, offset M)
    #   ch 2..K+1 : hinge features relu(u - b_j)  (timestep basis)
    #   remaining  : constant channels (offset M) -> unused by out
    # conv2 / conv3 : identity pass-through (centre tap = 1) so that the
    #                 conv3 output equals the conv1 output (ReLU-safe).
    # out : combines channels 0,1 with (1/2, -1/2) -> x, and the hinge
    #       channels with the fitted coefficients -> gain(t)*x + offset(t).

    M = 8.0                      # offset keeping every conv1 output > 0
    n_ch1 = 32
    n_hinge = K

    c1w = np.zeros((n_ch1, 2, 3, 3))
    c1b = np.full(n_ch1, M)

    c1w[0, 0, 1, 1] = 1.0        # +x
    c1w[1, 0, 1, 1] = -1.0       # -x
    # hinge channels: weight on the t-channel (broadcast constant field).
    # A 3x3 conv over a constant field with all-equal taps gives 9*w*u.
    for j in range(n_hinge):
        c1w[2 + j, 1, :, :] = 1.0 / 9.0   # -> contributes u
        c1b[2 + j] = 0.0                  # relu(u - b_j) needs the shift
    # implement relu(u - b_j) as relu( (u - b_j) ) by using a bias of -b_j
    for j in range(n_hinge):
        c1b[2 + j] = -float(brk[j])
    # remaining channels: constant M (already), unused.

    # conv2, conv3 : identity on the channels we need (0,1 and hinges)
    c2w = np.zeros((64, 32, 3, 3))
    c2b = np.zeros(64)
    c3w = np.zeros((64, 64, 3, 3))
    c3b = np.zeros(64)
    for ch in range(2 + n_hinge):
        c2w[ch, ch, 1, 1] = 1.0
        c3w[ch, ch, 1, 1] = 1.0

    # ---- out layer -------------------------------------------------
    ow = np.zeros((1, 64, 3, 3))
    ob = np.zeros(1)

    # x reconstruction: 0.5*(ch0 - ch1)  -> x
    ow[0, 0, 1, 1] = 0.5
    ow[0, 1, 1, 1] = -0.5

    # gain(t)*x :  coefficient cg[j] on hinge channel j, times x.
    #   hinge channel j holds relu(u-b_j) (a constant field), so we need
    #   the product with x -> realised by the *pair* trick below.
    #   Instead of a product we use the exact structure:
    #     out = g(u)*x + k(u)
    #   g(u) is realised by scaling the x channels.  Since the x channels
    #   are (x+M) and (-x+M), a per-example gain g(u) multiplying them
    #   would be a product; we therefore realise the gain through the
    #   hinge channels acting on a *copy* of x that is scaled by the
    #   hinge value.  This is impossible linearly, so instead we use the
    #   exact identity:  g(u)*x = sum_j dg_j * relu(u-b_j) * x
    #   approximated by  sum_j dg_j * (x - x*0) ... -> fall back to a
    #   *t-independent* gain plus hinge-dependent *offsets*, and let the
    #   hinge channels also carry a spatially constant copy of x scaled
    #   by the hinge value through the conv1 t-channel mixing:
    #   conv1 hinge channel j = relu(u - b_j) (constant field).
    #   To obtain relu(u-b_j)*x we add, in conv1, a channel that is
    #   x + (u-b_j) : not linear -> impossible.
    #   => Use the "affine-in-x, per-t" trick: build the gain from the
    #      hinge channels by *replacing* the x channels with hinge-scaled
    #      versions is impossible, so we approximate the gain function by
    #      a *single* global linear map in u and let the hinge channels
    #      carry the (t-dependent) *offset* field only.
    #      The residual gain error is absorbed by a global scale factor.

    # --- Global affine gain model:  g(u) ~ a0 + a1*u  (least squares)
    A = np.stack([np.ones(N), u], axis=1)
    solg, *_ = np.linalg.lstsq(A, g_target, rcond=None)
    a0, a1 = float(solg[0]), float(solg[1])

    # x channels carry (x + M) and (-x + M); we want gain*(x).
    # gain*(x+M) = gain*x + gain*M ; gain*M is a per-example constant that
    # we add back via the hinge channels (offset fit).  So:
    #   out = gain*(x+M) - gain*M + k(u)
    # The constant gain*M is a per-example constant -> representable by
    # the hinge basis (constant + hinges).  Fit it with the same basis.
    const_target = (a0 + a1 * u) * M
    cconst = fit_curve(const_target)

    # out = (a0 + a1*u) * (x + M)  +  [k(u) - (a0+a1*u)*M]
    #     = gain*(x+M) - gain*M + k(u)
    # The x channel (ch0) holds (x+M); the -x channel (ch1) holds (-x+M).
    # gain*(x+M) cannot be formed from (x+M) with a t-dependent scalar.
    # => final decision: use a t-INDEPENDENT gain on x and put the entire
    #    t-dependence into the offset field (hinge channels), which is
    #    exact for the offset part and approximate for the gain part.
    #    The gain part is handled by rescaling the *hinge* channels that
    #    carry x*M-like terms.

    # Simplest robust choice: implement
    #   out = gbar * x + offset(u)
    # with gbar the best single global gain, and offset(u) fitted to the
    # residual  k(t) - (g(t)-gbar)*x_mean_behaviour.  Because the optimal
    # gain varies strongly with t, we instead implement the gain exactly
    # by *replicating* the x channel for every hinge level and letting the
    # hinge channel provide the *level* while the x channel provides the
    # sign -- this is a product again, so we accept the following exact
    # construction instead:
    #
    #   conv1 channel 0 : x + M
    #   conv1 channel 1 : -x + M
    #   conv1 channel 2+j : relu(u - b_j) * 0  (constant field, hinge)
    #   out: 0.5*(ch0 - ch1) -> x   with FIXED gain 1
    #        plus hinge channels -> t-dependent offset
    #   => out = x + offset(u)   (gain 1)
    # The optimal predictor needs gain(t) ~ c(t) in [0,1]; a gain of 1
    # over-predicts at small t.  We therefore scale the x channels by a
    # global factor gbar and add the hinge offset, and we additionally
    # exploit that the *hinge* channels can carry a per-t multiple of the
    # *mean image* (a constant field) which is the dominant offset term.

    gbar = float(np.mean(g_target))
    # residual offset to add:  k(t) - (g(t)-gbar)*E[x]  with E[x]=sa*m
    ex_mean = (sa * m.reshape(-1))[None, :].mean(axis=1)   # [N]
    off_target = k_target - (g_target - gbar) * ex_mean
    c_off = fit_curve(off_target)

    ow[0, 0, 1, 1] = gbar
    ow[0, 1, 1, 1] = -gbar
    for j in range(n_hinge):
        ow[0, 2 + j, 1, 1] = c_off[1 + j]
    ob[0] = c_off[0]

    # -----------------------------------------------------------------
    # The above "gain 1 + offset" family is clearly suboptimal because
    # the optimal gain varies with t.  We now implement the t-dependent
    # gain EXACTLY using a legitimate construction:
    #   conv1 channel 0 : x + M
    #   conv1 channel 1 : -x + M
    #   conv1 channel 2+j : relu(u - b_j)  (constant field, per-example)
    #   conv2 channel 2+j : 0.5*(ch0 - ch1) * ...  -> no product.
    # The only way to multiply a constant field by a varying field with
    # linear layers is impossible, BUT we can multiply the *hinge* by the
    # *constant* part of x (the mean image) and handle the varying part
    # of x with a t-dependent *scale* applied to the x channels via a
    # per-example additive term that is exactly zero-mean:
    #   gain(t)*x = gbar*x + (g(t)-gbar)*x
    # and (g(t)-gbar)*x is a product of a per-example scalar and the
    # image.  We approximate it by (g(t)-gbar) * (x - mean) which is
    # zero-mean-ish, plus (g(t)-gbar)*mean handled exactly by the hinge
    # channels (constant field).  The approximation error is
    #   (g(t)-gbar)*(x - E[x]) which is NOT zero -> it is a genuine
    #   product term.  We therefore instead use the *exact* identity:
    #   gain(t)*x  =  gain(t)*(x - m_x) + gain(t)*m_x
    # with m_x the per-example spatial mean of x (a per-example scalar).
    # gain(t)*(x - m_x) is still a product.
    #
    # FINAL: we accept a per-example *global* rescaling implemented by
    # folding the gain into the hinge basis via a *spatially uniform*
    # term only, and we additionally exploit a second, independent
    # mechanism: the timestep channel can be mixed with x in conv1 with
    # a *t-dependent threshold* to produce a piecewise-linear-in-x
    # family whose slope can vary with t.  Concretely, we build
    #   conv1 channel 2+j : relu( x + s_j(u) )  with s_j(u) = -(b_j) - M
    # i.e. relu(x - b_j) style thresholds that move with t.
    # The out layer then forms sum_j w_j relu(x - tau_j(t)) which is a
    # piecewise-linear function of x whose *slope* is selected by the
    # position of x relative to the moving knots.  For x ~ N(0,1) the
    # relevant slope is w_j for x in the j-th segment, which is not a
    # clean gain.  => discarded.
    #
    # => We settle for the best *achievable* closed-form solution of the
    #    form  out = gbar*x + offset(u) + (per-t affine correction of the
    #    x-channel magnitude implemented by scaling the x channels with a
    #    t-dependent factor through the *bias* of the hinge channels is
    #    impossible) ...
    #
    # Instead of fighting this, we use a DIFFERENT exact route:
    #   the network input contains x_t; we can make conv1 compute
    #   relu( x_t + M ) and relu( -x_t + M ) and ALSO make conv1 compute
    #   relu( x_t + M - tau_j(u) ) for a set of t-dependent thresholds.
    #   The *derivative* trick: for a fixed t, the family
    #     sum_j w_j relu(x - tau_j(t))
    #   has slope sum_{j: tau_j(t) < x} w_j.  If we choose the knots
    #   tau_j(t) = q_j * u (all scaled by u) with q_j spanning the range
    #   of x, then for a given x the active set changes with u, giving a
    #   t-dependent effective gain.  With K knots we obtain a piecewise
    #   constant approximation of gain(u) * x.  This IS implementable
    #   and exact (no products): conv1 channel 2+j computes
    #     relu( x + M - q_j*u )  -> needs x and u in the SAME channel
    #     with a *product* q_j*u ... but q_j is a CONSTANT, so q_j*u is
    #     a constant field (linear in u) -> LINEAR in the input!
    #   YES: q_j*u is linear in the t-channel, so
    #     relu( x + M - q_j*u )
    #   is a legal conv1 output (weights: 1 on x centre tap, -q_j/9 on
    #   all t taps, bias M).  This gives a t-dependent *threshold* on x.
    #
    #   With thresholds tau_j(t) = q_j * u  and out weights w_j, the
    #   output is  sum_j w_j relu(x - q_j u)  which is a piecewise-linear
    #   function of x with knots moving with t.  We fit w_j by least
    #   squares to the target field  c(t)*x  over the training data.
    #   This gives a genuine t-dependent gain (approximated in x).
    # -----------------------------------------------------------------

    # ---- exact-ish t-dependent gain via moving thresholds -----------
    # Build a richer basis: for a grid of slopes q, the channel value is
    # relu(x + M - q*u).  Fit out weights (per q) by least squares on
    # the training pairs (x, target c(t)*x + d(t)).
    n_q = 24
    qs = np.linspace(-3.0, 3.0, n_q)
    # channel layout: 0:+x, 1:-x, 2..2+n_q-1 : relu(x + M - q*u)
    #                then hinge channels for the constant offset
    n_hinge2 = 12
    brk2 = np.linspace(0.0, 1.0, n_hinge2 + 1)[1:-1]
    n_ch1 = 2 + n_q + n_hinge2
    if n_ch1 > 32:
        n_q = 32 - 2 - n_hinge2
        qs = np.linspace(-3.0, 3.0, n_q)
        n_ch1 = 2 + n_q + n_hinge2

    c1w = np.zeros((32, 2, 3, 3))
    c1b = np.full(32, M)
    c1w[0, 0, 1, 1] = 1.0
    c1w[1, 0, 1, 1] = -1.0
    for j in range(n_q):
        ch = 2 + j
        c1w[ch, 0, 1, 1] = 1.0
        c1w[ch, 1, :, :] = -float(qs[j]) / 9.0
        c1b[ch] = M
    for j in range(n_hinge2):
        ch = 2 + n_q + j
        c1w[ch, 1, :, :] = 1.0 / 9.0
        c1b[ch] = -float(brk2[j])

    # identity pass-through conv2 / conv3 for the used channels
    used = list(range(2 + n_q + n_hinge2))
    c2w = np.zeros((64, 32, 3, 3))
    c2b = np.zeros(64)
    c3w = np.zeros((64, 64, 3, 3))
    c3b = np.zeros(64)
    for ch in used:
        c2w[ch, ch, 1, 1] = 1.0
        c3w[ch, ch, 1, 1] = 1.0

    # ---- design matrix for the out-layer least squares --------------
    # channel 0 -> x + M ; channel 1 -> -x + M
    # channel 2+j -> relu(x + M - q_j*u)
    # channel 2+n_q+j -> relu(u - brk2_j)
    # target: c(t)*x + d(t)*m_field  (per pixel)
    # We fit out.weight (centre taps) and out.bias.
    # Build design matrix rows = (example, pixel), cols = channels+1
    # To keep memory bounded, accumulate normal equations.
    ncol = 1 + len(used)
    A = np.zeros((ncol, ncol))
    bvec = np.zeros(ncol)

    # chunk over examples
    CH = 512
    for s in range(0, N, CH):
        e = min(N, s + CH)
        xs = x[s:e].reshape(e - s, -1)                       # [n,256]
        us = u[s:e]
        ct = c_t[s:e][:, None]
        dt = d_t[s:e][:, None]
        tgt = ct * xs + dt * m.reshape(-1)[None, :]          # [n,256]
        n = e - s
        cols = np.zeros((n, 256, ncol))
        cols[:, :, 0] = (xs + M)
        cols[:, :, 1] = (-xs + M)
        for j in range(n_q):
            cols[:, :, 2 + j] = np.maximum(xs + M - qs[j] * us[:, None], 0.0)
        for j in range(n_hinge2):
            cols[:, :, 2 + n_q + j] = np.maximum(us[:, None] - brk2[j], 0.0)
        # normal equations
        F = cols.reshape(n * 256, ncol)
        A += F.T @ F
        bvec += F.T @ tgt.reshape(-1)
        del cols, F

    A += 1e-6 * np.eye(ncol)
    sol = np.linalg.solve(A, bvec)

    ow = np.zeros((1, 64, 3, 3))
    ob = np.zeros(1)
    # channel 0/1 are the +x/-x channels (their combination gives x)
    ow[0, 0, 1, 1] = 0.5 * sol[0]
    ow[0, 1, 1, 1] = -0.5 * sol[1]
    for j in range(n_q):
        ow[0, 2 + j, 1, 1] = sol[2 + j]
    for j in range(n_hinge2):
        ow[0, 2 + n_q + j, 1, 1] = sol[2 + n_q + j]
    ob[0] = sol[0] * 0.0  # constant handled through channels

    # The constant offset (bias) is not identifiable separately from the
    # M-offset inside the channels; we remove the M contribution by
    # subtracting the fitted constant part.  Because every channel
    # carries an offset M, the total constant = M*(sum of weights).
    # We therefore set out.bias so that the *net* constant equals the
    # desired one: desired constant field = d(t)*m  (already in target
    # through the hinge channels? no) -> recompute explicitly below.

    # --- explicit forward evaluation to calibrate the bias ----------
    def forward(xb, ub):
        # conv1
        inp = np.concatenate([xb, ub[:, None, None, None].repeat(H * Wd, 1).repeat(Wd, 2)], axis=1)
        # build padded input
        pad = np.zeros((xb.shape[0], 2, H + 2, Wd + 2))
        pad[:, 0, 1:-1, 1:-1] = xb[:, 0]
        pad[:, 1, 1:-1, 1:-1] = ub[:, None, None].repeat(H, 1).repeat(Wd, 2)
        o1 = np.zeros((xb.shape[0], 32, H, Wd))
        for f in range(32):
            acc = np.zeros((xb.shape[0], H, Wd))
            for i in range(2):
                for kr in range(3):
                    for kc in range(3):
                        acc += c1w[f, i, kr, kc] * pad[:, i, kr:kr + H, kc:kc + Wd]
            o1[:, f] = acc + c1b[f]
        o1 = np.maximum(o1, 0.0)
        # conv2 / conv3 (identity taps only -> trivial)
        o2 = np.zeros_like(o1)
        o3 = np.zeros_like(o1)
        for f in range(32):
            o2[:, f] = o1[:, f] + c2b[f]
        o2 = np.maximum(o2, 0.0)
        for f in range(64):
            o3[:, f] = o2[:, f] + c3b[f]
        o3 = np.maximum(o3, 0.0)
        # out (centre taps only)
        out = np.zeros((xb.shape[0], 1, H, Wd))
        for f in range(64):
            out[:, 0] += ow[0, f, 1, 1] * o3[:, f]
        out += ob[0]
        return out

    # calibrate out.bias by least squares on the residual constant
    sub = np.arange(0, N, max(1, N // 256))
    pred = forward(x[sub], u[sub])
    tgt_sub = (c_t[sub][:, None, None, None] * x[sub]
               + d_t[sub][:, None, None, None] * m.reshape(1, 1, H, Wd))
    ob[0] = float(np.mean(tgt_sub - pred))
    pred = forward(x[sub], u[sub]) + ob[0]
    mse_sub = float(np.mean((pred - tgt_sub) ** 2))

    # --- assemble output weights -----------------------------------
    out_w = {}
    out_w["conv1.weight"] = c1w.astype(np.float32)
    out_w["conv1.bias"] = c1b.astype(np.float32)
    out_w["conv2.weight"] = c2w.astype(np.float32)
    out_w["conv2.bias"] = c2b.astype(np.float32)
    out_w["conv3.weight"] = c3w.astype(np.float32)
    out_w["conv3.bias"] = c3b.astype(np.float32)
    out_w["out.weight"] = ow.astype(np.float32)
    out_w["out.bias"] = np.array([ob[0]], dtype=np.float32)

    # safety: clip, finite, preserve shapes
    for k in W.keys():
        if k not in out_w:
            out_w[k] = W[k].astype(np.float32)
        else:
            out_w[k] = np.asarray(out_w[k], dtype=np.float32)
        out_w[k] = np.nan_to_num(out_w[k], nan=0.0, posinf=0.0, neginf=0.0)
        out_w[k] = np.clip(out_w[k], -1000.0, 1000.0)
        assert out_w[k].shape == W[k].shape, (k, out_w[k].shape, W[k].shape)

    diagnostics = {
        "identity_resid_mse": round(resid, 6),
        "sigma2_clean": round(sig2, 6),
        "mean_img_abs": round(float(np.abs(m).mean()), 6),
        "c_t_min": round(float(c_t.min()), 5),
        "c_t_max": round(float(c_t.max()), 5),
        "n_q": int(n_q),
        "mse_vs_analytic_target": round(mse_sub, 6),
        "target_theoretical_mse": round(float(np.mean(
            (abar[np.clip(t, 0, len(abar) - 1)] * sig2) /
            np.clip(abar[np.clip(t, 0, len(abar) - 1)] * sig2 + 1 - abar[np.clip(t, 0, len(abar) - 1)], 1e-8, None))), 6),
    }
    return {"weights": out_w, "diagnostics": diagnostics}
