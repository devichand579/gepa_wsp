import numpy as np

def edit(w, f):
    # ------------------------------------------------------------------
    # 0. basic data
    # ------------------------------------------------------------------
    x   = np.asarray(f["x"], dtype=np.float64)          # [N,1,16,16]
    t   = np.asarray(f["t"], dtype=np.int64)             # [N]
    y   = np.asarray(f["y"], dtype=np.float64)          # [N,1,16,16] target eps
    cl  = np.asarray(f["clean"], dtype=np.float64)      # [N,1,16,16]
    ab  = np.asarray(f["alphabar"], dtype=np.float64)    # [32]
    N   = x.shape[0]
    H   = W_ = 16

    # ------------------------------------------------------------------
    # 1. recover the true epsilon implied by the schedule
    #    x = sqrt(ab)*clean + sqrt(1-ab)*eps   ->  eps = (x - sqrt(ab)*clean)/sqrt(1-ab)
    # ------------------------------------------------------------------
    a_t   = ab[t]                                     # [N]
    s_ab  = np.sqrt(a_t)
    s_1ma = np.sqrt(np.maximum(1.0 - a_t, 1e-12))
    eps_hat = (x - s_ab[:, None, None, None] * cl) / s_1ma[:, None, None, None]
    # consistency check against the supplied target
    cons = float(np.mean((eps_hat - y) ** 2))

    # ------------------------------------------------------------------
    # 2. closed-form per-timestep optimal *scalar* gain on x_t
    #    eps_hat = g(t) * (x_t - c(t))   with c(t) = E[x_t] per timestep
    #    -> g(t) = Cov(x,eps)/Var(x)  (per timestep, per pixel)
    # ------------------------------------------------------------------
    xf = x.reshape(N, -1)
    ef = eps_hat.reshape(N, -1)
    g_tab = np.zeros(32); c_tab = np.zeros(32); mse_tab = np.zeros(32)
    for k in range(32):
        m = (t == k)
        if m.sum() == 0:
            continue
        xk = xf[m]; ek = ef[m]
        c = xk.mean()
        xc = xk - c
        v = float((xc * xc).mean())
        cov = float((xc * (ek - ek.mean(0))).mean())
        g = cov / v if v > 1e-12 else 0.0
        g_tab[k] = g; c_tab[k] = c
        r = g * xc - (ek - ek.mean(0))
        mse_tab[k] = float((r * r).mean())

    # smooth the gain over t (monotone, low-variance) with a local average
    g_s = g_tab.copy()
    for k in range(32):
        lo, hi = max(0, k - 2), min(32, k + 3)
        g_s[k] = g_tab[lo:hi].mean()
    # enforce monotone non-increasing in t (more noise -> larger gain)
    for k in range(1, 32):
        if g_s[k] > g_s[k - 1]:
            g_s[k] = g_s[k - 1]
    g_s = np.clip(g_s, 0.0, 1.2)

    # ------------------------------------------------------------------
    # 3. build a *linear-in-x* feature path so the network can output
    #    g(t) * (x_t - c(t)) exactly.
    #
    #    conv1 : 32 channels, each = relu( a_j * x_p + b_j(t) )  (identity regime)
    #    conv2 : 64 channels, each = relu( sum_j A_jc * F1_j + bias )  (identity regime)
    #    conv3 : 64 channels, each = relu( sum_c B_cc * F2_c + bias )  (identity regime)
    #    out   : 1x1-ish linear read-out  ->  g(t)*(x - c(t))
    #
    #    The t-dependence is injected as a spatially constant offset in conv1
    #    (the t input channel is broadcast, so any 3x3 kernel on it yields a
    #    constant offset).  Offsets are cancelled exactly by bias bookkeeping.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(0)

    # ---- conv1: encode x_t locally with several "tap groups" -------------
    # channel layout: 32 = 4 groups x 8 channels
    # group g uses a distinct 3x3 spatial kernel K_g on x_t (so that the
    # 3x3 read-out of conv3 can realise a 7x7 effective filter).
    K = np.zeros((4, 3, 3))
    K[0] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])           # centre
    K[1] = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])           # left
    K[2] = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])           # right
    K[3] = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])           # down
    # small random perturbations -> richer local basis
    K = K + 0.05 * rng.standard_normal((4, 3, 3))

    # per-channel gain on x (so that conv2 can form several scaled copies)
    ch_gain = np.ones(32)
    for j in range(8):
        ch_gain[j]     = 1.0
        ch_gain[8 + j] = 0.5
        ch_gain[16 + j] = 0.25
        ch_gain[24 + j] = 0.125

    # ---- time offsets ----------------------------------------------------
    # conv1 t-channel kernel: sum of entries S_j  -> offset slope = S_j * (t/31)
    # we want the *final* output offset to be  g(t)*c(t)  (to subtract c(t)).
    # We realise the time dependence of the gain through a set of "gain
    # channels" whose contribution is switched on/off by the t offset.
    #
    # Mechanism: conv1 channel j produces  F1_j = relu( a_j*x_p + d_j(t) )
    # where d_j(t) = s_j*(t/31) + e_j.  A channel with large negative d_j(t)
    # is switched OFF (output 0) for that t, and a channel with large positive
    # d_j(t) is ON.  Combining ON channels with different a_j gives the
    # desired t-dependent effective gain on x.
    #
    # We realise g(t) as a piecewise-constant function of t with 32 steps
    # using 32 channels: channel j is ON only for t >= j_threshold.
    # Simpler and smoother: use the *timestep index itself* through the
    # offset: choose d_j(t) = A_j * (t/31 - u_j) + BIG, so channel j is ON
    # for t above u_j and OFF below (via relu of a decreasing ramp).
    # To keep the network output exactly g(t)*x we instead use the following
    # equivalent construction (no switching needed):
    #
    #   Let the 32 conv1 channels be  x-local features scaled by a_j, plus
    #   t-only ramp features  R_j(t) = relu(BIG - A_j*(t/31 - u_j)).
    #   conv2/conv3 combine them so that the final read-out equals
    #        out = g(t) * x_p  -  g(t)*c(t)
    #   where g(t) is realised as a linear combination of the ramps
    #   (piecewise-linear in t, exactly the fitted g table interpolated).
    # ------------------------------------------------------------------
    # Ramp construction: R_j(t) = relu( -(t/31 - u_j) ) * scale, i.e. a
    # decreasing hinge.  With u_j = j/31 this gives a piecewise-linear
    # basis for any function of t on [0,31].
    n_ramp = 30
    u = np.arange(n_ramp) / 31.0                     # knots
    # conv1 channels 0..31 : 8 centre-ish local x features (4 groups x 2)
    #                       + 24 ramp features? -> we need 32 total.
    # Layout: channels 0..7   : local x features (group g = j//2, copy j%2)
    #         channels 8..31  : ramp features (24 of them)
    n_local = 8
    n_ramp_total = 32 - n_local

    w1 = np.zeros((32, 2, 3, 3))
    b1 = np.zeros(32)

    # local x channels
    for j in range(n_local):
        gidx = j // 2
        w1[j, 0] = ch_gain[j] * K[gidx]
        # t-channel: constant spatial kernel (all-ones/9) so the offset is
        # spatially uniform; slope chosen later (0 for now, set below)
        w1[j, 1] = 0.0
        b1[j] = 0.0

    # ramp channels: pre-activation = -(t/31) + u_j  (times a scale), plus a
    # large constant so that relu is a clean hinge:  relu( M - (t/31 - u_j) )
    M = 4.0
    ramp_u = np.linspace(0.0, 1.0, n_ramp_total)
    for j in range(n_ramp_total):
        w1[n_local + j, 0] = 0.0
        w1[n_local + j, 1] = -1.0 / 9.0          # sum over 3x3 -> -(t/31)
        b1[n_local + j] = M + ramp_u[j]

    # ------------------------------------------------------------------
    # 4. conv2 / conv3 : route local-x channels and ramp channels to the
    #    read-out, keeping everything in the ReLU identity regime.
    # ------------------------------------------------------------------
    # conv2: 64 channels.  We need conv3 to be able to read out:
    #   (a) the local x features (with the desired spatial filter), and
    #   (b) the ramp features (to form g(t)).
    # Strategy: conv2 copies channels (identity-ish) with a large positive
    # bias so ReLU is identity; conv3 does the same; out reads them.
    BIG = 50.0

    w2 = np.zeros((64, 32, 3, 3)); b2 = np.zeros(64)
    w3 = np.zeros((64, 64, 3, 3)); b3 = np.zeros(64)
    wo = np.zeros((1, 64, 3, 3));  bo = np.zeros(1)

    # conv2 channel c (c<32) copies conv1 channel c with a 3x3 kernel that
    # realises the desired spatial filter for that feature group.
    # desired effective filter (7x7) built from the 4 group kernels:
    #   centre group K0 -> identity
    #   left/right/down groups -> shifted copies
    # We implement the target filter as a linear combination of the group
    # features; conv2 applies a 3x3 kernel to conv1 outputs, conv3 applies
    # another 3x3 -> total 7x7 support.
    #
    # Simplify: use conv2 to select group g (kernel = K_g) and conv3 to
    # select the required 3x3 shift of the 7x7 window.
    # conv2 channel for (group g, copy c) uses kernel K_g on conv1 channels
    # of that group.
    #
    # Build the desired 7x7 linear filter F7 (target: identity + small
    # smoothing) and express it in the group basis.
    # group kernels act on x; conv3 shifts by 3x3 -> we can realise shifts
    # of +-1,+-2 in x space.
    # Desired: out_pix = sum_{q in 7x7} F7[q] * x[pix+q]
    # With groups: g0 = centre tap (offset 0), g1 = left (offset -1 in col),
    # g2 = right (+1 col), g3 = down (+1 row).  conv3 can shift each group
    # feature by up to +-1 pixel, giving offsets in {-2..0} x {0,1,2} etc.
    # We therefore approximate F7 with the available offset set.
    # Use a simple, robust choice: F7 = identity (only centre tap) plus a
    # small amount of neighbour averaging to denoise.
    # offsets available (row,col) in units of pixels, from group+shift:
    #   g0 (centre) shifted by (dr,dc) in {-1,0,1}^2 -> offsets in [-1,1]^2
    #   g1 (left)  -> col offset -1 + shift -> [-2,0]
    #   g2 (right) -> col offset +1 + shift -> [0,2]
    #   g3 (down)  -> row offset +1 + shift -> [0,2]
    # So we can realise any filter supported on rows {0,1,2} (down group and
    # centre) and cols {-2..2}.  Good enough: build F7 with support
    # rows {0,1,2}, cols {-2..2}.
    # ------------------------------------------------------------------
    # target filter: mild 5x5 binomial-ish smoothing around centre
    # (row,col) offsets relative to output pixel: dr in {0,1,2}, dc in {-2..2}
    # We use: F[0,0]=1.0 (centre), plus small weights on neighbours.
    # Represent F7 as: sum over groups g and conv3 shifts (dr3,dc3).
    # group g has base offset o_g = (0,0),(0,-1),(0,+1),(+1,0)
    # conv3 shift s in {-1,0,1}^2 -> total offset o_g + s.
    # Build coefficient map over offsets: coef[(dr,dc)].
    coef = {}
    def addcoef(dr, dc, val):
        coef[(dr, dc)] = coef.get((dr, dc), 0.0) + val
    # identity-ish: centre tap
    addcoef(0, 0, 1.0)
    # small neighbour smoothing (helps denoise, keeps variance low)
    for (dr, dc), val in [(0, 1), (0, -1), (1, 0), (0, 2), (0, -2),
                          (1, 1), (1, -1), (2, 0)]:
        addcoef(dr, dc, 0.06)
    # decompose coef into group+shift pairs
    # group bases: g0 (0,0), g1 (0,-1), g2 (0,1), g3 (1,0)
    bases = {0: (0, 0), 1: (0, -1), 2: (0, 1), 3: (1, 0)}
    # assign each requested offset to the best group (exact match preferred)
    plan = {}   # (group, shift) -> coefficient
    for (dr, dc), val in coef.items():
        best = None
        for g, (br, bc) in bases.items():
            sdr, sdc = dr - br, dc - bc
            if -1 <= sdr <= 1 and -1 <= sdc <= 1:
                if best is None or abs(sdr) + abs(sdc) < best[0]:
                    best = (abs(sdr) + abs(sdc), g, (sdr, sdc))
        if best is not None:
            _, g, s = best
            plan[(g, s)] = plan.get((g, s), 0.0) + val
        else:
            # unreachable offsets -> drop (tiny effect)
            pass

    # conv2: for each group g, create 2 channels (to have enough capacity)
    # that apply kernel K_g to the conv1 channels of that group.
    # conv1 group g channels are indices {g*2, g*2+1} (local channels).
    group_ch = {g: [g * 2, g * 2 + 1] for g in range(4)}
    # conv2 channels 0..7 : group copies
    for g in range(4):
        for c in range(2):
            ch = g * 2 + c
            for src in group_ch[g]:
                w2[ch, src] = K[g]
            b2[ch] = BIG
    # conv2 channels 8..31 : ramp copies (identity on ramp channels)
    for j in range(n_ramp_total):
        ch = n_local + j
        w2[ch, n_local + j] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        b2[ch] = BIG

    # conv3: channels 0..7 : apply the planned 3x3 shifts to conv2 group
    # channels; channels 8..31 : copy ramp channels (identity).
    # conv3 channel index -> (group, shift)
    plan_items = list(plan.items())
    # we need one conv3 channel per (group,shift) pair; <= 8 channels
    conv3_map = []
    for (g, s), val in plan_items:
        conv3_map.append((g, s, val))
    # conv3 channels 0..len-1 for spatial part
    for idx, (g, s, val) in enumerate(conv3_map):
        ch = idx
        src = g * 2            # use first copy of the group
        w3[ch, src] = np.roll(np.roll(np.eye(3), s[0], axis=0), s[1], axis=1)
        b3[ch] = BIG
    # conv3 channels for ramps: index offset
    ramp_base = len(conv3_map)
    for j in range(n_ramp_total):
        ch = ramp_base + j
        w3[ch, n_local + j] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        b3[ch] = BIG
    n3 = ramp_base + n_ramp_total
    assert n3 <= 64, n3

    # ------------------------------------------------------------------
    # 5. read-out: out = sum over conv3 channels of W * F3 + b
    #    spatial part: combine conv3 spatial channels with the plan values
    #    ramp part:    combine ramp channels with coefficients so that the
    #                  time-dependent gain g(t) is produced.
    # ------------------------------------------------------------------
    # Spatial: for each conv3 spatial channel idx with plan value val, the
    # conv3 output equals (approximately) BIG + (x filtered by the group
    # kernel and shifted).  The read-out weight must cancel the BIG offset:
    # we handle offsets globally at the end (all conv3 outputs carry a
    # constant offset; the read-out bias cancels the total).
    # Because the ramp channels also carry BIG offsets and the ramp values
    # themselves are functions of t, we must be careful: we want the final
    # output to be  g(t)*(x - c(t))  with NO extra t-dependent constant.
    #
    # Construction of the ramp part:
    #   conv1 ramp channel j  -> R_j = relu(M - (t/31) + u_j)  (a hinge)
    #   conv2/conv3 copy it  -> F3_j = BIG + R_j
    #   read-out weight a_j on F3_j contributes a_j*(BIG + R_j)
    # We want the total t-dependent part of the output to be  g(t)*x  only,
    # i.e. no additive t-dependent constant.  Therefore the coefficients
    # a_j must satisfy  sum_j a_j * R_j(t) = 0 for all t, while the
    # *spatial* channels must carry the t-dependent gain on x.
    #
    # => The gain must be applied to the SPATIAL channels, not added.
    # => We need the spatial channel values themselves to be t-dependent
    #    in a multiplicative way.  Achieve this with a second set of spatial
    #    channels that are *gated* by the ramp:  G = relu(x_p + BIG - BIG*...)
    # Simplest exact construction:
    #    use two spatial channels per group:
    #      S  = relu( x_p + BIG )            -> x_p + BIG   (always on)
    #      S' = relu( x_p + BIG - K(t) )     -> x_p + BIG - K(t)  when on
    #    where K(t) is a t-dependent constant that is large at low t
    #    (so S' is OFF, contributing 0) and 0 at high t (S' on).
    #    Then out = w1*S + w2*S' gives gain (w1 + w2) at high t and w1 at
    #    low t (plus offsets).  With several thresholds we obtain a
    #    piecewise-constant gain in t.
    #    The offsets: S' = x_p + BIG - K(t) when on, 0 when off.  The
    #    read-out bias can cancel only a constant, so the t-dependent -K(t)
    #    must be cancelled by pairing with a ramp channel carrying +K(t).
    #    We therefore also add ramp channels that reproduce K(t) exactly
    #    (K is a hinge of t, so a ramp channel with the same hinge works).
    # ------------------------------------------------------------------
    # Rebuild conv1 with the gated spatial channels.
    #
    # Final conv1 layout (32 channels):
    #   0..7   : "always-on" spatial channels (group g = j//2, copy j%2)
    #   8..15  : "gated" spatial channels (group g = (j-8)//2, copy)
    #            with gate threshold u_j (t-dependent offset)
    #   16..31 : ramp channels (hinges in t) used to cancel offsets and to
    #            realise the piecewise-linear gain
    # ------------------------------------------------------------------
    w1 = np.zeros((32, 2, 3, 3)); b1 = np.zeros(32)
    n_always = 8
    n_gated = 8
    n_ramp_total = 32 - n_always - n_gated

    for j in range(n_always):
        gidx = j // 2
        w1[j, 0] = K[gidx]
        b1[j] = 0.0
    # gated channels: pre-act = x_p + BIG - K(t)
    # K(t) = ramp of t: we use  K(t) = A*(t/31 - u)  clipped at 0 by relu?
    # We need: gate OFF for t < u (K large) and ON for t > u (K ~ 0).
    # Use pre-act = x_p + BIG + (t/31 - u)*S  with S large negative:
    #   at t < u: (t/31-u)*S > 0 (since S<0 and t/31-u<0) -> adds positive
    #   -> stays positive -> still ON.  We want OFF at low t.
    # Reverse: pre-act = x_p + BIG - (t/31 - u)*S' with S'>0:
    #   at t<u: -(negative)*S' = positive -> ON (wrong).
    # We want OFF at low t: pre-act must be negative at low t.
    # pre-act = x_p + BIG - C*(u - t/31)  with C>0 large:
    #   at t<u: -C*(u - t/31) < 0 and large negative -> OFF (relu -> 0)
    #   at t>u: +C*(t/31-u) > 0 -> ON, value x_p + BIG + C*(t/31-u)
    # The ON value has a t-dependent offset C*(t/31-u) which grows with t.
    # We cancel it with a ramp channel carrying the same hinge.
    # Ramp channel for the same hinge: relu(M + (t/31 - u)) -> at t<u: M + neg
    #   choose M large so it's positive everywhere: gives M + (t/31-u) (linear
    #   in t, not a hinge).  We need the *hinge* (t/31-u)_+ to cancel
    #   C*(t/31-u)_+ .  relu(t/31 - u) is exactly the hinge!  Use
    #   ramp channel: pre-act = (t/31 - u)  -> relu = (t/31-u)_+.
    #   But then at t<u the ramp is 0 and the gated channel is 0 -> consistent.
    #   So: gated pre-act = x_p + BIG - C*(u - t/31)  -> for t<u: negative
    #   (OFF, relu=0); for t>=u: x_p + BIG + C*(t/31-u)  (ON).
    #   offset to cancel: C*(t/31-u) = C*(t/31-u)_+ + C*(t/31-u)_-
    #   The negative part (t<u) is irrelevant because the gated channel is
    #   OFF there.  So we need to cancel C*(t/31-u)_+ -> use ramp channel
    #   with relu(t/31-u) and weight -C.
    #   But careful: ramp channel value must be exactly (t/31-u)_+; with
    #   relu pre-act = (t/31-u) and t broadcast, conv1 t-channel weight
    #   sum gives -(t/31) etc.  Fine.
    #   However the ramp channel also needs a large offset? No: relu of
    #   (t/31-u) is exactly the hinge (0 at t<u).  But then conv2/conv3
    #   copy it with a BIG bias -> F3 = BIG + hinge; the read-out weight
    #   on it contributes w*(BIG + hinge).  The BIG part is a constant
    #   (cancelled by the global read-out bias) and the hinge part is what
    #   we want.  Good.
    # ------------------------------------------------------------------
    # gate thresholds: we want a piecewise-constant gain g(t) that matches
    # the fitted g table.  Use 8 gates with thresholds spread over t.
    gate_u = np.linspace(0.0, 1.0, n_gated)
    C = 8.0
    for j in range(n_gated):
        ch = n_always + j
        gidx = j // 2
        w1[ch, 0] = K[gidx]
        # t-channel: -(C)*(t/31 - u)  -> sum over 3x3 kernel = -C*(t/31-u)
        w1[ch, 1] = -C / 9.0
        b1[ch] = BIG - C * gate_u[j]

    # ramp channels: 16..31, hinges at knots u_k
    ramp_knots = np.linspace(0.0, 1.0, n_ramp_total)
    for j in range(n_ramp_total):
        ch = n_always + n_gated + j
        w1[ch, 1] = 1.0 / 9.0          # sum -> +(t/31)
        b1[ch] = -ramp_knots[j]        # -> relu(t/31 - u_k)

    # ------------------------------------------------------------------
    # 6. conv2 / conv3 routing (rebuild with new layout)
    # ------------------------------------------------------------------
    w2 = np.zeros((64, 32, 3, 3)); b2 = np.zeros(64)
    w3 = np.zeros((64, 64, 3, 3)); b3 = np.zeros(64)
    wo = np.zeros((1, 64, 3, 3)); bo = np.zeros(1)

    # conv2 channels 0..7 : always-on group copies
    for g in range(4):
        for c in range(2):
            ch = g * 2 + c
            for src in group_ch[g]:
                w2[ch, src] = K[g]
            b2[ch] = BIG
    # conv2 channels 8..15 : gated group copies
    for g in range(4):
        for c in range(2):
            ch = 8 + g * 2 + c
            for src in group_ch[g]:
                w2[ch, src] = K[g]
            b2[ch] = BIG
    # conv2 channels 16..31 : ramp copies
    for j in range(n_ramp_total):
        ch = 16 + j
        w2[ch, 16 + j] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        b2[ch] = BIG
    # conv2 channels 32..63 : unused (kept zero, bias 0 -> relu(0)=0)

    # conv3 channels: spatial routing
    #   conv3 ch 0..7   : apply plan shifts to conv2 always-on channels
    #   conv3 ch 8..15  : apply plan shifts to conv2 gated channels
    #   conv3 ch 16..31 : copy ramp channels from conv2 ch 16..31
    conv3_spatial = []
    for idx, (g, s, val) in enumerate(conv3_map):
        ch = idx
        src = g * 2
        w3[ch, src] = np.roll(np.roll(np.eye(3), s[0], axis=0), s[1], axis=1)
        b3[ch] = BIG
        conv3_spatial.append((ch, val, False))
    for idx, (g, s, val) in enumerate(conv3_map):
        ch = len(conv3_map) + idx
        src = 8 + g * 2
        w3[ch, src] = np.roll(np.roll(np.eye(3), s[0], axis=0), s[1], axis=1)
        b3[ch] = BIG
        conv3_spatial.append((ch, val, True))
    ramp_base = 2 * len(conv3_map)
    for j in range(n_ramp_total):
        ch = ramp_base + j
        w3[ch, 16 + j] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        b3[ch] = BIG

    # ------------------------------------------------------------------
    # 7. read-out weights
    #    out = sum_ch wo[0,ch] * F3_ch + bo
    #    F3_always(ch) = BIG + (x filtered by plan)
    #    F3_gated(ch)  = BIG + (x filtered) + C*(t/31-u)_+   (when gate on)
    #    F3_ramp(ch)   = BIG + (t/31 - u_k)_+
    #    We want: out = g(t) * (x - c(t))
    #    => spatial weights: always-on channels get weight  wA = g_low,
    #       gated channels get weight wG = (g_high - g_low) ... but the
    #       gain must be a function of t, and the gated channels switch on
    #       at their thresholds.
    #    With 8 gates at thresholds u_j, the effective gain is:
    #       g(t) = wA + sum_j wG_j * 1[t >= u_j]  (each gated channel
    #       contributes its own filter value)
    #    We choose the thresholds & weights so that the resulting
    #    piecewise-constant g(t) approximates the fitted g table.
    #    Because each gated channel carries the *same* spatial filter as
    #    its always-on partner (group g), the total spatial filter is
    #    sum over groups of (wA_g + sum of active wG) * plan filter.
    #    To keep it simple we use the SAME plan filter for all groups and
    #    set the always-on weights to realise the base filter.
    # ------------------------------------------------------------------
    # Simplify: use a single spatial filter (identity + smoothing) applied
    # via the centre group only for the always-on part, and the same for
    # gated part.  Then:
    #   out_spatial(t) = A(t) * (F x)  where A(t) = wA + sum_{j: t>=u_j} wG_j
    # and out_offset(t) must equal -A(t)*c(t).
    # The offset contributions: gated channels contribute C*(t/31-u_j)_+ * wG_j,
    # ramp channels can cancel them (choose ramp weights accordingly) and
    # also produce the -A(t)*c(t) term.
    #
    # We therefore need to express the desired offset function
    #       d(t) = -A(t)*c(t) - sum_j wG_j * C*(t/31-u_j)_+
    # as a linear combination of the ramp basis {(t/31-u_k)_+}.
    # The ramp basis (hinges at knots) spans all piecewise-linear functions
    # of t that are 0 at t=0 and continuous.  d(t) is piecewise-linear in t
    # (A(t) piecewise constant, c(t) piecewise constant) -> representable
    # except for the constant term (value at t=0) which the read-out bias
    # can absorb.
    # ------------------------------------------------------------------
    # --- build A(t) from the fitted gain table ---------------------------
    # gates: 8 thresholds u_j = j/7 (in t/31 units) -> gate j turns on at
    # t >= u_j*31.  We want A(t) ~ g(t).  Use a simple construction:
    #   A(t) = g(0) + sum_j (g(u_j) - g(u_{j-1})) * 1[t >= u_j]
    # i.e. stepwise approximation of g with 8 steps.
    # choose thresholds at quantiles of t so the steps are balanced.
    thresh = np.array([0.0, 4, 8, 12, 16, 20, 24, 28])   # in t units
    # A(t) = base + sum_j step_j(t)
    # base = g at t=0 ; step_j = g(thresh_j) - g(thresh_{j-1})
    g_at = np.array([g_s[min(31, int(round(th)))] for th in thresh])
    base = g_at[0]
    steps = np.diff(g_at)                    # 7 increments
    # assign increments to gated channels (8 gated channels available; use
    # the first 7)
    # gated channel j (j=0..7) has threshold gate_u[j]; we want gate_u to
    # match thresh.  Set gate_u[j] = thresh[j]/31 for j=0..7 (gate 0 always
    # on since u=0 -> pre-act = x + BIG + C*t/31 > 0 always).
    gate_u = thresh / 31.0
    # recompute w1/b1 for gated channels with the new thresholds
    for j in range(n_gated):
        ch = n_always + j
        gidx = j // 2
        w1[ch, 0] = K[gidx]
        w1[ch, 1] = -C / 9.0
        b1[ch] = BIG - C * gate_u[j]
    # increments for gates 1..7 (gate 0 is the base, always on)
    # We assign: gate j (j>=1) turns on at thresh[j]; increment = g_at[j]-g_at[j-1]
    # But we have 8 gated channels and only groups 0..3 x 2 copies.
    # We need each gate to carry the SAME spatial filter (the desired F).
    # => use group 0 (centre) for all gated channels, but then conv3 must
    #    route them with the plan shifts.  The plan has multiple (g,s)
    #    entries; we need the gated set to mirror the always-on set.
    # Simplify the spatial filter to a single 3x3 kernel (identity) so the
    # plan has one entry (group 0, shift (0,0)) -> then everything is easy.
    # ------------------------------------------------------------------
    # Rebuild plan with a single entry: identity filter on the centre group.
    conv3_map = [(0, (0, 0), 1.0)]
    # always-on channels: use group 0 copies only (channels 0 and 1)
    # gated channels: use group 0 copies only (channels 8 and 9)
    # We need 8 gated channels -> conv1 has 8 gated slots; all can use the
    # centre kernel K[0] (with tiny random perturbation for diversity).
    for j in range(n_gated):
        ch = n_always + j
        w1[ch, 0] = K[0] + 0.02 * rng.standard_normal((3, 3))
        w1[ch, 1] = -C / 9.0
        b1[ch] = BIG - C * gate_u[j]
    # always-on: channels 0,1 use K[0]; channels 2..7 use other groups but
    # are given zero read-out weight (harmless) -> keep them, they just cost
    # compute.  Actually give them zero weight to avoid offset issues.
    for j in range(n_always):
        gidx = j // 2
        w1[j, 0] = K[gidx]
        b1[j] = 0.0
    # ------------------------------------------------------------------
    # rebuild conv2 / conv3 with the simplified routing
    # ------------------------------------------------------------------
    w2 = np.zeros((64, 32, 3, 3)); b2 = np.zeros(64)
    w3 = np.zeros((64, 64, 3, 3)); b3 = np.zeros(64)
    wo = np.zeros((1, 64, 3, 3)); bo = np.zeros(1)

    # conv2 ch 0 : always-on centre feature (K[0] applied to conv1 ch 0)
    w2[0, 0] = K[0]; b2[0] = BIG
    # conv2 ch 1..7 : always-on other groups (unused downstream)
    for g in range(4):
        for c in range(2):
            ch = g * 2 + c
            if ch == 0:
                continue
            for src in group_ch[g]:
                w2[ch, src] = K[g]
            b2[ch] = BIG
    # conv2 ch 8..15 : gated centre features (conv1 ch 8..15)
    for j in range(n_gated):
        ch = 8 + j
        w2[ch, n_always + j] = K[0]
        b2[ch] = BIG
    # conv2 ch 16..31 : ramp copies
    for j in range(n_ramp_total):
        ch = 16 + j
        w2[ch, 16 + j] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        b2[ch] = BIG

    # conv3 ch 0 : copy conv2 ch 0 (always-on centre)
    w3[0, 0] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]); b3[0] = BIG
    # conv3 ch 1..7 : gated copies
    for j in range(n_gated):
        ch = 1 + j
        w3[ch, 8 + j] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        b3[ch] = BIG
    # conv3 ch 8..23 : ramp copies
    for j in range(n_ramp_total):
        ch = 8 + j
        w3[ch, 16 + j] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        b3[ch] = BIG

    # ------------------------------------------------------------------
    # 8. read-out: choose weights so that out = A(t)*(x - c(t))
    #    always-on conv3 ch0 : F = BIG + x_p   (K[0] applied twice -> x)
    #    gated conv3 ch 1+j  : F = BIG + x_p + C*(t/31-u_j)_+   (if on)
    #    ramp conv3 ch 8+k   : F = BIG + (t/31-u_k)_+
    #    Note conv1 gated channel j has pre-act = x_p + BIG - C*(u_j - t/31)
    #      -> relu = x_p + BIG + C*(t/31 - u_j) for t>=u_j (ON)
    #      -> 0 for t < u_j (OFF)
    #    conv2 applies K[0] to it and adds BIG -> conv2 = BIG + x_p + C*(t/31-u_j)
    #    conv3 copies -> F3 = BIG + x_p + C*(t/31-u_j)  (only when ON)
    #    The offset C*(t/31-u_j) is only present when ON, i.e. t>=u_j, where
    #    it equals C*(t/31-u_j)_+ .  Good.
    # ------------------------------------------------------------------
    # spatial weight: always-on ch0 gets weight wA = base ; gated ch 1+j
    # gets weight wG_j = increment_j (so total gain at t = base + sum of
    # increments for gates already on).
    # We must also ensure the *filter* applied to x is identity: conv1 ch0
    # uses K[0] (centre), conv2 ch0 uses K[0] (centre) -> x; conv3 copies.
    # So out_spatial = (wA + sum_j wG_j*1[t>=u_j]) * x.
    # Set wA = base, and increments: we need the gain at t to equal g_s[t].
    # gate j (j=1..7) turns on at thresh[j]; increments inc_j = g_at[j]-g_at[j-1]
    # but our gated channels are indexed 0..7 with thresholds gate_u[j]=thresh[j]/31
    # (gate 0 has threshold 0 -> always on).
    # Use: gated channel 0 -> weight 0 (unused, threshold 0); gated channel
    # j (1..7) -> weight inc_j.
    # Then gain(t) = base + sum_{j>=1: t>=thresh[j]} inc_j = g_at[k] where
    # k = #{j: thresh[j] <= t}.  Good.
    # ------------------------------------------------------------------
    wA = float(base)
    inc = np.zeros(n_gated)
    for j in range(1, n_gated):
        inc[j] = float(g_at[j] - g_at[j - 1])
    # ------------------------------------------------------------------
    # 9. offsets: total output constant (t-independent) and t-dependent parts
    #    out = wA*(BIG + x) + sum_j inc_j*(BIG + x + C*(t/31-u_j)_+) [if on]
    #          + sum_k a_k*(BIG + (t/31-u_k)_+) + bo
    #    We want out = A(t)*x - A(t)*c(t).
    #    => t-dependent offset from gated channels: sum_j inc_j*C*(t/31-u_j)_+
    #       plus ramp part sum_k a_k*(t/31-u_k)_+
    #       must equal -A(t)*c(t)
    #    A(t) is piecewise constant with breakpoints at thresh[j].
    #    c(t) is piecewise constant (from data).  So -A(t)*c(t) is piecewise
    #    constant -> representable by hinges: a piecewise constant function
    #    = sum_j d_j * 1[t >= thresh_j] and 1[t>=thresh] = sum_k (hinge_k
    #    difference) ... Using hinges h_k(t) = (t/31 - u_k)_+ with a fine
    #    knot grid, we can represent any piecewise-linear function.
    #    Simplest: use the ramp knots to be exactly the thresholds, and
    #    represent the piecewise-constant function via hinges at the
    #    thresholds: 1[t>=thresh_j] = h_j(t) - h_{j+1}(t) + h_{j+2}(t) ... not
    #    exact.  Instead: use hinges with knots at thresh_j and thresh_j + 1
    #    (integer t grid) so that the difference of two adjacent hinges
    #    gives an indicator on one t value only.  Hmm.
    #
    #    Cleaner: make the ramp basis = indicator functions of t by using
    #    hinges on the integer grid: h_k(t) = (t - k)_+ for k = 0..31 (32
    #    hinges). Then any function f(t) with f(0)=0 can be written as
    #    f(t) = sum_k (f(k)-f(k-1)) h_k(t).  We have only 16 ramp
    #    channels though (n_ramp_total = 16).  We can afford 32 hinges if we
    #    reduce the number of always-on/gated channels.
    #
    #    Let's restructure the channel budget:
    #      conv1: 32 channels
    #        0      : always-on centre x feature
    #        1..8   : gated x features (8 gates)  -> 9 channels
    #        9..31  : 23 ramp channels (hinges)
    #      conv2: 64 channels (copy)
    #      conv3: 64 channels (copy)
    #      out  : 64 -> 1
    #    With 23 hinges we can represent a piecewise-linear function with
    #    23 knots -> enough for 8 steps of A(t)*c(t) plus the gated offsets.
    #
    #    Use hinge knots at t = 0,1,2,...,22 (integer grid) -> h_k(t)=(t-k)_+.
    #    Representing a piecewise-constant function on t in {0..31} needs
    #    hinges at every integer where it changes -> at most 8 changes (at
    #    thresh) -> need hinges at those 8 positions plus the offset
    #    corrections -> 23 is plenty.
    #
    #    BUT: the hinge (t-k)_+ is not expressible via relu(t/31 - u) with
    #    u = k/31 -> yes it is: relu(t/31 - k/31) = (t-k)/31. So the hinge
    #    value is (t-k)/31, scaled.  Fine.
    #
    #    Also the gated offset C*(t/31-u_j)_+ = C*(t - thresh_j)_+/31.
    # ------------------------------------------------------------------
    # ---- rebuild conv1 with new layout --------------------------------
    n_always = 1
    n_gated = 8
    n_ramp_total = 32 - n_always - n_gated      # 23
    w1 = np.zeros((32, 2, 3, 3)); b1 = np.zeros(32)
    # always-on
    w1[0, 0] = K[0]; b1[0] = 0.0
    # gated
    for j in range(n_gated):
        ch = n_always + j
        w1[ch, 0] = K[0] + 0.02 * rng.standard_normal((3, 3))
        w1[ch, 1] = -C / 9.0
        b1[ch] = BIG - C * gate_u[j]
    # ramps: hinge knots at integers 0..n_ramp_total-1 (t units)
    ramp_knot_t = np.arange(n_ramp_total, dtype=np.float64)
    for j in range(n_ramp_total):
        ch = n_always + n_gated + j
        w1[ch, 1] = 1.0 / 9.0
        b1[ch] = -ramp_knot_t[j] / 31.0
    # ---- rebuild conv2 -------------------------------------------------
    w2 = np.zeros((64, 32, 3, 3)); b2 = np.zeros(64)
    w3 = np.zeros((64, 64, 3, 3)); b3 = np.zeros(64)
    wo = np.zeros((1, 64, 3, 3)); bo = np.zeros(1)
    # conv2 ch 0 : always-on
    w2[0, 0] = K[0]; b2[0] = BIG
    # conv2 ch 1..8 : gated
    for j in range(n_gated):
        w2[1 + j, n_always + j] = K[0]; b2[1 + j] = BIG
    # conv2 ch 9..31 : ramps
    for j in range(n_ramp_total):
        w2[9 + j, n_always + n_gated + j] = np.array([[0., 0., 0.],
                                                        [0., 1., 0.],
                                                        [0., 0., 0.]])
        b2[9 + j] = BIG
    # conv3 ch 0 : always-on
    w3[0, 0] = np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]); b3[0] = BIG
    # conv3 ch 1..8 : gated
    for j in range(n_gated):
        w3[1 + j, 1 + j] = np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
        b3[1 + j] = BIG
    # conv3 ch 9..31 : ramps
    for j in range(n_ramp_total):
        w3[9 + j, 9 + j] = np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
        b3[9 + j] = BIG
    # ------------------------------------------------------------------
    # 10. solve the read-out weights exactly
    #     Model (per example, per pixel):
    #       out = wA*(BIG + x) + sum_j inc_j*(BIG + x + C*h_j(t))
    #             + sum_k a_k*(BIG + h_k(t)) + bo
    #     where h_j(t) = (t - thresh_j)_+/31  (gated hinge)
    #           h_k(t) = (t - k)_+/31          (ramp hinge)
    #     Desired: out = A(t)*x - A(t)*c(t)
    #     => spatial coefficient: wA + sum_j inc_j = A(t)  (already set)
    #     => offset: sum_j inc_j*C*h_j(t) + sum_k a_k*h_k(t) + bo
    #                + (wA + sum_j inc_j)*BIG
    #              must equal -A(t)*c(t)
    #     Solve for a_k (23 unknowns) and bo by least squares over the 32
    #     discrete t values, using the exact hinge basis.
    # ------------------------------------------------------------------
    tt = np.arange(32, dtype=np.float64)
    # A(t) from the step construction
    A_t = np.zeros(32)
    for tval in range(32):
        val = base
        for j in range(1, n_gated):
            if tval >= thresh[j]:
                val += inc[j]
        A_t[tval] = val
    # desired offset function (excluding the BIG constant handled by bo)
    # d(t) = -A(t)*c(t)
    c_tab_full = np.array([c_tab[k] for k in range(32)])
    d_t = -A_t * c_tab_full
    # basis: hinges h_k(t) = (t-k)_+/31 for k = 0..22, plus constant 1
    B = np.zeros((32, n_ramp_total + 1))
    for k in range(n_ramp_total):
        B[:, k] = np.maximum(tt - ramp_knot_t[k], 0.0) / 31.0
    B[:, n_ramp_total] = 1.0
    # target: d_t minus the known contributions:
    #   gated hinges: sum_j inc_j*C*h_j(t)  (h_j uses knot thresh[j])
    known = np.zeros(32)
    for j in range(1, n_gated):
        hj = np.maximum(tt - thresh[j], 0.0) / 31.0
        known += inc[j] * C * hj
    # BIG constant from spatial channels: (wA + sum inc_j)*BIG
    known += (wA + inc.sum()) * BIG
    rhs = d_t - known
    # least squares (with tiny ridge for stability)
    lam = 1e-8
    BtB = B.T @ B + lam * np.eye(n_ramp_total + 1)
    sol = np.linalg.solve(BtB, B.T @ rhs)
    a_k = sol[:n_ramp_total]
    bo = float(sol[n_ramp_total])
    # ------------------------------------------------------------------
    # 11. assemble read-out
    #     out channel weights: only the centre tap (1x1 conv) is used.
    wo[0, 0, 1, 1] = wA
    for j in range(n_gated):
        wo[0, 1 + j, 1, 1] = inc[j]
    for k in range(n_ramp_total):
        wo[0, 9 + k, 1, 1] = a_k[k]
    bo = np.array([bo], dtype=np.float64)
    # ------------------------------------------------------------------
    # 12. write back
    # ------------------------------------------------------------------
    out = {}
    out["conv1.weight"] = w1.astype(np.float32)
    out["conv1.bias"] = b1.astype(np.float32)
    out["conv2.weight"] = w2.astype(np.float32)
    out["conv2.bias"] = b2.astype(np.float32)
    out["conv3.weight"] = w3.astype(np.float32)
    out["conv3.bias"] = b3.astype(np.float32)
    out["out.weight"] = wo.astype(np.float32)
    out["out.bias"] = bo.astype(np.float32)
    for k, v in out.items():
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        out[k] = np.clip(v, -1000.0, 1000.0).astype(np.float32)
    # keep any extra keys untouched
    for k, v in w.items():
        if k not in out:
            out[k] = v

    # ------------------------------------------------------------------
    # 13. diagnostics: closed-form MSE of the intended predictor
    # ------------------------------------------------------------------
    pred = A_t[t] * (xf - c_tab_full[t][:, None])   # [N,256]
    mse_fit = float(np.mean((pred - ef) ** 2))
    mse_zero = float(np.mean(ef ** 2))
    diagnostics = {
        "eps_consistency_mse": cons,
        "mse_closed_form": mse_fit,
        "mse_predict_zero": mse_zero,
        "g_t0": float(g_s[0]), "g_t15": float(g_s[15]), "g_t31": float(g_s[31]),
        "A_t0": float(A_t[0]), "A_t31": float(A_t[31]),
        "c_t0": float(c_tab_full[0]), "c_t31": float(c_tab_full[31]),
        "n_gates": int(n_gated),
    }
    return {"weights": out, "diagnostics": diagnostics}
