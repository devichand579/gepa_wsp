import numpy as np

def edit(w, f):
    # ---------------- data ----------------
    x = np.asarray(f["x"], dtype=np.float64)          # [N,1,16,16]
    y = np.asarray(f["y"], dtype=np.float64)          # [N,1,16,16] target eps
    t = np.asarray(f["t"]).astype(np.int64).ravel()   # [N]
    N = x.shape[0]
    H = W_ = 16
    ab = np.asarray(f["alphabar"], dtype=np.float64).ravel()   # [32]

    # ---------------- local patch design (7x7 offsets, zero padded) -------
    R = 3
    offs = [(dr, dc) for dr in range(-R, R + 1) for dc in range(-R, R + 1)]
    P = len(offs)                                     # 49
    # design matrix: rows = N*256 pixels, cols = P offsets (+ intercept)
    xp = x[:, 0]                                      # [N,16,16]
    cols = []
    for (dr, dc) in offs:
        cols.append(np.roll(np.roll(xp, dr, axis=1), dc, axis=2))
    Xp = np.stack(cols, axis=-1).reshape(N * 256, P)   # [N*256,49]
    Yp = y[:, 0].reshape(N * 256)

    # ---------------- per-timestep ridge least squares -------------------
    lam = 3.0
    w_t = np.zeros((32, P))
    b_t = np.zeros(32)
    for tt in range(32):
        idx = np.where(t == tt)[0]
        if idx.size < 5:
            continue
        rows = (idx[:, None] * 256 + np.arange(256)[None, :]).ravel()
        A = Xp[rows]
        b = Yp[rows]
        A1 = np.concatenate([A, np.ones((A.shape[0], 1))], axis=1)
        G = A1.T @ A1
        G[np.arange(P), np.arange(P)] += lam * max(1.0, np.trace(G[:P, :P]) / P)
        G[P, P] += 1e-6
        rhs = A1.T @ b
        try:
            sol = np.linalg.solve(G, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(G, rhs, rcond=None)[0]
        w_t[tt] = sol[:P]
        b_t[tt] = sol[P]

    # ---------------- realize in the fixed architecture -------------------
    # conv1: 32 channels
    #   ch 0..8   : x_t shifted by (dr,dc)          (offset basis, 3x3)
    #   ch 9..17  : x_t shifted by (dr,dc) + c(t)   (t-dependent offsets)
    #   ch 18..31 : constant gates (on/off per t)   (unused, kept inert)
    c1w = np.zeros((32, 2, 3, 3))
    c1b = np.zeros(32)
    # offset basis channels: 3x3 kernel with single 1 at (dr+1,dc+1)
    for i, (dr, dc) in enumerate(offs[:9]):
        c1w[i, 0, dr + 1, dc + 1] = 1.0
    # t-dependent offset channels: same spatial kernel, plus t-channel gain
    #   preact = shifted_x + s*(t/31) + b   ->  c(t) = b + s*t/31
    #   choose s so that c(t) sweeps a wide range across t=0..31
    s_amp = 60.0
    for i, (dr, dc) in enumerate(offs[:9]):
        c1w[9 + i, 0, dr + 1, dc + 1] = 1.0
        c1w[9 + i, 1, 1, 1] = s_amp            # gain on the t/31 channel
        c1b[9 + i] = 0.0
    # gate channels 18..31: constant, always-on (large positive bias)
    for i in range(18, 32):
        c1b[i] = 10.0

    # conv2 / conv3: identity pass-through on the first 18 channels
    c2w = np.zeros((64, 32, 3, 3))
    c2b = np.zeros(64)
    for j in range(18):
        c2w[j, j, 1, 1] = 1.0
    c3w = np.zeros((64, 64, 3, 3))
    c3b = np.zeros(64)
    for j in range(18):
        c3w[j, j, 1, 1] = 1.0

    # out: 3x3 filter over the 9 t-dependent-offset channels
    #   out(p) = sum_i Wo[0,i,1,1] * (shift_i x)(p) + sum_i Wo[0,i,dr,dc] * x(p+off)
    # we build:  base filter K (7x7) from the 9 offset channels' spatial taps,
    #   plus per-t gain g(t) applied to the identity-offset channels.
    # channels 0..8 hold shifted x (no t dependence) -> used for the base filter
    # channels 9..17 hold shifted x + c_i(t) -> used for the t-dependent part
    #
    # Desired: eps_hat(p) = sum_{off} K(off) x(p+off) + g(t) * sum_{off2} G(off2) x(p+off2) + b(t)
    # We realise the t-dependent part by giving channels 9..17 the SAME spatial
    # kernel as channels 0..8 but with an extra constant c_i(t); the out-layer
    # spatial taps then pick up (x shifted) and the constant part is t-dependent.
    # Simplify: use a single shared 3x3 spatial filter for the base term and let the
    # per-t gain act on the 3x3 neighbourhood via channels 9..17.
    #
    # Implementation: base filter K3 (3x3) applied to x  -> channels 0..8 give the
    # 9 spatial taps of x; out spatial taps on channels 0..8 realise K3.
    # t-dependent gain: channels 9..17 carry (x shifted by off) + c_i(t); out
    # spatial taps on channels 9..17 realise G3 * (shifted x) plus constant
    # sum_i Wo[0,9+i,1,1]*c_i(t) which is t-dependent -> gives b(t) shape.
    # To get a t-dependent *gain* on x we use pairs: channel 9+i carries
    #   shifted x + c_i(t) with c_i(t) = -h(t) and channel 18+i carries
    #   shifted x + h(t); their difference (out weights +1/-1) cancels constants
    #   only if the two shifts are identical -> then difference = 0. Instead:
    #   use channel 9+i = shifted_i x + c(t) and channel 18+i = shifted_i x + c(t)
    #   with out weights (g_i(t)?) -- out weights are fixed, so gain must come
    #   from the *activation pattern*, i.e. from c(t) crossing thresholds.
    # => We therefore implement the t-dependence through thresholded channels:
    #    channel 9+i: preact = shifted_i x + c_i(t), c_i(t) = -M + M*t/31 (on late)
    #    channel 18+i: preact = shifted_i x + d_i(t), d_i(t) = M - M*t/31 (on early)
    #    and the out layer forms  sum_i [ Wo(9+i)*ch9_i + Wo(18+i)*ch18_i ].
    #    With Wo(9+i)=+1, Wo(18+i)=-1 and identical shifts, the sum equals
    #    (shifted x + c(t))_+ - (shifted x + d(t))_+ which is a soft-threshold
    #    (nonlinear) feature; the *linear* part is x whenever both are active.
    # => The cleanest exact construction: keep the base 3x3 filter realised on
    #    channels 0..8 (always active), and realise the per-timestep gain g(t) by
    #    scaling that filter through a set of always-on/always-off gated channels
    #    that duplicate the 9 spatial taps.  Gate channels are constant in x, so
    #    their contribution to out is a constant (not a gain on x).
    #    Therefore: realise per-timestep filters by DUPLICATING the 9 tap channels
    #    for each of a small number of t-regimes, each duplicate gated on/off by
    #    its own c(t) threshold, and give each regime its own out filter.
    #    With 32 conv1 channels we can afford 9 base taps + 9 taps for one
    #    t-gated copy + 14 spare.  A single extra copy gives a 2-regime model.
    #    To obtain many regimes we instead exploit that each of the 9 gated
    #    channels has its OWN threshold c_i(t) (different s_i), so the pattern of
    #    which gated channels are active varies with t -> the out layer then
    #    implements a t-dependent linear combination of the 9 taps.
    #    => use channels 9..17 with per-channel thresholds s_i chosen so that
    #       each channel switches at a distinct timestep; out weights then give
    #       a piecewise-constant-in-t filter.  Combined with the always-on base
    #       channels this yields a rich t-dependent filter.

    # --- per-timestep filter design: express w_t (7x7) in terms of the 9 taps
    #     available from channels 0..8 (3x3 neighbourhood) and channels 9..17
    #     (t-gated 3x3 neighbourhood).
    # 3x3 taps of channels 0..8 give the 3x3 neighbourhood of x.
    # For the t-dependent part we use channels 9..17 which carry the SAME 3x3
    # taps but with a t-dependent offset c_i(t) that gates them on/off.
    # Choose thresholds so channel 9+i is active only for t >= thr_i.
    # Then out = sum_i Wo0_i * tap_i(x) + sum_i Wo1_i * gate_i(t) * tap_i(x)
    # with gate_i(t) = 1[t >= thr_i].  This is a t-dependent filter built from
    # 3x3 taps: filter_t = Wo0 + sum_{i: thr_i <= t} Wo1_i.
    # We choose thresholds spread over t and solve for Wo0, Wo1 by least squares
    # over the 32 timesteps (each t has its own target filter w_t).
    # Represent each w_t restricted to the central 3x3 (offsets with |dr|,|dc|<=1)
    # as target; the outer taps of the 7x7 fit are folded into the base channels
    # by using the 9 base channels to carry the full 7x7 information is impossible
    # (3x3 only) -> so we fit the model directly in the 3x3-tap space:
    #   eps_hat(p) = sum_i A_i(t) * x(p + off_i) + b(t),  off_i in 3x3
    # with A_i(t) = Wo0_i + sum_{j: thr_j<=t} Wo1_j.
    # Fit A(t) directly per timestep by least squares (9 taps + intercept), then
    # decompose A(t) into Wo0 + cumulative Wo1 (any decomposition works; choose
    # thresholds = 1..31 so Wo1_j = A(j) - A(j-1)).
    # The 7x7 outer taps are dropped (they only help marginally); we instead
    # enlarge the effective receptive field by using conv2/conv3 to compose
    # 3x3 -> 5x5 -> 7x7 for the base (always-on) part.

    # ---- Step 1: fit per-timestep 3x3-tap linear model (with intercept) ----
    taps3 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
    cols3 = []
    for (dr, dc) in taps3:
        cols3.append(np.roll(np.roll(xp, dr, axis=1), dc, axis=2))
    X3 = np.stack(cols3, axis=-1).reshape(N * 256, 9)

    A_t = np.zeros((32, 9))
    c_t = np.zeros(32)
    for tt in range(32):
        idx = np.where(t == tt)[0]
        if idx.size < 5:
            continue
        rows = (idx[:, None] * 256 + np.arange(256)[None, :]).ravel()
        A = X3[rows]
        bb = Yp[rows]
        A1 = np.concatenate([A, np.ones((A.shape[0], 1))], axis=1)
        G = A1.T @ A1
        G[np.arange(9), np.arange(9)] += 2.0
        G[9, 9] += 1e-6
        sol = np.linalg.solve(G, A1.T @ bb)
        A_t[tt] = sol[:9]
        c_t[tt] = sol[9]

    # ---- Step 2: build a 7x7 base filter from the 3x3 fit (zero outer) ----
    # base channels 0..8 carry the 3x3 taps of x; out spatial taps realise A_base
    # (a fixed 3x3 filter) -> contributes A_base * x for all t.
    # t-dependent part: channels 9..17 carry the same 3x3 taps but gated by
    # thresholds; out spatial taps realise A_delta, giving A(t) = A_base + cumsum.
    # thresholds: channel 9+i switches on at timestep thr_i (1..31).
    # choose thr_i = i+1 for i in 0..8  (9 switches) -> A(t) piecewise constant
    # with 9 breakpoints; fit A(t) as base + cumulative increments by least
    # squares over t (so the piecewise-constant approximation of A(t) is optimal
    # in the weighted-by-count sense).
    # Instead of approximating, we can do better: use the fact that the *gate*
    # value is binary; but we can also exploit the soft nature of ReLU: a channel
    # with preact = tap + c(t) where c(t) is large-magnitude is fully on/off.
    # Use: for channel 9+i, c_i(t) = M*(thr_i - t)/31 - M*0.5  -> on iff t<thr_i
    # (fully off for t >= thr_i).  With M large (e.g. 40) and |tap|<=4 the
    # transition is sharp; but a sharp transition means the model is exactly
    # piecewise-constant in t.  To get a smoother (better) approximation we can
    # additionally allow the always-on base to carry the average.
    # Weighted LS for the decomposition:
    #   minimise sum_t n_t || A_base + sum_{i: thr_i<=t} D_i - A_t ||^2
    #   -> linear in (A_base, D_0..D_8); solve 9+9 unknowns.
    thr = np.array([1, 3, 5, 7, 9, 12, 15, 19, 24])   # switch-on timesteps
    cnt = np.array([np.sum(t == tt) for tt in range(32)], dtype=np.float64)
    M = 40.0
    # design matrix rows: timesteps, cols: [base(9), D_0..D_8]
    Dm = np.zeros((32, 18))
    for tt in range(32):
        Dm[tt, :9] = 1.0
        for i in range(9):
            if thr[i] <= tt:
                Dm[tt, 9 + i] = 1.0
    Wt = cnt[:, None] * Dm
    G = Dm.T @ Wt
    G[np.arange(18), np.arange(18)] += 1e-3
    rhs = Dm.T @ (Wt * A_t)
    sol = np.linalg.solve(G, rhs)
    A_base = sol[:9]
    D_inc = sol[9:]

    # ---- Step 3: assemble conv1 / conv2 / conv3 / out ----
    # conv1 channel layout (32 channels):
    #   0..8   : tap_i(x)                       (always on)
    #   9..17  : tap_i(x) + c_i(t)              (on for t < thr_i, i.e. "early")
    #   18..26 : tap_i(x) + d_i(t)              (on for t >= thr_i, i.e. "late")
    #   27..31 : unused, constant positive (inert)
    # out: out(p) = sum_i Wo[0,i,1,1] * ch_i(p)  for the 9 base channels gives
    #      A_base * (3x3 taps of x)
    #      plus gated channels give the increments.
    # To realise A_base * tap: channel i holds tap_i(x) at spatial offset off_i,
    # so out needs weight A_base[i] on channel i at the spatial tap that undoes
    # the shift: out(p) = sum_i Wo[0,i,dr,dc] * ch_i(p + (dr,dc)) ; we want
    # sum_i A_base[i] * x(p+off_i).  With ch_i(q) = x(q+off_i) we need
    # Wo[0,i,dr,dc] such that sum over taps of ch_i gives x(p+off_i): choose the
    # single tap (dr,dc)=(-off_i) i.e. Wo[0,i,1-off_i_dc... ] -> set
    # Wo[0,i,1,1] = A_base[i] and instead define ch_i to be x shifted by -off_i.
    # Simpler: define channel i to hold x shifted by (-off_i) so that
    # out(p) = sum_i A_base[i] * x(p + off_i) with Wo[0,i,1,1] = A_base[i].
    taps3 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
    c1w = np.zeros((32, 2, 3, 3))
    c1b = np.zeros(32)
    for i, (dr, dc) in enumerate(taps3):
        c1w[i, 0, 1 - dr, 1 - dc] = 1.0            # ch = x shifted by (-dr,-dc)
    for i, (dr, dc) in enumerate(taps3):
        c1w[9 + i, 0, 1 - dr, 1 - dc] = 1.0
        c1w[9 + i, 1, 1, 1] = -M / 31.0            # c_i(t) = -M*t/31 + const
        c1b[9 + i] = M * 0.5 - M * thr[i] / 31.0 * 0.0
        # want on iff t < thr_i :  preact_extra = -M*(t - thr_i + 0.5)/31
        c1w[9 + i, 1, 1, 1] = -M / 31.0
        c1b[9 + i] = M * (thr[i] - 0.5) / 31.0
        c1w[18 + i, 0, 1 - dr, 1 - dc] = 1.0
        c1w[18 + i, 1, 1, 1] = M / 31.0
        c1b[18 + i] = M * (0.5 - thr[i]) / 31.0      # on iff t >= thr_i
    for i in range(27, 32):
        c1b[i] = 5.0

    # conv2 / conv3 identity pass-through on channels 0..26
    c2w = np.zeros((64, 32, 3, 3))
    c2b = np.zeros(64)
    for j in range(27):
        c2w[j, j, 1, 1] = 1.0
    c3w = np.zeros((64, 64, 3, 3))
    c3b = np.zeros(64)
    for j in range(27):
        c3w[j, j, 1, 1] = 1.0

    # out layer: 1x64x3x3
    ow = np.zeros((1, 64, 3, 3))
    ob = np.zeros(1)
    for i in range(9):
        ow[0, i, 1, 1] = A_base[i]
    for i in range(9):
        ow[0, 9 + i, 1, 1] = D_inc[i]
        ow[0, 18 + i, 1, 1] = -D_inc[i]
    # constant part: intercepts.  Channels 9..17 contribute D_inc[i]*c_i(t) when
    # active, channels 18..26 contribute -D_inc[i]*d_i(t) when active.
    # We realise the per-timestep intercept b(t) by adding a dedicated constant
    # channel set: channels 27..31 are constants (always on) -> they can only
    # give a constant, not t-dependent.  Instead: use the SAME gating trick with
    # channels 9..17 / 18..26 but with zero spatial filter and a constant
    # preactivation?  A channel's preactivation is (spatial filter of x) + c(t);
    # with zero spatial filter it is constant c(t) -> after ReLU it is
    # max(c(t),0) which is 0 or M-ish -> gives a t-dependent constant.
    # We need 9 more channels for that; we have 27..31 (5).  Instead we encode
    # the intercept by adding a constant channel to the 3x3 tap set: use the
    # centre tap channel to also carry a constant?  Not possible (one preact).
    # Practical compromise: fold the intercept into the tap fit by adding a
    # constant "tap" channel.  We have 5 spare channels (27..31).  We use them
    # for a piecewise-constant-in-t intercept with 5 breakpoints, and accept the
    # residual intercept error (small: intercepts are ~0.2-0.35 and mostly
    # absorbed by the mean tap behaviour).
    # Better: reduce the number of gated pairs to 8 and free 2 channels for the
    # intercept (each gated pair needs 2 channels: early + late).  We use 8
    # pairs (16 channels) -> 9 base + 16 = 25 channels, leaving 7 for intercept
    # pieces (each piece needs 1 early + 1 late channel = 2 channels per
    # breakpoint -> 3 breakpoints with 6 channels, 1 spare).
    # Rebuild with 8 gated pairs (thr_1..thr_8) and 3 intercept pieces.
    # ---- recompute decomposition with 8 increments + intercept pieces ----
    thr8 = np.array([2, 5, 8, 11, 15, 19, 24, 29])
    Dm2 = np.zeros((32, 9 + 8))
    for tt in range(32):
        Dm2[tt, :9] = 1.0
        for i in range(8):
            if thr8[i] <= tt:
                Dm2[tt, 9 + i] = 1.0
    Wt2 = cnt[:, None] * Dm2
    G2 = Dm2.T @ Wt2
    G2[np.arange(17), np.arange(17)] += 1e-3
    rhs2 = Dm2.T @ (Wt2 * A_t)
    sol2 = np.linalg.solve(G2, rhs2)
    A_base2 = sol2[:9]
    D_inc2 = sol2[9:]

    # intercept pieces: fit c(t) with 3 step functions (breakpoints)
    thrI = np.array([8, 17, 26])
    DmI = np.zeros((32, 3))
    for tt in range(32):
        for j in range(3):
            DmI[tt, j] = 1.0 if tt >= thrI[j] else 0.0
    WtI = cnt[:, None] * DmI
    GI = DmI.T @ WtI
    GI[np.arange(3), np.arange(3)] += 1e-3
    rhsI = DmI.T @ (WtI * c_t[:, None])
    cI = np.linalg.solve(GI, rhsI)

    # ---- final assembly ----
    c1w = np.zeros((32, 2, 3, 3))
    c1b = np.zeros(32)
    for i, (dr, dc) in enumerate(taps3):
        c1w[i, 0, 1 - dr, 1 - dc] = 1.0
    for i in range(8):
        dr, dc = taps3[i]
        c1w[9 + i, 0, 1 - dr, 1 - dc] = 1.0
        c1w[9 + i, 1, 1, 1] = -M / 31.0
        c1b[9 + i] = M * (thr8[i] - 0.5) / 31.0
        c1w[17 + i, 0, 1 - dr, 1 - dc] = 1.0
        c1w[17 + i, 1, 1, 1] = M / 31.0
        c1b[17 + i] = M * (0.5 - thr8[i]) / 31.0
    # intercept pieces: channels 25,26,27  (early-on / late-on pairs -> 6 chans)
    # use channels 25..30 as 3 pairs
    for j in range(3):
        c1w[25 + 2 * j, 0, 1, 1] = 0.0
        c1w[25 + 2 * j, 1, 1, 1] = -M / 31.0
        c1b[25 + 2 * j] = M * (thrI[j] - 0.5) / 31.0
        c1w[26 + 2 * j, 1, 1, 1] = M / 31.0
        c1b[26 + 2 * j] = M * (0.5 - thrI[j]) / 31.0
    c1b[31] = 5.0

    c2w = np.zeros((64, 32, 3, 3))
    c2b = np.zeros(64)
    for j in range(31):
        c2w[j, j, 1, 1] = 1.0
    c3w = np.zeros((64, 64, 3, 3))
    c3b = np.zeros(64)
    for j in range(31):
        c3w[j, j, 1, 1] = 1.0

    ow = np.zeros((1, 64, 3, 3))
    ob = np.zeros(1)
    for i in range(9):
        ow[0, i, 1, 1] = A_base2[i]
    for i in range(8):
        ow[0, 9 + i, 1, 1] = D_inc2[i]
        ow[0, 17 + i, 1, 1] = -D_inc2[i]
    for j in range(3):
        ow[0, 25 + 2 * j, 1, 1] = cI[j]
        ow[0, 26 + 2 * j, 1, 1] = -cI[j]

    # ---------------- verify numerically with an exact forward pass --------
    def conv3x3(inp, wt, bias):
        # inp [N,C,H,W], wt [O,C,3,3] cross-correlation, pad 1
        N, C, H, Wd = inp.shape
        O = wt.shape[0]
        p = np.pad(inp, ((0, 0), (0, 0), (1, 1), (1, 1)))
        out = np.zeros((N, O, H, Wd))
        for kr in range(3):
            for kc in range(3):
                sl = p[:, :, kr:kr + H, kc:kc + Wd]
                out += np.tensordot(sl, wt[:, :, kr, kc], axes=([1], [1])).transpose(1, 0, 2, 3)
        out += bias[None, :, None, None]
        return out

    def forward(wt, xin, tin):
        N = xin.shape[0]
        tt = (tin.astype(np.float64) / 31.0)[:, None, None, None]
        z = np.concatenate([xin, np.broadcast_to(tt, (N, 1, 16, 16))], axis=1)
        h1 = np.maximum(conv3x3(z, wt["conv1.weight"], wt["conv1.bias"]), 0.0)
        h2 = np.maximum(conv3x3(h1, wt["conv2.weight"], wt["conv2.bias"]), 0.0)
        h3 = np.maximum(conv3x3(h2, wt["conv3.weight"], wt["conv3.bias"]), 0.0)
        o = conv3x3(h3, wt["out.weight"], wt["out.bias"])
        return o

    neww = {
        "conv1.weight": c1w.astype(np.float32),
        "conv1.bias": c1b.astype(np.float32),
        "conv2.weight": c2w.astype(np.float32),
        "conv2.bias": c2b.astype(np.float32),
        "conv3.weight": c3w.astype(np.float32),
        "conv3.bias": c3b.astype(np.float32),
        "out.weight": ow.astype(np.float32),
        "out.bias": ob.astype(np.float32),
    }
    pred = forward(neww, x, t)
    mse_new = float(np.mean((pred - y) ** 2))
    mse_old = float(np.mean((np.asarray(f["prediction"], dtype=np.float64) - y) ** 2))

    # safety: keep everything finite and bounded
    for k in neww:
        v = np.nan_to_num(neww[k], nan=0.0, posinf=0.0, neginf=0.0)
        v = np.clip(v, -1000.0, 1000.0).astype(np.float32)
        neww[k] = v

    return {"weights": neww,
            "diagnostics": {"mse_new": mse_new, "mse_old": mse_old,
                            "n_timesteps_used": int((cnt > 0).sum()),
                            "A_base_absmax": float(np.abs(A_base2).max()),
                            "D_inc_absmax": float(np.abs(D_inc2).max())}}
