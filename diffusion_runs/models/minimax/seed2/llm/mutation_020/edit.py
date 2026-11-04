import numpy as np

def edit(w, f):
    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def conv2d(x, wt, b, pad=1):
        # x: [N,C,H,W], wt: [O,C,kh,kw] (cross-correlation)
        N, C, H, W = x.shape
        O, C2, kh, kw = wt.shape
        xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        cols = np.lib.stride_tricks.sliding_window_view(xp, (kh, kw), axis=(2, 3))
        # cols: [N,C,H,W,kh,kw]
        cols = np.transpose(cols, (0, 3, 4, 1, 5, 6)).reshape(N * H * W, C * kh * kw)
        out = cols @ wt.reshape(O, C * kh * kw).T
        out = out.reshape(N, H, W, O).transpose(0, 3, 1, 2)
        if b is not None:
            out = out + b.reshape(1, O, 1, 1)
        return out

    def get(k):
        v = f[k]
        return np.asarray(v, dtype=np.float64)

    x = get("x")            # [N,1,16,16]
    t = get("t").astype(np.int64)
    y = get("y")            # [N,1,16,16]
    clean = get("clean")
    ab = get("alphabar")    # [32]
    N = x.shape[0]

    # ------------------------------------------------------------------
    # 1) per-timestep optimal gain c_t for the model  eps_hat = c_t * x_t
    #    (closed form: c_t = <x,y>/<x,x> per timestep)
    # ------------------------------------------------------------------
    xf = x.reshape(N, -1)
    yf = y.reshape(N, -1)
    c_t = np.zeros(32)
    for k in range(32):
        m = t == k
        if m.sum() >= 2:
            xk = xf[m]
            yk = yf[m]
            den = np.sum(xk * xk)
            c_t[k] = np.sum(xk * yk) / den if den > 1e-12 else 0.0
    c_t = np.clip(c_t, -3.0, 3.0)

    # ------------------------------------------------------------------
    # 2) build a "gate" that multiplies x_t by a t-dependent positive factor
    #    and a complementary channel so that the readout can recover
    #       x_t  and  x_t * (t-dependent scale)
    #    conv1 input = [x_t , t/31]
    #    channel 0 : relu( x_t + K )              -> x_t + K   (always active)
    #    channel 1 : relu( x_t + K - G*(t/31) )  -> x_t + K - G*t/31
    #    channel 2 : relu( -x_t + K )            -> -x_t + K
    #    channel 3 : relu( -x_t + K + G*(t/31) )  -> -x_t + K + G*t/31
    #    with K large and G large so the "on" set is exactly t <= 15.
    #    Then  ch1 - ch0 = -G*t/31  (a per-example constant) and
    #          ch3 - ch2 = +G*t/31.
    #    The readout (linear) can therefore form any affine function of
    #    (x_t, t/31) *within* the gate region, and 0 outside it.
    # ------------------------------------------------------------------
    G = 100.0
    K = 50.0
    half = 16  # t/31 in [0,1]; threshold at 0.5 -> t <= 15

    w1 = np.zeros((32, 2, 3, 3))
    b1 = np.zeros(32)
    # ch0: +x_t
    w1[0, 0, 1, 1] = 1.0
    b1[0] = K
    # ch1: +x_t - G*t/31
    w1[1, 0, 1, 1] = 1.0
    w1[1, 1, 1, 1] = -G / 31.0
    b1[1] = K
    # ch2: -x_t
    w1[2, 0, 1, 1] = -1.0
    b1[2] = K
    # ch3: -x_t + G*t/31
    w1[3, 0, 1, 1] = -1.0
    w1[3, 1, 1, 1] = G / 31.0
    b1[3] = K

    # ------------------------------------------------------------------
    # 3) conv2 / conv3 : pass the useful channels through unchanged.
    #    conv2 maps 32 -> 64 ; conv3 maps 64 -> 64.
    #    We keep channels 0..3 (gate channels) and additionally create
    #    "identity" copies of x_t so that the readout has a clean handle.
    # ------------------------------------------------------------------
    w2 = np.zeros((64, 32, 3, 3))
    b2 = np.zeros(64)
    w3 = np.zeros((64, 64, 3, 3))
    b3 = np.zeros(64)

    # conv2: channels 0..3 -> 0..3 (identity, delta kernel)
    for c in range(4):
        w2[c, c, 1, 1] = 1.0
    # conv2: channel 4 = ch0 - ch2  -> 2*x_t  (pure x_t handle)
    w2[4, 0, 1, 1] = 1.0
    w2[4, 2, 1, 1] = -1.0
    # conv2: channel 5 = ch1 - ch3  -> -2*G*t/31 (per-example constant)
    w2[5, 1, 1, 1] = 1.0
    w2[5, 3, 1, 1] = -1.0
    # conv2: channel 6 = ch0 + ch2  -> 2K (constant)
    w2[6, 0, 1, 1] = 1.0
    w2[6, 2, 1, 1] = 1.0
    # conv2: channel 7 = ch1 + ch3  -> 2K  (constant)
    w2[7, 1, 1, 1] = 1.0
    w2[7, 3, 1, 1] = 1.0

    # conv3: identity on 0..7, plus extra x_t copies (shifted) so the
    # readout can use a 9x9 neighbourhood of x_t.
    for c in range(8):
        w3[c, c, 1, 1] = 1.0
    # conv3 channel 8..: copies of conv2 channel 4 (2 x_t) with shifts
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for i, (dy, dx) in enumerate(shifts):
        w3[8 + i, 4, 1 + dy, 1 + dx] = 1.0
    # conv3 channel 16 : copy of conv2 channel 5 (constant -2G t/31)
    w3[16, 5, 1, 1] = 1.0
    # conv3 channel 17 : copy of conv2 channel 6 (constant 2K)
    w3[17, 6, 1, 1] = 1.0

    # ------------------------------------------------------------------
    # 4) readout design: out = sum_k W[k] * conv3_out[k] + b
    #    conv3 channel 4  = 2 * x_t          (centre)
    #    conv3 channel 8+i= 2 * x_t shifted by (dy,dx)
    #    conv3 channel 16 = -2G t/31
    #    conv3 channel 17 = 2K
    #
    #    We want  out = c_t * x_t  (+ optional constant).
    #    The gate: ch0 active for all t; ch1 active for t <= 15.
    #    So conv3 channel 0 = x_t + K (all t), channel 1 = x_t + K - G t/31
    #    (t<=15 only).  Using these two, the readout can build
    #        x_t * a(t)  +  const(t)
    #    where a(t) is a free gain per gate region (2 regions) and const(t)
    #    is a free per-region constant.
    #
    #    We realise, for region r with gain g_r and offset o_r:
    #        out = g_r * x_t + o_r
    #    by using conv3 channels 0 (x_t+K) and 1 (x_t+K-G t/31) with
    #    readout weights  W0 = (g_r - g_s)/2 ... implemented below.
    # ------------------------------------------------------------------
    # Build a per-region linear model:
    #   region A : t <= 15   (ch1 active)
    #   region B : t >= 16   (ch1 inactive -> only ch0)
    #   ch0 = x_t + K  (always)
    #   ch1 = x_t + K - G*t/31 (region A only)
    #   => in region A:  x_t = (ch0 + ch1)/2 + G*t/31/2
    #      and  t/31 = (ch0 - ch1)/G
    #   => in region B:  x_t = ch0 - K
    #
    #   So define readout weights:
    #      region A: out = gA * ((ch0+ch1)/2 + G*t/31/2) + oA
    #      region B: out = gB * (ch0 - K) + oB
    #   To realise both simultaneously with ONE linear readout we exploit
    #   that in region B, ch1 = 0 (constant), so any coefficient on ch1
    #   contributes a constant in region B which we can absorb into the
    #   region-B offset.  Therefore:
    #      W0 = (gA + gB)/2      (coefficient on ch0)
    #      W1 = (gA - gB)/2      (coefficient on ch1)
    #      out = W0*ch0 + W1*ch1 + b
    #   region A: out = W0(x_t+K) + W1(x_t+K-G t/31) + b
    #             = (W0+W1) x_t + (W0+W1)K - W1 G t/31 + b
    #             = gA x_t + gA K - W1 G t/31 + b
    #   region B: out = W0 (x_t + K) + b = gB x_t + gB K + b
    #
    #   We want region A: gA x_t + oA  => need the t/31 term to vanish:
    #     W1 G t/31 must be constant -> not possible unless W1 = 0.
    #   => Instead absorb the t-dependence by *also* using the constant
    #      channel 16 (= -2G t/31) to cancel it:
    #      out = W0*ch0 + W1*ch1 + W16*ch16 + b
    #      region A: (W0+W1) x_t + (W0+W1)K + (W1*G - W16*2G) t/31 + b
    #      choose W16 = W1/2  -> t/31 term cancels exactly.
    #      => region A: gA x_t + gA K + b,  gA = W0+W1
    #      region B: W0 x_t + W0 K + b (ch1=0, ch16 = -2G t/31 is NOT zero!)
    #      careful: ch16 = -2G t/31 is active in BOTH regions (it is a plain
    #      constant channel, always present).  So region B also gets
    #      W16 * (-2G t/31) = -W1 G t/31 -> t-dependent.  Bad.
    #
    #   FIX: build the constant channel itself with the same gate so that
    #        it is zero in region B.  Use conv3 channel 5 = ch1 - ch3 which
    #        is -2G t/31 in region A and 0 in region B (because ch3 is
    #        active only for t>=16 ... wait ch3 = relu(-x_t + K + G t/31)
    #        is active for t>=16, so ch1-ch3 = (x_t+K-Gt/31) - (-x_t+K+Gt/31)
    #        = 2x_t - 2G t/31 in region A ... not what we want.
    #
    #   SIMPLER FIX: use conv2 channel 5 = ch1 - ch3 and note in region B
    #   ch1 = 0 and ch3 = -x_t + K + Gt/31  -> ch1-ch3 = x_t - K - Gt/31.
    #   Not a clean constant.  So instead build the constant channel from
    #   ch0 and ch2 only: conv2 channel 6 = ch0 + ch2 = 2K (constant, all t).
    #   And build a *gated* constant: we need something that is 1 in region A
    #   and 0 in region B.  ch1 - ch0 = -G t/31 in region A, 0 in region B
    #   -> that IS a gated constant!  (region A: -G t/31 ; region B: 0)
    #   So: conv2 channel 8 = ch1 - ch0  (gated t/31 constant)
    #       conv2 channel 9 = ch0 - ch2  = 2 x_t
    #       conv2 channel 10 = ch0 - ch1 = G t/31 in region A, 0 in region B
    #   Now build in conv3:
    #       ch3[0] = ch0 = x_t + K                       (all t)
    #       ch3[1] = ch1 = x_t + K - G t/31              (region A only)
    #       ch3[16] = conv2 ch8 = -G t/31 (region A), 0 (region B)
    #   readout: out = W0*ch3[0] + W1*ch3[1] + W16*ch3[16] + b
    #     region A: (W0+W1) x_t + (W0+W1)K + (-W1 G - W16 G) t/31 + b
    #                set W16 = -W1  -> t-term cancels
    #                => gA x_t + gA K + b,  gA = W0 + W1
    #     region B: W0 x_t + W0 K + b  (ch3[1]=0, ch3[16]=0)
    #                => gB x_t + gB K + b,  gB = W0
    #   => choose W0 = gB, W1 = gA - gB, W16 = -(gA-gB), b = oB - gB*K
    #      and region A offset oA = gA*K + b = gA K + oB - gB K.
    #   Perfect: we can realise ANY (gA,oA,gB,oB) with 2 regions.
    # ------------------------------------------------------------------

    # ---- choose gains from closed-form per-timestep regression --------
    # region A: t in [0,15], region B: t in [16,31]
    def fit_gain(mask):
        if mask.sum() < 2:
            return 0.0, 0.0
        xk = xf[mask]
        yk = yf[mask]
        # joint least squares for out = g*x + o
        A = np.array([[np.sum(xk * xk), np.sum(xk)],
                      [np.sum(xk), float(xk.shape[0])]])
        rhs = np.array([np.sum(xk * yk), np.sum(yk)])
        try:
            sol = np.linalg.solve(A + 1e-8 * np.eye(2), rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
        return float(sol[0]), float(sol[1])

    gA, oA = fit_gain(t <= 15)
    gB, oB = fit_gain(t >= 16)

    K_eff = K
    wout = np.zeros((1, 64, 3, 3))
    bout = np.zeros(1)
    W0 = gB
    W1 = gA - gB
    W16 = -(gA - gB)
    b_out = oB - gB * K_eff
    # conv3 channel 0 = conv2 channel 0 = ch0 = x_t + K
    wout[0, 0, 1, 1] = W0
    # conv3 channel 1 = conv2 channel 1 = ch1
    wout[0, 1, 1, 1] = W1
    # conv3 channel 16 = conv2 channel 8 = ch1 - ch0
    wout[0, 16, 1, 1] = W16

    # ------------------------------------------------------------------
    # 5) assemble the conv2 / conv3 channel plan consistently
    # ------------------------------------------------------------------
    w2 = np.zeros((64, 32, 3, 3))
    b2 = np.zeros(64)
    # identity copies of gate channels 0..3
    for c in range(4):
        w2[c, c, 1, 1] = 1.0
    # channel 4 = ch0 - ch2 = 2 x_t
    w2[4, 0, 1, 1] = 1.0
    w2[4, 2, 1, 1] = -1.0
    # channel 5 = ch0 - ch1  ( = G t/31 in region A, 0 in region B )
    w2[5, 0, 1, 1] = 1.0
    w2[5, 1, 1, 1] = -1.0
    # channel 6 = ch0 + ch2 = 2K (constant)
    w2[6, 0, 1, 1] = 1.0
    w2[6, 2, 1, 1] = 1.0
    # channel 7 = ch1 - ch0 ( = -G t/31 in region A, 0 in region B )
    w2[7, 1, 1, 1] = 1.0
    w2[7, 0, 1, 1] = -1.0
    # channel 8 = ch1 - ch0  (same as 7, kept for clarity) -> use 7
    # channel 9 = ch0 (copy of x_t + K)
    w2[9, 0, 1, 1] = 1.0
    # channel 10 = ch1 (copy)
    w2[10, 1, 1, 1] = 1.0

    w3 = np.zeros((64, 64, 3, 3))
    b3 = np.zeros(64)
    # conv3 keeps channels 0,1 (gate) and 7 (gated constant) and 4 (2 x_t)
    w3[0, 0, 1, 1] = 1.0
    w3[1, 1, 1, 1] = 1.0
    w3[7, 7, 1, 1] = 1.0
    w3[4, 4, 1, 1] = 1.0
    # shifted copies of 2*x_t for a 9x9 receptive field (unused by readout
    # but harmless; kept for potential local filtering)
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for i, (dy, dx) in enumerate(shifts):
        w3[8 + i, 4, 1 + dy, 1 + dx] = 1.0

    # readout weights (channels: 0 -> ch0, 1 -> ch1, 7 -> gated const)
    wout = np.zeros((1, 64, 3, 3))
    bout = np.zeros(1)
    wout[0, 0, 1, 1] = W0
    wout[0, 1, 1, 1] = W1
    wout[0, 7, 1, 1] = W16
    bout[0] = b_out

    # ------------------------------------------------------------------
    # 6) verify the constructed network on the training data
    # ------------------------------------------------------------------
    def forward(xin, tin, ww):
        Nn = xin.shape[0]
        u = np.empty((Nn, 2, 16, 16))
        u[:, 0:1] = xin
        u[:, 1:2] = (tin.astype(np.float64) / 31.0).reshape(Nn, 1, 1, 1) * np.ones((1, 1, 16, 16))
        h1 = np.maximum(conv2d(u, ww["conv1.weight"], ww["conv1.bias"]), 0.0)
        h2 = np.maximum(conv2d(h1, ww["conv2.weight"], ww["conv2.bias"]), 0.0)
        h3 = np.maximum(conv2d(h2, ww["conv3.weight"], ww["conv3.bias"]), 0.0)
        return conv2d(h3, ww["out.weight"], ww["out.bias"])

    new_w = {
        "conv1.weight": w1.astype(np.float32),
        "conv1.bias": b1.astype(np.float32),
        "conv2.weight": w2.astype(np.float32),
        "conv2.bias": b2.astype(np.float32),
        "conv3.weight": w3.astype(np.float32),
        "conv3.bias": b3.astype(np.float32),
        "out.weight": wout.astype(np.float32),
        "out.bias": bout.astype(np.float32),
    }

    pred = forward(x, t, new_w)
    mse = float(np.mean((pred - y) ** 2))

    # baseline: predict zero
    mse_zero = float(np.mean(y ** 2))

    # per-timestep mse of new model (for diagnostics)
    per_t = {}
    for k in range(32):
        m = t == k
        if m.sum() > 0:
            per_t[str(k)] = round(float(np.mean((pred[m] - y[m]) ** 2)), 4)

    # sanity: check keys/shapes preserved against original w
    for k in new_w:
        if k in w:
            assert new_w[k].shape == np.asarray(w[k]).shape, k

    diag = {
        "mse_new": round(mse, 5),
        "mse_zero_baseline": round(mse_zero, 5),
        "gainA": round(gA, 4),
        "gainB": round(gB, 4),
        "offA": round(oA, 4),
        "offB": round(oB, 4),
        "c_t_regression": [round(float(v), 4) for v in c_t],
        "per_t_mse": per_t,
    }

    # clip & ensure finite
    for k in new_w:
        new_w[k] = np.clip(np.nan_to_num(new_w[k], nan=0.0, posinf=0.0, neginf=0.0),
                           -1000.0, 1000.0).astype(np.float32)

    return {"weights": new_w, "diagnostics": diag}
