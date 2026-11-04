import numpy as np

def edit(w, f):
    # ---------------- data ----------------
    x = np.asarray(f["x"], dtype=np.float64)          # [N,1,16,16]
    y = np.asarray(f["y"], dtype=np.float64)          # [N,1,16,16] target eps
    t = np.asarray(f["t"], dtype=np.int64).ravel()     # [N]
    N = x.shape[0]
    H = W = 16

    # ---------------- forward helpers (numpy conv, cross-correlation) --------
    def conv(a, wt, b):
        # a:[N,C,H,W], wt:[O,C,3,3], b:[O]
        Nn, C = a.shape[0], a.shape[1]
        O = wt.shape[0]
        ap = np.zeros((Nn, C, H + 2, W + 2), dtype=np.float64)
        ap[:, :, 1:-1, 1:-1] = a
        out = np.zeros((Nn, O, H, W), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                out += np.tensordot(ap[:, :, i:i + H, j:j + W], wt[:, :, i, j], axes=([1], [1]))
        out += b.reshape(1, O, 1, 1)
        return out

    def forward(xin, tt, Wd):
        tchan = (tt.astype(np.float64) / 31.0).reshape(-1, 1, 1, 1)
        inp = np.concatenate([xin, np.broadcast_to(tchan, (xin.shape[0], 1, H, W))], axis=1)
        h = np.maximum(conv(inp, Wd["conv1.weight"], Wd["conv1.bias"]), 0.0)
        h = np.maximum(conv(h, Wd["conv2.weight"], Wd["conv2.bias"]), 0.0)
        h = np.maximum(conv(h, Wd["conv3.weight"], Wd["conv3.bias"]), 0.0)
        return conv(h, Wd["out.weight"], Wd["out.bias"])

    # ---------------- design: exactly affine  eps_hat = A(t)*x + B(t) --------
    # x-path: conv1 ch0 = relu(x + BIG), ch1 = relu(-x + BIG)  -> ch0-ch1 = x
    # t-path: conv1 ch2..31 = relu(ramp) gates, piecewise-linear in t/31
    # conv2/conv3: identity (center taps) on the 32 conv1 channels
    # out: out[0,0] = +1 center tap (x channel), out[0,1] = -1 center tap
    #       out[0,2+k] = wgt_k (gate k) ; out.bias = 0
    BIG = 20.0
    K = 30                      # number of t-gates (channels 2..31 of conv1)
    u = t.astype(np.float64) / 31.0

    # ---- fit A(t), B(t) per timestep by closed-form ridge least squares ----
    # model: eps_hat[p] = A_t * x[p] + B_t
    A = np.zeros(32); B = np.zeros(32); mse_fit = np.zeros(32)
    for tt in range(32):
        m = (t == tt)
        if m.sum() < 2:
            continue
        xs = x[m, 0].reshape(m.sum(), -1)
        ys = y[m, 0].reshape(m.sum(), -1)
        sx = xs.sum(1); sy = ys.sum(1)
        n = xs.shape[1]
        sxx = (xs * xs).sum(1)
        sxy = (xs * ys).sum(1)
        Sxx = sxx.sum(); Sxy = sxy.sum(); Sx = sx.sum(); Sy = sy.sum(); Nn = m.sum()
        # normal equations for [A,B]
        det = Sxx * Nn - Sx * Sx
        if abs(det) < 1e-12:
            A[tt] = 0.0; B[tt] = sy.mean()
        else:
            A[tt] = (Sxy * Nn - Sx * Sy) / det
            B[tt] = (Sxx * Sy - Sx * Sxy) / det
        pred = A[tt] * xs + B[tt]
        mse_fit[tt] = float(((pred - ys) ** 2).mean())

    # ---- realise A(t) as piecewise-linear function of u via gates ----
    # gate k (k=0..K-1) is a "hat" over u-interval [k/K, (k+1)/K]:
    #   g_k(u) = relu( (u - k/K) * S ) - relu( (u - (k+1)/K) * S )
    # with S large enough that the ramps are steep but still exactly linear pieces.
    # conv1 weight on t-channel for channel 2+k: +S at (0,0) and -S at (2,2)
    # bias for channel 2+k: -S*k/K  and +S*(k+1)/K
    # Then A(u) = sum_k a_k g_k(u) with a_k = A at the interval midpoint.
    # We choose a_k so that A(u) is the piecewise-linear interpolation of A(t).
    S = 200.0
    # piecewise-linear interpolation of A over u grid (32 knots)
    knots = np.arange(32) / 31.0
    def interp_A(uu):
        idx = np.clip(np.searchsorted(knots, uu, side="right") - 1, 0, 30)
        f = (uu - knots[idx]) / (knots[idx + 1] - knots[idx])
        return A[idx] * (1 - f) + A[idx + 1] * f
    # gate weights: a_k = A evaluated at the interval midpoint of gate k
    mids = (np.arange(K) + 0.5) / K
    a_k = interp_A(mids)
    # B(u) similarly, but we only have 30 gates + 2 x-channels; use gates for B too
    b_k = interp_A(mids) * 0.0
    # B: piecewise-linear via same gates (use separate gate set is impossible; reuse:
    # B(u) = sum_k b_k g_k(u) with b_k = B at midpoint) -> but we need both A and B
    # from the same gates; instead use the *offset* channel trick:
    # out = A(u)*x + B(u) requires two independent t-functions.
    # Use gates for A (30 channels) and encode B via the x-channel biases:
    #   ch0 = relu(x + BIG + c0(u)) is not possible (bias is constant).
    # => Use 15 gates for A and 15 gates for B (K=30 -> 15+15).
    KA = 15
    KB = 15
    S = 400.0
    def hat_weights(start, count):
        # returns list of (lo, hi) intervals in u for the hat functions
        return [(start + i) / count for i in range(count)]

    # A gates: intervals over u in [0,1] split into KA pieces
    a_vals = np.zeros(KA); b_vals = np.zeros(KB)
    a_mids = (np.arange(KA) + 0.5) / KA
    b_mids = (np.arange(KB) + 0.5) / KB
    a_vals = interp_A(a_mids)
    def interp_B(uu):
        idx = np.clip(np.searchsorted(knots, uu, side="right") - 1, 0, 30)
        f = (uu - knots[idx]) / (knots[idx + 1] - knots[idx])
        return B[idx] * (1 - f) + B[idx + 1] * f
    b_vals = interp_B(b_mids)

    # ---- build weights ----
    Wd = {k: np.array(v, dtype=np.float64) for k, v in w.items()}
    c1w = np.zeros((32, 2, 3, 3)); c1b = np.zeros(32)
    # x channels
    c1w[0, 0, 1, 1] = 1.0;  c1b[0] = BIG
    c1w[1, 0, 1, 1] = -1.0; c1b[1] = BIG
    # gate channels 2..31
    ch = 2
    # A gates: hat over u
    for k in range(KA):
        lo = k / KA; hi = (k + 1) / KA
        # relu(S*(u-lo)) - relu(S*(u-hi))
        c1w[ch, 1, 1, 1] += S;  c1b[ch] += -S * lo
        c1w[ch, 1, 1, 1] += -S; c1b[ch] += S * hi
        ch += 1
    # B gates
    for k in range(KB):
        lo = k / KB; hi = (k + 1) / KB
        c1w[ch, 1, 1, 1] += S;  c1b[ch] += -S * lo
        c1w[ch, 1, 1, 1] += -S; c1b[ch] += S * hi
        ch += 1
    assert ch == 32

    # conv2: identity center taps on conv1 channels 0..31 -> conv2 out ch j = conv1 ch j
    c2w = np.zeros((64, 32, 3, 3)); c2b = np.zeros(64)
    for j in range(32):
        c2w[j, j, 1, 1] = 1.0
    # conv3: identity center taps on conv2 channels 0..31
    c3w = np.zeros((64, 64, 3, 3)); c3b = np.zeros(64)
    for j in range(32):
        c3w[j, j, 1, 1] = 1.0
    # out: eps = A(u)*x + B(u)
    ow = np.zeros((1, 64, 3, 3)); ob = np.zeros(1)
    ow[0, 0, 1, 1] = 1.0     # +x
    ow[0, 1, 1, 1] = -1.0    # -(-x)  => +x  (ch0-ch1 = x)
    for k in range(KA):
        ow[0, 2 + k, 1, 1] = a_vals[k]
    for k in range(KB):
        ow[0, 2 + KA + k, 1, 1] = b_vals[k]

    Wd["conv1.weight"] = c1w; Wd["conv1.bias"] = c1b
    Wd["conv2.weight"] = c2w; Wd["conv2.bias"] = c2b
    Wd["conv3.weight"] = c3w; Wd["conv3.bias"] = c3b
    Wd["out.weight"] = ow;  Wd["out.bias"] = ob

    # ---- verify with the real forward pass ----
    pred = forward(x, t, Wd)
    mse = float(((pred - y) ** 2).mean())
    # per timestep
    pt = []
    for tt in range(32):
        m = (t == tt)
        if m.sum():
            pt.append(float(((pred[m] - y[m]) ** 2).mean()))
        else:
            pt.append(None)

    # ---- final safety: shapes, finiteness, magnitude ----
    out = {}
    for k, v in w.items():
        arr = np.asarray(Wd[k], dtype=np.float32)
        arr = np.clip(arr, -1000.0, 1000.0)
        arr = np.where(np.isfinite(arr), arr, np.float32(0.0))
        out[k] = arr.astype(np.float32)

    diag = {
        "mse_train": round(mse, 5),
        "mse_parent_ref": 0.5147,
        "A_fit": [round(float(v), 4) for v in A],
        "B_fit": [round(float(v), 4) for v in B],
        "per_t_mse": [None if v is None else round(v, 4) for v in pt],
        "n": int(N),
    }
    return {"weights": out, "diagnostics": diag}
