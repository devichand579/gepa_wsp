import numpy as np

def edit(w, f):
    # ------------------------------------------------------------------
    # 0. data
    # ------------------------------------------------------------------
    x = np.asarray(f["x"], dtype=np.float64)          # (N,1,16,16)
    y = np.asarray(f["y"], dtype=np.float64)          # (N,1,16,16) target eps
    t = np.asarray(f["t"]).astype(np.int64).ravel()   # (N,)
    ab = np.asarray(f["alphabar"], dtype=np.float64).ravel()  # (32,)

    N = x.shape[0]
    H = W = 16
    # per-pixel design: 3x3 spatial offsets of x  (+ intercept handled per-t)
    # build X: (N, 9) flattened neighbourhood of x_t
    xp = np.pad(x[:, 0], ((0, 0), (1, 1), (1, 1)))       # zero padding
    cols = []
    for dr in range(3):
        for dc in range(3):
            cols.append(xp[:, dr:dr + H, dc:dc + W].reshape(N, -1))
    Xn = np.stack(cols, axis=1)          # (N, 9, 256)
    Yn = y[:, 0].reshape(N, -1)         # (N, 256)

    # ------------------------------------------------------------------
    # 1. closed-form fit:  eps_hat = sum_k c_k * x_neighbour_k  +  d_t
    #    fitted per timestep group with a shared 9-tap spatial filter.
    #    Because the network can only realise a *fixed* spatial filter
    #    (t only enters as a spatially-constant map), we fit the shared
    #    filter on all data and let the per-timestep gain be piecewise
    #    linear in t.
    # ------------------------------------------------------------------
    # 1a. global shared filter + per-timestep intercept, joint least squares
    #     design rows: 9 filter taps (shared) + 32 one-hot timestep intercepts
    n_t = 32
    # accumulate normal equations
    # G[9,9] shared ; G[9,32] cross ; G[32,32] diag ; rhs
    # build big design per example: A (N, 9+32)
    # memory: N x 41 x 256 -> 2000*41*256*8 = 168MB  (ok) -> do in chunks
    G = np.zeros((9 + n_t, 9 + n_t))
    rhs = np.zeros((9 + n_t))
    chunk = 250
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        B = e - s
        # shared part: (B*256, 9)
        Xb = np.transpose(Xn[s:e], (0, 2, 1)).reshape(-1, 9)      # (B*256, 9)
        Yb = Yn[s:e].reshape(-1)
        # timestep one-hot per pixel row
        oh = np.zeros((B * 256, n_t))
        idx = np.repeat(t[s:e], 256)
        oh[np.arange(B * 256), idx] = 1.0
        D = np.concatenate([Xb, oh], axis=1)
        G += D.T @ D
        rhs += D.T @ Yb
    # ridge for stability
    lam = 1e-3
    G[np.arange(9 + n_t), np.arange(9 + n_t)] += lam * max(1.0, np.trace(G) / (9 + n_t))
    beta = np.linalg.solve(G, rhs)
    filt = beta[:9]            # shared 3x3 filter (row-major dr,dc)
    dts = beta[9:]             # per-timestep intercept

    # ------------------------------------------------------------------
    # 2. realise the model inside the fixed architecture
    #    out[p] = sum_k filt_k * x[p + off_k] + g(t)
    #    g(t) must be a spatially-constant map produced by the network.
    #    The network's only t-input is the broadcast channel t/31, so
    #    g(t) is realised as a piecewise-linear function of t built from
    #    ReLU ramps on the broadcast channel (t/31 in [0,1]).
    # ------------------------------------------------------------------
    # We realise g(t) exactly as a piecewise-linear function with knots at
    # a set of timesteps, using channels that encode relu ramps of t/31.
    # To keep the map affine we make every layer's pre-activation >= 0.
    #
    # Layer budget:
    #   conv1 (2->32):  ch0 = x_t (identity tap), ch1..ch31 = ramp channels
    #   conv2 (32->64): mixes
    #   conv3 (64->64): mixes
    #   out  (64->1):  reads the spatial part and the t-part
    #
    # We need conv1 to output:  x_t  (for the spatial path)  and  ramp_j(t)
    # ramp_j(t) = relu(t/31 - k_j)  -> piecewise linear in t, exactly.
    # conv1 has 32 output channels: ch0 = x_t, ch1..ch31 = ramps with
    # knots k_j = j/31 for j=1..31 (t/31 - j/31).
    # To keep conv1 pre-activations non-negative we shift x_t channel by a
    # constant C and subtract it later in the out layer (out has no ReLU).
    #
    # conv2 must keep pre-activation non-negative: it computes
    #   ch0: x_t + C   (identity, from conv1 ch0)
    #   ch1..: ramp channels (already >=0) combined with positive weights
    # conv3 similar: keep everything non-negative by adding a positive
    # constant channel and using non-negative mixing weights.
    # Finally out reads:
    #   - spatial: sum_k filt_k * (x_t shifted)  -> uses conv3 ch0 with
    #     a 3x3 kernel equal to filt (so the spatial filter is realised)
    #   - t-part:  combination of ramp channels -> g(t) (piecewise linear)
    #   - constant corrections to cancel the shifts.
    #
    # Because conv2/conv3 are 3x3 convs, the spatial filter is realised by
    # conv3 ch0 = conv2 ch0 (identity 3x3) and out kernel = filt.
    # The t-part must be spatially constant: conv1 ramp channels are
    # spatially constant maps; conv2/conv3 with delta kernels keep them
    # constant; out with delta kernel keeps constant.

    C = 4.0          # shift for x_t channel (x_t in ~[-3,3])
    KNOT = 32

    # ---- conv1 -------------------------------------------------------
    c1w = np.zeros((32, 2, 3, 3))
    c1b = np.zeros(32)
    # ch0: x_t + C  (center tap on channel 0)
    c1w[0, 0, 1, 1] = 1.0
    c1b[0] = C
    # ch1..ch31: ramp_j(t) = relu(t/31 - j/31)  -> conv on t channel
    #   kernel center tap on channel 1 with weight 1/31, bias -j/31
    for j in range(1, 32):
        c1w[j, 1, 1, 1] = 1.0 / 31.0
        c1b[j] = -float(j) / 31.0
    # NOTE: ramp_j pre-activation = (t - j)/31 which is negative for t<j;
    #       relu makes it 0 -> fine, ramp channels are >= 0 after ReLU.

    # ---- conv2 -------------------------------------------------------
    # ch0: identity copy of conv1 ch0 (x_t + C)  -> keeps spatial path
    # ch1..ch31: copy of conv1 ramp channels (>=0)  -> t path
    # ch32: constant 1 channel (for bias bookkeeping)
    # We use 64 channels; extra channels unused (zero weights, bias 0 ->
    # relu(0)=0, harmless).
    c2w = np.zeros((64, 32, 3, 3))
    c2b = np.zeros(64)
    c2w[0, 0, 1, 1] = 1.0
    for j in range(1, 32):
        c2w[j, j, 1, 1] = 1.0
    c2w[32, 0, 1, 1] = 0.0
    c2b[32] = 1.0        # constant-1 channel (pre-activation = 1 > 0)
    # conv2 pre-activations: ch0 = x_t + C >= 0 (need C >= max|x_t|),
    #   ch1..31 = ramps >= 0, ch32 = 1.

    # ---- conv3 -------------------------------------------------------
    # ch0: identity of conv2 ch0 (x_t + C)
    # ch1..31: identity of conv2 ramp channels
    # ch32: identity of constant channel
    c3w = np.zeros((64, 64, 3, 3))
    c3b = np.zeros(64)
    c3w[0, 0, 1, 1] = 1.0
    for j in range(1, 32):
        c3w[j, j, 1, 1] = 1.0
    c3w[32, 32, 1, 1] = 1.0

    # ---- out ---------------------------------------------------------
    # spatial part: out = sum_k filt_k * (x_t + C) at offset k
    #   -> kernel on conv3 ch0 = filt (3x3, row-major dr,dc)
    # t-part: g(t) = sum_j a_j * ramp_j(t)  (piecewise linear, exact)
    # constant: bias cancels the C*sum(filt) term and adds d_t part.
    # We split d_t into a constant part (from bias) plus ramp combination.
    ow = np.zeros((1, 64, 3, 3))
    ob = np.zeros(1)
    for dr in range(3):
        for dc in range(3):
            ow[0, 0, dr, dc] = filt[dr * 3 + dc]
    # t-part: represent d(t) on the 32 timesteps as
    #   d(t) = d0 + sum_{j=1}^{31} a_j * ramp_j(t)
    # ramp_j(t) = max(0, (t-j)/31)  -> for t in 0..31.
    R = np.zeros((32, 32))
    for tt in range(32):
        for j in range(1, 32):
            R[tt, j] = max(0.0, (tt - j) / 31.0)
    # least squares for (d0, a_1..a_31) to match dts
    A = np.concatenate([np.ones((32, 1)), R[:, 1:]], axis=1)
    coef, *_ = np.linalg.lstsq(A, dts, rcond=None)
    d0 = coef[0]
    av = coef[1:]
    for j in range(1, 32):
        ow[0, j, 1, 1] += av[j - 1]
    # constant correction: spatial path contributes C * sum(filt) everywhere
    ob[0] = d0 - C * float(np.sum(filt))

    # ------------------------------------------------------------------
    # 3. safety: verify the realised network reproduces the intended map
    # ------------------------------------------------------------------
    def conv3x3(a, b):   # a: (N,C,H,W) input, b: (O,C,3,3) weight
        N_, C_, H_, W_ = a.shape
        O = b.shape[0]
        ap = np.pad(a, ((0, 0), (0, 0), (1, 1), (1, 1)))
        out = np.zeros((N_, O, H_, W_))
        for dr in range(3):
            for dc in range(3):
                sl = ap[:, :, dr:dr + H_, dc:dc + W_]
                out += np.einsum('nchw,oc->nohw', sl, b[:, :, dr, dc])
        return out

    def forward(xin, tin, W1, b1, W2, b2, W3, b3, Wo, bo):
        N_ = xin.shape[0]
        tt = (tin.astype(np.float64) / 31.0).reshape(N_, 1, 1, 1)
        inp = np.concatenate([xin, np.broadcast_to(tt, (N_, 1, H, W))], axis=1)
        h = np.maximum(conv3x3(inp, W1) + b1[None, :, None, None], 0.0)
        h = np.maximum(conv3x3(h, W2) + b2[None, :, None, None], 0.0)
        h = np.maximum(conv3x3(h, W3) + b3[None, :, None, None], 0.0)
        return conv3x3(h, Wo) + bo[None, :, None, None]

    # check on a subset
    idx = np.arange(0, N, max(1, N // 200))
    W1 = c1w; b1 = c1b; W2 = c2w; b2 = c2b; W3 = c3w; b3 = c3b
    Wo = ow; bo = ob
    pred = forward(x[idx], t[idx], W1, b1, W2, b2, W3, b3, Wo, bo)
    mse_new = float(np.mean((pred - y[idx]) ** 2))
    mse_old = float(np.mean((np.asarray(f["prediction"], dtype=np.float64)[idx] - y[idx]) ** 2))

    # also compute the "ideal linear" mse for reference on the same subset
    diag = {
        "mse_before_subset": mse_old,
        "mse_after_subset": mse_new,
        "shared_filter": [round(float(v), 6) for v in filt],
        "per_t_intercept": [round(float(v), 5) for v in dts],
        "alphabar": [round(float(v), 6) for v in ab],
    }

    # ------------------------------------------------------------------
    # 4. write back
    # ------------------------------------------------------------------
    def clamp(a):
        a = np.asarray(a, dtype=np.float32)
        return np.clip(a, -1000.0, 1000.0)

    w["conv1.weight"] = clamp(c1w)
    w["conv1.bias"] = clamp(c1b)
    w["conv2.weight"] = clamp(c2w)
    w["conv2.bias"] = clamp(c2b)
    w["conv3.weight"] = clamp(c3w)
    w["conv3.bias"] = clamp(c3b)
    w["out.weight"] = clamp(ow)
    w["out.bias"] = clamp(ob)

    return {"weights": w, "diagnostics": diag}
