import numpy as np


def edit(w, f):
    # ------------------------------------------------------------------
    # 0. basic data
    # ------------------------------------------------------------------
    x = np.asarray(f["x"], dtype=np.float64)            # [N,1,16,16]
    y = np.asarray(f["y"], dtype=np.float64)            # [N,1,16,16]
    clean = np.asarray(f["clean"], dtype=np.float64)    # [N,1,16,16]
    t = np.asarray(f["t"]).astype(np.int64).ravel()     # [N]
    abar = np.asarray(f["alphabar"], dtype=np.float64).ravel()  # [32]

    N = x.shape[0]
    H = W_ = 16
    P = H * W_

    # ------------------------------------------------------------------
    # 1. clean-image statistics: mean image + power spectrum
    # ------------------------------------------------------------------
    c = clean.reshape(N, P)
    mu = c.mean(axis=0)                       # [P] mean clean image
    dc = c - mu
    # power spectrum of the fluctuating part (per frequency, averaged)
    F = np.fft.fft2(dc.reshape(N, H, W_), axes=(1, 2)) / P
    S = (np.abs(F) ** 2).mean(axis=0).ravel()  # [P]
    S = np.maximum(S, 1e-8)

    # ------------------------------------------------------------------
    # 2. per-timestep Wiener filter for epsilon
    #    eps_hat(k) = sqrt(1-a) * X(k) / (a*S(k) + 1-a)
    # ------------------------------------------------------------------
    a = abar[t]                               # [N]
    one_m_a = np.maximum(1.0 - a, 1e-8)
    # filter for each example (depends on its timestep)
    G = np.sqrt(one_m_a)[:, None] / (a[:, None] * S[None, :] + one_m_a[:, None])  # [N,P]

    # ------------------------------------------------------------------
    # 3. realise the linear map inside the fixed architecture
    #    conv1 : identity on x_t (channel 0), zero on t-channel, zero bias
    #    conv2 : identity on conv1 output (channel 0), zero bias
    #    conv3 : identity on conv2 output (channel 0), zero bias
    #    out   : fitted by closed-form ridge regression
    #    => composite map is exactly linear in x_t, with t entering only
    #       through the fitted readout (per-example t is not available at
    #       the readout, so the t-dependence is carried by the *features*
    #       we build: we therefore give conv1 a t-dependent channel too).
    # ------------------------------------------------------------------
    # We need the readout to know t.  conv1 channel 1 receives t/31, so we
    # build features  [x_t patch (9x9) , t/31 * x_t patch (9x9) , 1]
    # and realise them with conv1/conv2/conv3 as identity channels.
    # conv1 out channels: 0..15  -> x_t shifted copies (identity)
    #                     16..31 -> (t/31) * x_t shifted copies
    # conv2/conv3 keep channels 0..31 as identity (channel-wise), so the
    # final feature map has 32 channels: 16 x-shifts + 16 t-scaled shifts.
    # The readout then implements  eps_hat = sum_j c_j(t) x_t(p+j) + d(t).
    # ------------------------------------------------------------------
    # Build the analytic target for the readout:
    #   eps_hat = Wiener filter applied to x_t (circular convolution).
    Xf = np.fft.fft2(x.reshape(N, H, W_), axes=(1, 2))
    eps_hat = np.real(np.fft.ifft2(G[:, :, None] * Xf, axes=(1, 2))).reshape(N, P)

    # ------------------------------------------------------------------
    # 4. construct the feature tensor that the architecture will produce
    #    and fit the readout with closed-form ridge least squares.
    # ------------------------------------------------------------------
    # feature channels (32) at each pixel p:
    #   ch 0..15  : x_t(p + j) for j in a 4x4 neighbourhood (offsets)
    #   ch 16..31 : (t/31) * x_t(p + j)  (same offsets)
    # The offsets must be reachable by the receptive field (9x9) -> fine.
    offs = [(dy, dx) for dy in range(4) for dx in range(4)]   # 16 offsets
    xp = np.pad(x.reshape(N, H, W_), ((0, 0), (4, 4), (4, 4)), mode="wrap")
    feats = np.empty((N, 32, H, W_), dtype=np.float64)
    for k, (dy, dx) in enumerate(offs):
        feats[:, k] = xp[:, dy:dy + H, dx:dx + W_]
    tv = (t / 31.0)[:, None, None]
    feats[:, 16:32] = feats[:, 16:32] * 0.0
    for k in range(16):
        feats[:, 16 + k] = feats[:, k] * tv[:, :, 0]

    # design matrix rows: (n, p) -> 32 features + intercept
    A = feats.transpose(0, 2, 3, 1).reshape(N * P, 32)
    A = np.concatenate([A, np.ones((A.shape[0], 1))], axis=1)
    b = eps_hat.reshape(-1)

    # ------------------------------------------------------------------
    # 5. closed-form ridge (normal equations with small Tikhonov term)
    # ------------------------------------------------------------------
    Gm = A.T @ A
    rhs = A.T @ b
    lam = 1e-3 * np.trace(Gm) / Gm.shape[0]
    Gm[np.diag_indices_from(Gm)] += lam
    theta = np.linalg.solve(Gm, rhs)          # [33]

    # ------------------------------------------------------------------
    # 6. build the network weights realising those features + readout
    # ------------------------------------------------------------------
    # conv1: [32,2,3,3]
    #  ch k (0..15)  : copy of x_t with offset (dy,dx) -> kernel centre at
    #                   (1-dy, 1-dx) so that conv output at p equals
    #                   x_t(p + (dy,dx))  (cross-correlation convention:
    #                   out[p] = sum_{a,b} w[a,b] * in[p + a - 1])
    #  ch 16+k       : same kernel but on the t/31 channel (input channel 1)
    c1w = np.zeros((32, 2, 3, 3), dtype=np.float64)
    for k, (dy, dx) in enumerate(offs):
        c1w[k, 0, 1 - dy, 1 - dx] = 1.0
        c1w[16 + k, 1, 1 - dy, 1 - dx] = 1.0
    c1b = np.zeros(32, dtype=np.float64)

    # conv2 / conv3: identity on the first 32 channels, zero on the rest
    c2w = np.zeros((64, 32, 3, 3), dtype=np.float64)
    for k in range(32):
        c2w[k, k, 1, 1] = 1.0
    c2b = np.zeros(64, dtype=np.float64)

    c3w = np.zeros((64, 64, 3, 3), dtype=np.float64)
    for k in range(32):
        c3w[k, k, 1, 1] = 1.0
    c3b = np.zeros(64, dtype=np.float64)

    # out layer: [1,64,3,3]  -> readout over the 32 useful feature channels
    # with the fitted coefficients, placed at the centre tap.
    ow = np.zeros((1, 64, 3, 3), dtype=np.float64)
    for k in range(32):
        ow[0, k, 1, 1] = theta[k]
    ob = np.array([theta[32]], dtype=np.float64)

    # ------------------------------------------------------------------
    # 7. sanity check: simulate the fixed forward pass and measure MSE
    # ------------------------------------------------------------------
    def conv2d(inp, wt, bi):
        # inp [N,C,H,W], wt [O,C,3,3] -> [N,O,H,W], cross-correlation, pad 1
        Nn, C, HH, WW = inp.shape
        O = wt.shape[0]
        ip = np.pad(inp, ((0, 0), (0, 0), (1, 1), (1, 1)))
        out = np.zeros((Nn, O, HH, WW), dtype=np.float64)
        for a in range(3):
            for b in range(3):
                out += np.einsum('nchw,oc->nohw', ip[:, :, a:a + HH, b:b + WW], wt[:, :, a, b])
        return out + bi[None, :, None, None]

    def forward(xin, tvec, c1w_, c1b_, c2w_, c2b_, c3w_, c3b_, ow_, ob_):
        Nn = xin.shape[0]
        tv = (tvec / 31.0).astype(np.float64)[:, None, None, None]
        inp = np.concatenate([xin, np.broadcast_to(tv, (Nn, 1, H, W_))], axis=1)
        h1 = np.maximum(conv2d(inp, c1w_, c1b_), 0.0)
        h2 = np.maximum(conv2d(h1, c2w_, c2b_), 0.0)
        h3 = np.maximum(conv2d(h2, c3w_, c3b_), 0.0)
        o = conv2d(h3, ow_, ob_)
        return o

    pred = forward(x, t, c1w, c1b, c2w, c2b, c3w, c3b, ow, ob)
    mse_new = float(np.mean((pred - y) ** 2))
    mse_old = float(np.mean((np.asarray(f["prediction"], dtype=np.float64) - y) ** 2))
    mse_wiener = float(np.mean((eps_hat.reshape(N, 1, H, W_) - y) ** 2))

    # ------------------------------------------------------------------
    # 8. safety: if the closed-form construction does not beat the
    #    current network, fall back to the analytic Wiener target only
    #    (still a strict improvement over the near-zero current net).
    # ------------------------------------------------------------------
    if not np.isfinite(mse_new) or mse_new > 0.999 * min(mse_old, 1.0):
        # fallback: use the Wiener filter directly, fitted as a linear map
        # from the 16 x-offsets (no t-scaled channel) -> keep same code path
        A2 = feats[:, :16].transpose(0, 2, 3, 1).reshape(N * P, 16)
        A2 = np.concatenate([A2, np.ones((A2.shape[0], 1))], axis=1)
        G2 = A2.T @ A2
        r2 = A2.T @ b
        lam2 = 1e-3 * np.trace(G2) / G2.shape[0]
        G2[np.diag_indices_from(G2)] += lam2
        th2 = np.linalg.solve(G2, r2)
        ow2 = np.zeros((1, 64, 3, 3), dtype=np.float64)
        for k in range(16):
            ow2[0, k, 1, 1] = th2[k]
        ob2 = np.array([th2[16]], dtype=np.float64)
        pred2 = forward(x, t, c1w, c1b, c2w, c2b, c3w, c3b, ow2, ob2)
        mse2 = float(np.mean((pred2 - y) ** 2))
        if np.isfinite(mse2) and mse2 < mse_new:
            ow, ob = ow2, ob2
            mse_new = mse2

    # ------------------------------------------------------------------
    # 9. write out (float32, clipped, keys preserved)
    # ------------------------------------------------------------------
    out = {}
    out["conv1.weight"] = np.clip(c1w, -1000, 1000).astype(np.float32)
    out["conv1.bias"] = np.clip(c1b, -1000, 1000).astype(np.float32)
    out["conv2.weight"] = np.clip(c2w, -1000, 1000).astype(np.float32)
    out["conv2.bias"] = np.clip(c2b, -1000, 1000).astype(np.float32)
    out["conv3.weight"] = np.clip(c3w, -1000, 1000).astype(np.float32)
    out["conv3.bias"] = np.clip(c3b, -1000, 1000).astype(np.float32)
    out["out.weight"] = np.clip(ow, -1000, 1000).astype(np.float32)
    out["out.bias"] = np.clip(ob, -1000, 1000).astype(np.float32)

    for k, v in w.items():
        if k not in out:
            out[k] = np.asarray(v, dtype=np.float32)

    diagnostics = {
        "mse_before": round(mse_old, 6),
        "mse_after": round(mse_new, 6),
        "mse_wiener_analytic": round(mse_wiener, 6),
        "alphabar_min": float(abar.min()),
        "alphabar_max": float(abar.max()),
        "clean_pixel_var": float(c.var()),
        "n_train": int(N),
    }
    return {"weights": out, "diagnostics": diagnostics}
