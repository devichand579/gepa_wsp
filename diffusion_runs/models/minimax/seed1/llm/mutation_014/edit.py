import numpy as np

def edit(w, f):
    rng = np.random.default_rng(0)

    # ---------------- helpers ----------------
    def conv2d(x, W, b, pad=1):
        # x: (N,C,H,W), W: (O,C,3,3) cross-correlation, zero padding
        N, C, H, Wd = x.shape
        O = W.shape[0]
        xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        # sliding windows -> (N,C,3,3,H,W)
        s = xp.strides
        win = np.lib.stride_tricks.as_strided(
            xp, shape=(N, C, 3, 3, H, Wd),
            strides=(s[0], s[1], s[2], s[3], s[2], s[3]), writeable=False)
        out = np.einsum('ncijkl,ocij->nokl', win, W, optimize=True)
        out = out + b.reshape(1, O, 1, 1)
        return out

    def fwd(x, t, W1, b1, W2, b2, W3, b3, Wo, bo):
        N = x.shape[0]
        u = (t.astype(np.float32) / 31.0).reshape(N, 1, 1, 1)
        inp = np.concatenate([x, np.broadcast_to(u, (N, 1, x.shape[2], x.shape[3]))], axis=1)
        h1 = np.maximum(conv2d(inp, W1, b1), 0.0)
        h2 = np.maximum(conv2d(h1, W2, b2), 0.0)
        h3 = np.maximum(conv2d(h2, W3, b3), 0.0)
        out = conv2d(h3, Wo, bo)
        return out, h3

    # ---------------- data ----------------
    x = np.asarray(f["x"], dtype=np.float32)
    t = np.asarray(f["t"]).astype(np.int64)
    y = np.asarray(f["y"], dtype=np.float32)
    N = x.shape[0]

    # ---------------- feature design (fixed, non-learned) ----------------
    # conv1 : 2 -> 32, 3x3
    #   ch0: random spatial filters on x_t  (multi-scale, 8 filters)
    #   ch1: random spatial filters on u    (timestep channel, 8 filters)
    #   ch2: relu(x_t - c) hinge features    (8 thresholds)
    #   ch3: relu(c - x_t) hinge features    (8 thresholds)
    W1 = np.zeros((32, 2, 3, 3), dtype=np.float32)
    b1 = np.zeros(32, dtype=np.float32)

    # spatial filters on x_t: random + multi-scale (identity-like, blur, edge)
    n_sp = 8
    for i in range(n_sp):
        if i == 0:
            k = np.zeros((3, 3), np.float32); k[1, 1] = 1.0
        elif i == 1:
            k = np.ones((3, 3), np.float32) / 9.0
        elif i == 2:
            k = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32) / 3.0
        elif i == 3:
            k = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], np.float32) / 3.0
        else:
            k = rng.standard_normal((3, 3)).astype(np.float32)
            k /= (np.abs(k).sum() + 1e-6)
        W1[i, 0] = k
        W1[n_sp + i, 1] = k
    for i in range(8):
        c = -1.5 + 3.0 * i / 7.0
        W1[16 + i, 0, 1, 1] = 1.0
        b1[16 + i] = -c
        W1[24 + i, 0, 1, 1] = -1.0
        b1[24 + i] = c

    # conv2 : 32 -> 64  (random mixing, keeps nonlinearity)
    W2 = (rng.standard_normal((64, 32, 3, 3)).astype(np.float32)
          * (1.0 / np.sqrt(32 * 9.0)))
    b2 = (rng.standard_normal(64).astype(np.float32) * 0.05)

    # conv3 : 64 -> 64  (random mixing)
    W3 = (rng.standard_normal((64, 64, 3, 3)).astype(np.float32)
          * (1.0 / np.sqrt(64 * 9.0)))
    b3 = (rng.standard_normal(64).astype(np.float32) * 0.05)

    # ---------------- forward to get features (parent-independent) ----------------
    _, h3 = fwd(x, t, W1, b1, W2, b2, W3, b3,
                np.zeros((1, 64, 3, 3), np.float32), np.zeros(1, np.float32))

    # ---------------- closed-form ridge readout (out layer) ----------------
    # design matrix: for each output pixel, 64*9 features + bias
    # build via im2col on h3
    Nn, C3, H, Wd = h3.shape
    hp = np.pad(h3, ((0, 0), (0, 0), (1, 1), (1, 1)))
    s = hp.strides
    cols = np.lib.stride_tricks.as_strided(
        hp, shape=(Nn, C3, 3, 3, H, Wd),
        strides=(s[0], s[1], s[2], s[3], s[2], s[3]), writeable=False)
    # (N, C, 3,3, H, W) -> (N, H, W, C*9)
    cols = np.transpose(cols, (0, 4, 5, 1, 2, 3)).reshape(Nn, H * Wd, C3 * 9)
    A = cols.reshape(Nn * H * Wd, C3 * 9).astype(np.float64)
    Y = y.reshape(Nn * H * Wd).astype(np.float64)

    # standardize columns
    mu = A.mean(axis=0)
    sd = A.std(axis=0)
    sd[sd < 1e-8] = 1.0
    As = (A - mu) / sd

    # ridge with several lambdas, pick best by MSE (closed form, no iterations)
    G = As.T @ As
    rhs = As.T @ Y
    tr = np.trace(G) / G.shape[0]
    best = None
    for lam in [1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
        Areg = G + lam * tr * np.eye(G.shape[0])
        try:
            beta = np.linalg.solve(Areg, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(Areg, rhs, rcond=None)[0]
        resid = As @ beta - Y
        mse = float(np.mean(resid * resid))
        if best is None or mse < best[0]:
            best = (mse, lam, beta)

    mse, lam, beta = best
    # back to original feature scaling
    w_out = (beta / sd).reshape(1, C3, 3, 3).astype(np.float32)
    b_out = float(-np.sum(mu / sd * beta))

    # ---------------- write weights ----------------
    out = {}
    for k, v in w.items():
        out[k] = np.array(v, dtype=np.float32, copy=True)
    out["conv1.weight"] = W1
    out["conv1.bias"] = b1
    out["conv2.weight"] = W2
    out["conv2.bias"] = b2
    out["conv3.weight"] = W3
    out["conv3.bias"] = b3
    out["out.weight"] = w_out
    out["out.bias"] = np.array([b_out], dtype=np.float32)

    # ---------------- diagnostics: honest train MSE with the final weights ----
    pred, _ = fwd(x, t, out["conv1.weight"], out["conv1.bias"],
                  out["conv2.weight"], out["conv2.bias"],
                  out["conv3.weight"], out["conv3.bias"],
                  out["out.weight"], out["out.bias"])
    tr_mse = float(np.mean((pred - y) ** 2))
    base_mse = float(np.mean((np.asarray(f["prediction"], np.float32) - y) ** 2))

    for k in out:
        out[k] = np.clip(np.nan_to_num(out[k], nan=0.0, posinf=0.0, neginf=0.0),
                         -1000.0, 1000.0).astype(np.float32)

    return {"weights": out,
            "diagnostics": {"train_mse": tr_mse, "parent_mse": base_mse,
                            "ridge_lambda": float(lam),
                            "out_bias": float(b_out)}}
