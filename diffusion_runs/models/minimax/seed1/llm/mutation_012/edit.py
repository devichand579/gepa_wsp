import numpy as np

def edit(w, f):
    # ---------------- helpers ----------------
    def conv2d(x, W, b):
        # x: [N,C,H,W], W: [O,C,3,3] (cross-correlation), pad 1
        N, C, H, Wd = x.shape
        O = W.shape[0]
        xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
        out = np.zeros((N, O, H, Wd), dtype=np.float64)
        for ky in range(3):
            for kx in range(3):
                out += np.tensordot(xp[:, :, ky:ky + H, kx:kx + Wd], W[:, :, ky, kx], axes=([1], [1])).transpose(1, 0, 2, 3)
        return out + b.reshape(1, O, 1, 1)

    def forward(x, t, W):
        # x: [N,1,16,16], t: [N] in 0..31
        N = x.shape[0]
        tt = (t.astype(np.float64) / 31.0).reshape(N, 1, 1, 1)
        inp = np.concatenate([x.astype(np.float64), np.broadcast_to(tt, (N, 1, 16, 16))], axis=1)
        h = np.maximum(conv2d(inp, W['conv1.weight'], W['conv1.bias']), 0.0)
        h = np.maximum(conv2d(h, W['conv2.weight'], W['conv2.bias']), 0.0)
        h = np.maximum(conv2d(h, W['conv3.weight'], W['conv3.bias']), 0.0)
        return conv2d(h, W['out.weight'], W['out.bias'])

    def ridge_solve(A, y, lam):
        # A: [M,P] design (no intercept), y: [M]
        d = A.shape[1]
        G = A.T @ A
        G[np.diag_indices(d)] += lam * (np.trace(G) / d + 1e-12)
        b = A.T @ y
        try:
            return np.linalg.solve(G, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(G, b, rcond=None)[0]

    # ---------------- data ----------------
    x = np.asarray(f['x'], dtype=np.float64)          # [N,1,16,16]
    y = np.asarray(f['y'], dtype=np.float64)          # [N,1,16,16]
    t = np.asarray(f['t']).astype(np.int64)
    clean = np.asarray(f['clean'], dtype=np.float64)
    N = x.shape[0]
    abar = np.asarray(f['alphabar'], dtype=np.float64)  # [32]

    # ---------------- analytic Wiener predictor of eps ----------------
    # eps = (x_t - sqrt(abar) x0)/sqrt(1-abar); regress x0 on x_t per timestep
    c_t = np.zeros(32); d_t = np.zeros(32)
    for k in range(32):
        m = t == k
        if m.sum() < 8:
            continue
        Xk = x[m].reshape(m.sum(), -1)
        Yk = clean[m].reshape(m.sum(), -1)
        Xc = Xk - Xk.mean(0)
        Yc = Yk - Yk.mean(0)
        Gk = Xc.T @ Xc
        Gk[np.diag_indices(256)] += 1e-3 * (np.trace(Gk) / 256 + 1e-12)
        A = np.linalg.solve(Gk, Yc.T)
        a = A.mean(1)                      # isotropic (per-pixel) gain
        b = (Yk.mean(0) - Xk.mean(0) @ A).mean()
        s = np.sqrt(max(abar[k], 1e-12))
        c_t[k] = (1.0 - s * a) / np.sqrt(max(1.0 - abar[k], 1e-12))
        d_t[k] = -s * b / np.sqrt(max(1.0 - abar[k], 1e-12))
    c_t = np.clip(c_t, -50.0, 50.0)
    d_t = np.clip(d_t, -50.0, 50.0)

    # ---------------- design conv1 as t-gated linear probes ----------------
    # channel k: ReLU(alpha_k * x_center + beta_k * t/31 + gamma_k)
    # thresholds spread over t so that the active set changes with t.
    K = 32
    thr = np.arange(K) * (31.0 / (K - 1))          # 0..31
    alpha = np.ones(K)
    beta = np.full(K, 6.0)
    gamma = np.full(K, 6.0) - 6.0 * thr / 31.0     # so that feature ~ on for t>thr
    # make sure the probe is fully on for t>thr (over x in [-4,4]) and fully off for t<thr
    gamma = 6.0 - 6.0 * thr / 31.0 - 0.0
    gamma = np.clip(gamma, -6.0, 6.0)
    # widen margin: use beta=12, gamma=12-12*thr/31
    beta = np.full(K, 12.0)
    gamma = 12.0 - 12.0 * thr / 31.0

    W1 = np.zeros((32, 2, 3, 3), dtype=np.float64)
    b1 = np.zeros(32, dtype=np.float64)
    for k in range(K):
        W1[k, 0, 1, 1] = alpha[k]
        W1[k, 1, 1, 1] = beta[k]
        b1[k] = gamma[k]
    # conv2 / conv3 : near-identity (center tap = 1)
    W2 = np.zeros((64, 32, 3, 3), dtype=np.float64); W2[:, :, 1, 1] = 1.0
    b2 = np.zeros(64)
    W3 = np.zeros((64, 64, 3, 3), dtype=np.float64); W3[:, :, 1, 1] = 1.0
    b3 = np.zeros(64)

    Wnew = {
        'conv1.weight': W1, 'conv1.bias': b1,
        'conv2.weight': W2, 'conv2.bias': b2,
        'conv3.weight': W3, 'conv3.bias': b3,
        'out.weight': np.zeros((1, 64, 3, 3)), 'out.bias': np.zeros(1)
    }

    # ---------------- forward with new trunk -> features ----------------
    N = x.shape[0]
    tt = (t.astype(np.float64) / 31.0).reshape(N, 1, 1, 1)
    inp = np.concatenate([x, np.broadcast_to(tt, (N, 1, 16, 16))], axis=1)
    h1 = np.maximum(conv2d(inp, W1, b1), 0.0)
    h2 = np.maximum(conv2d(h1, W2, b2), 0.0)
    h3 = np.maximum(conv2d(h2, W3, b3), 0.0)     # [N,64,16,16] features

    # ---------------- ridge fit of the read-out (center tap) ----------------
    # design: for each (n,pixel): [h3_center(64), 1]
    F = h3.transpose(0, 2, 3, 1).reshape(-1, 64)   # [N*256, 64]
    Yv = y.reshape(-1)
    # add the analytic Wiener predictor as an extra regressor (helps low-t regime)
    Wpred = np.zeros((N, 16, 16))
    for k in range(32):
        m = t == k
        if m.any():
            Wpred[m] = (c_t[k] * x[m][:, 0] + d_t[k])
    Wp = Wpred.reshape(-1, 1)
    A = np.concatenate([F, Wp, np.ones((F.shape[0], 1))], axis=1)
    lam = 1e-4 * (np.trace(A.T @ A) / A.shape[1] + 1e-12)
    coef = ridge_solve(A, Yv, lam)
    w_out = np.zeros((1, 64, 3, 3))
    w_out[0, :, 1, 1] = coef[:64]
    b_out = np.array([coef[66]])
    # the Wiener regressor cannot be produced by the network -> fold it in by
    # rescaling: we instead simply drop it if it is not representable.
    # (keep only the representable part: features + bias)
    coef2 = coef[:64]
    b_out = np.array([coef[66]])

    # ---------------- verify / refine with a second closed-form pass ----------
    # Recompute predictions with the fitted read-out and report MSE.
    def eval_mse(Wd):
        p = forward(x, t, Wd)
        return float(np.mean((p - y) ** 2))

    Wnew['out.weight'] = w_out
    Wnew['out.bias'] = b_out
    mse_new = eval_mse(Wnew)

    # If the analytic Wiener regressor helped a lot in-sample, try to realise it
    # more directly: increase the number of t-gated probes' resolution by using
    # the full 32 channels for 32 distinct t thresholds (already done) and
    # additionally allow the read-out to use the 3x3 neighbourhood (spatial
    # smoothing of x_t) via conv2/conv3 taps -> do a second closed-form fit.
    # (Second stage: fit out layer with 3x3 taps on the identity trunk.)
    # Build richer features: conv2 = [center x, left, right, up, down] style taps
    taps = [(1, 1), (0, 1), (2, 1), (1, 0), (1, 2)]
    W2b = np.zeros((64, 32, 3, 3))
    for i, (ky, kx) in enumerate(taps):
        W2b[i, :, ky, kx] = 1.0
    for i in range(len(taps), 64):
        W2b[i, :, 1, 1] = 1.0
    b2b = np.zeros(64)
    W3b = np.zeros((64, 64, 3, 3)); W3b[:, :, 1, 1] = 1.0
    b3b = np.zeros(64)
    h1b = h1
    h2b = np.maximum(conv2d(h1b, W2b, b2b), 0.0)
    h3b = np.maximum(conv2d(h2b, W3b, b3b), 0.0)
    Fb = h3b.transpose(0, 2, 3, 1).reshape(-1, 64)
    Ab = np.concatenate([Fb, np.ones((Fb.shape[0], 1))], axis=1)
    lamb = 1e-4 * (np.trace(Ab.T @ Ab) / Ab.shape[1] + 1e-12)
    coefb = ridge_solve(Ab, Yv, lamb)
    Wtry = dict(Wnew)
    Wtry['conv2.weight'] = W2b
    Wtry['conv3.weight'] = W3b
    w_outb = np.zeros((1, 64, 3, 3))
    w_outb[0, :, 1, 1] = coefb[:64]
    Wtry['out.weight'] = w_outb
    Wtry['out.bias'] = np.array([coefb[64]])
    mse_try = eval_mse(Wtry)
    if mse_try < mse_new:
        Wnew = Wtry
        mse_new = mse_try

    # ---------------- write back, preserve keys/shapes ----------------
    out = {}
    for k, v in w.items():
        arr = np.asarray(v)
        newv = Wnew[k]
        newv = np.clip(newv, -1000.0, 1000.0).astype(np.float32)
        out[k] = newv.reshape(arr.shape).astype(np.float32)

    diag = {
        'mse_train_new': float(mse_new),
        'mse_train_old': float(np.mean((np.asarray(f['prediction'], dtype=np.float64) - y) ** 2)),
        'c_t_min': float(c_t.min()), 'c_t_max': float(c_t.max()),
        'd_t_absmax': float(np.abs(d_t).max()),
        'n': int(N)
    }
    return {'weights': out, 'diagnostics': diag}
