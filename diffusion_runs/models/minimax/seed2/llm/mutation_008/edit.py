import numpy as np

def edit(w, f):
    # ---------------- helpers ----------------
    def conv3x3(x, W, b):
        # x: (N,C,H,W), W: (O,C,3,3) cross-correlation, pad 1
        N, C, H, Wd = x.shape
        O = W.shape[0]
        xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
        out = np.zeros((N, O, H, Wd), dtype=np.float64)
        for kr in range(3):
            for kc in range(3):
                out += np.einsum('nchw,oc->nohw', xp[:, :, kr:kr + H, kc:kc + Wd], W[:, :, kr, kc])
        out += b[None, :, None, None]
        return out

    def conv3x3_grad(x, W):
        N, C, H, Wd = x.shape
        O = W.shape[0]
        xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
        g = np.zeros((N, O, H, Wd), dtype=np.float64)
        for kr in range(3):
            for kc in range(3):
                g += np.einsum('nchw,oc->nohw', xp[:, :, kr:kr + H, kc:kc + Wd], W[:, :, kr, kc])
        return g

    def ridge(A, y, lam):
        d = A.shape[1]
        G = A.T @ A
        G.flat[:: d + 1] += lam
        try:
            return np.linalg.solve(G, A.T @ y)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(G, A.T @ y, rcond=None)[0]

    # ---------------- data ----------------
    x = np.asarray(f['x'], dtype=np.float64)          # (N,1,16,16)
    t = np.asarray(f['t']).astype(np.int64).ravel()    # (N,)
    y = np.asarray(f['y'], dtype=np.float64)          # (N,1,16,16)
    clean = np.asarray(f['clean'], dtype=np.float64)  # (N,1,16,16)
    abar = np.asarray(f['alphabar'], dtype=np.float64).ravel()
    N = x.shape[0]

    # ---------------- teacher: per-timestep ridge x0 ~ x_t ----------------
    # teacher: eps_hat = (x - sqrt(a)*x0hat)/sqrt(1-a), x0hat = A_t*x + B_t
    eps_hat = np.empty_like(x)
    x0hat = np.empty_like(x)
    for k in np.unique(t):
        m = (t == k)
        a = float(abar[k])
        sa = np.sqrt(a)
        sb = np.sqrt(max(1.0 - a, 1e-12))
        Xm = x[m].reshape(m.sum(), -1)
        Cm = clean[m].reshape(m.sum(), -1)
        A = np.concatenate([Xm, np.ones((Xm.shape[0], 1))], axis=1)
        coef = ridge(A, Cm, 1e-3)
        x0h = (A @ coef).reshape(-1, 1, 16, 16)
        x0hat[m] = x0h
        eps_hat[m] = (x[m] - sa * x0h) / sb

    mse_teacher = float(np.mean((eps_hat - y) ** 2))
    mse_zero = float(np.mean(y ** 2))

    # ---------------- fit conv1 (2 -> 32) by ridge on [x_t, t/31] ----------------
    # target: conv1 pre-activation should reproduce (x_t, t/31) up to an affine map
    tnorm = (t.astype(np.float64) / 31.0)
    inp = np.concatenate([x.reshape(N, -1), np.repeat(tnorm[:, None], 256, axis=1)], axis=1)  # (N, 512)

    # design: each conv1 output channel = w_x * x + w_t * t + b  (per pixel)
    # => 32 channels, each with 3 params, fitted to reproduce x and t/31
    # Build least squares for parameters (w_x_k, w_t_k, b_k) per channel k:
    #   channel k should equal  x  for k < 16 and t/31 for k >= 16
    # rows: (N*256) samples; per channel 3 unknowns -> solve 32 independent 3x3 systems
    # Use structure: for channel k, features per sample: [x_pix, tnorm, 1]
    xs = x.reshape(N, 256)
    F = np.empty((N, 256, 3))
    F[:, :, 0] = xs
    F[:, :, 1] = np.repeat(tnorm[:, None], 256, axis=1)
    F[:, :, 2] = 1.0
    Ff = F.reshape(-1, 3)
    conv1_w = np.zeros((32, 2, 3, 3), dtype=np.float64)
    conv1_b = np.zeros(32, dtype=np.float64)
    for k in range(32):
        tgt = xs if k < 16 else np.repeat(tnorm[:, None], 256, axis=1)
        coef = ridge(Ff, tgt.reshape(-1), 1e-6)
        wx, wt, b = coef
        # distribute over 3x3 kernel: center tap carries most, others 0
        conv1_w[k, 0, 1, 1] = wx
        conv1_w[k, 1, 1, 1] = wt
        conv1_b[k] = b
    # keep only 2 informative channels to avoid redundancy; but 32 channels are allowed:
    # channel 0 -> x, channel 1 -> t/31 ; others zero
    conv1_w = np.zeros((32, 2, 3, 3), dtype=np.float64)
    conv1_b = np.zeros(32, dtype=np.float64)
    conv1_w[0, 0, 1, 1] = 1.0
    conv1_w[1, 1, 1, 1] = 1.0
    conv1_b[0] = 0.0
    conv1_b[1] = 0.0

    # ---------------- reconstruct teacher slope per timestep from conv1 features ----
    # conv1 output (pre-ReLU) channels: ch0 = x, ch1 = t/31 (others 0)
    # We want conv2 to implement: out = S_t * x  where S_t is the teacher's effective
    # per-timestep linear map.  Use the "always-positive" trick:
    #   conv2 channel j output = relu( A_j * x + B_j * t/31 + C_j + M )
    # with M large => output = A_j x + B_j (t/31) + C_j + M  (linear).
    # conv3 then computes conv2 output (linear) and out reads it.
    # The teacher's map eps_hat = (x - sa*(A_t x + B_t))/sb = g_t x + h_t
    #   g_t = (1 - sa*A_t)/sb ,  h_t = -sa*B_t/sb
    # We must produce g_t*x + h_t using channels whose x-coefficient is fixed by the
    # conv1 structure.  Since conv1 gives us x and t/31 as separate channels, a
    # conv2 channel can have x-coefficient A_j and t-coefficient B_j independently.
    # => we can realize ANY affine-in-t family: out = (a0 + a1*t/31)*x + (b0 + b1*t/31)
    # Choose a0,a1,b0,b1 by least squares over all (t,x) to match teacher.
    # Build per-timestep teacher coefficients g_t, h_t:
    g = np.zeros(32); h = np.zeros(32)
    for k in range(32):
        m = (t == k)
        a = float(abar[k]); sa = np.sqrt(a); sb = np.sqrt(max(1.0 - a, 1e-12))
        Xm = x[m].reshape(m.sum(), -1); Cm = clean[m].reshape(m.sum(), -1)
        A = np.concatenate([Xm, np.ones((Xm.shape[0], 1))], axis=1)
        coef = ridge(A, Cm, 1e-3)
        g[k] = (1.0 - sa * coef[0]) / sb
        h[k] = (-sa * coef[-1]) / sb

    # fit out = (a0+a1*u)*x + (b0+b1*u), u = t/31  -> teacher (g_t x + h_t)
    # design rows per (example,pixel): [x, x*u, 1, u]
    u_all = np.repeat(tnorm[:, None], 256, axis=1)
    D = np.stack([xs, xs * u_all, np.ones_like(xs), u_all], axis=-1).reshape(-1, 4)
    Tg = np.empty((N, 256))
    Th = np.empty((N, 256))
    for k in range(32):
        m = (t == k)
        Tg[m] = g[k]; Th[m] = h[k]
    Tg = Tg.reshape(-1); Th = Th.reshape(-1)
    c4 = ridge(D, Tg, 1e-6)
    d4 = ridge(D, Th, 1e-6)
    a0, a1, b0, b1 = c4
    e0, e1, f0, f1 = d4

    # ---------------- realise this map with the conv stack ----------------
    # conv2 channel 0: relu( a0*x + a1*u + M )  -> linear
    # conv2 channel 1: relu( b0 + b1*u + M )    -> constant spatially
    # conv3 channel 0: conv2 ch0 (identity kernel) + bias -> relu( a0 x + a1 u + M2 )
    # conv3 channel 1: conv2 ch1 -> relu( b0 + b1 u + M2 )
    # out: weight on conv3 ch0 = 1, ch1 = 1, bias cancels constants.
    M = 50.0
    conv2_w = np.zeros((64, 32, 3, 3), dtype=np.float64)
    conv2_b = np.zeros(64, dtype=np.float64)
    conv2_w[0, 0, 1, 1] = a0
    conv2_w[0, 1, 1, 1] = a1
    conv2_b[0] = M
    conv2_w[1, 1, 1, 1] = b1
    conv2_b[1] = b0 + M

    conv3_w = np.zeros((64, 64, 3, 3), dtype=np.float64)
    conv3_b = np.zeros(64, dtype=np.float64)
    conv3_w[0, 0, 1, 1] = 1.0
    conv3_w[1, 1, 1, 1] = 1.0
    conv3_b[0] = M
    conv3_b[1] = M

    out_w = np.zeros((1, 64, 3, 3), dtype=np.float64)
    out_b = np.zeros(1, dtype=np.float64)
    out_w[0, 0, 1, 1] = 1.0
    out_w[0, 1, 1, 1] = 1.0
    out_b[0] = -2.0 * M

    # ---------------- verify with our own forward pass ----------------
    def forward(xt, tt, c1w, c1b, c2w, c2b, c3w, c3b, ow, ob):
        Nn = xt.shape[0]
        u = (tt.astype(np.float64) / 31.0).reshape(Nn, 1, 1, 1)
        inp2 = np.concatenate([xt, np.broadcast_to(u, (Nn, 1, 16, 16))], axis=1)
        h1 = np.maximum(conv3x3(inp2, c1w, c1b), 0.0)
        h2 = np.maximum(conv3x3(h1, c2w, c2b), 0.0)
        h3 = np.maximum(conv3x3(h2, c3w, c3b), 0.0)
        return conv3x3(h3, ow, ob)

    pred = forward(x, t, conv1_w, conv1_b, conv2_w, conv2_b,
                   conv3_w, conv3_b, out_w, out_b)
    mse_net = float(np.mean((pred - y) ** 2))

    # ---------------- assemble output weights ----------------
    out = {}
    for k in w.keys():
        out[k] = np.array(w[k], dtype=np.float32, copy=True)
    out['conv1.weight'] = conv1_w.astype(np.float32)
    out['conv1.bias'] = conv1_b.astype(np.float32)
    out['conv2.weight'] = conv2_w.astype(np.float32)
    out['conv2.bias'] = conv2_b.astype(np.float32)
    out['conv3.weight'] = conv3_w.astype(np.float32)
    out['conv3.bias'] = conv3_b.astype(np.float32)
    out['out.weight'] = out_w.astype(np.float32)
    out['out.bias'] = out_b.astype(np.float32)

    for k in out:
        out[k] = np.clip(np.nan_to_num(out[k], nan=0.0, posinf=0.0, neginf=0.0), -1000.0, 1000.0).astype(np.float32)

    diagnostics = {
        'mse_zero': round(mse_zero, 5),
        'mse_teacher': round(mse_teacher, 5),
        'mse_network': round(mse_net, 5),
        'teacher_g_t0': round(float(g[0]), 5),
        'teacher_g_t31': round(float(g[31]), 5),
        'a0': round(float(a0), 5), 'a1': round(float(a1), 5),
        'b0': round(float(b0), 5), 'b1': round(float(b1), 5),
    }
    return {'weights': out, 'diagnostics': diagnostics}
