import numpy as np

def edit(w, f):
    # ---------------- helpers ----------------
    def conv2d(x, W, b):
        # x: (N,C,H,W), W: (O,C,kh,kw) cross-correlation, pad 1, stride 1
        N, C, H, Wd = x.shape
        O, C2, kh, kw = W.shape
        xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
        cols = np.lib.stride_tricks.as_strided(
            xp, shape=(N, C, kh, kw, H, Wd),
            strides=xp.strides[:2] + xp.strides[2:] + xp.strides[2:])
        # cols: (N,C,kh,kw,H,W)
        out = np.einsum('nckhw,ock->nohw', cols, W, optimize=True)
        return out + b.reshape(1, -1, 1, 1)

    def forward(x, t, W1, b1, W2, b2, W3, b3, Wo, bo):
        tb = (t.astype(np.float32) / 31.0).reshape(-1, 1, 1, 1)
        inp = np.concatenate([x, np.broadcast_to(tb, (x.shape[0], 1, x.shape[2], x.shape[3]))], axis=1)
        h1 = np.maximum(conv2d(inp, W1, b1), 0.0)
        h2 = np.maximum(conv2d(h1, W2, b2), 0.0)
        h3 = np.maximum(conv2d(h2, W3, b3), 0.0)
        o = conv2d(h3, Wo, bo)
        return o, h3

    # ---------------- data ----------------
    x = np.asarray(f['x'], dtype=np.float32)
    y = np.asarray(f['y'], dtype=np.float32)
    t = np.asarray(f['t'])
    N = x.shape[0]
    clean = np.asarray(f['clean'], dtype=np.float32)

    # ---------------- design conv1 ----------------
    # channels 0..15 : relu(x)      (delta kernel on x channel)
    # channels 16..31: relu(-x)     (-delta kernel on x channel)
    W1 = np.zeros((32, 2, 3, 3), dtype=np.float32)
    b1 = np.zeros((32,), dtype=np.float32)
    for c in range(16):
        W1[c, 0, 1, 1] = 1.0
        W1[c + 16, 0, 1, 1] = -1.0

    # ---------------- design conv2 ----------------
    # 32 channels: relu( ramp_j(t) * x )        (t-dependent gate times x)
    # 32 channels: relu( ramp_j(t) * (-x) )      (negative part)
    # ramp_j(t) = relu( s*t/31 - (j-1) ) with s chosen so ramps are one-hot-ish
    # We build ramp_j via conv1 channel j: pre = s*t/31 - (j-1)  -> relu
    # conv1 has only 32 channels; we instead realise ramps inside conv2 using
    # the t channel of the *input* is gone after conv1 -> so we create ramps in
    # conv1 channels 0..31 instead of x channels.  Layout below:
    #   conv1 ch 0..15  : relu(x)     (delta on x)
    #   conv1 ch 16..31 : relu(-x)    (-delta on x)
    #   ramp channels are created in conv2 from ... we need t; so we ALSO need
    #   t in conv2 -> not possible. => ramps must be built in conv1.
    #   => Use 16 x-channels + 16 ramp channels is not enough for 32 gates.
    #   => Instead: use 32 gates built from 32 ramp channels in conv1 and get
    #      signed x through 2 extra channels -> need 34 channels. Not possible.
    #   => Compromise: 16 gates (j=0..15) covering t in two halves via two
    #      different ramp slopes, plus 16 x channels.  We use gates:
    #      ramp_j(t) = relu( (t - (j-0.5)) ) style with j=0..15 over t/31
    #      giving 16 overlapping tent-like gates; combined with 16 x channels.
    #      Actually we can afford: conv1 ch 0..15 = relu(x), ch 16..31 = relu(-x)
    #      and NO ramp channels; instead conv2 receives t indirectly? No.
    #   => FINAL: conv1 ch 0..15 = relu(x), 16..31 = relu(-x); conv2 forms
    #      gate-free features; t-dependence is injected by conv2 biases? no.
    #      => Use a different, exact route: put the t-ramp into conv1 by
    #         *replacing* half of the x channels with ramps and reconstruct x
    #         from a single channel pair: relu(x), relu(-x) (2 channels) and
    #         30 ramp channels -> 30 gates (t=0..29) + 2 fallback gates.
    #         30 gates is plenty (t=30,31 handled by a wide gate).
    W1 = np.zeros((32, 2, 3, 3), dtype=np.float32)
    b1 = np.zeros((32,), dtype=np.float32)
    # ch 0: relu(x), ch 1: relu(-x)
    W1[0, 0, 1, 1] = 1.0
    W1[1, 0, 1, 1] = -1.0
    # ch 2..31 : 30 ramp channels, ramp_j(t) = relu( (t/31 - c_j) ) * scale
    # choose c_j = j/30 for j=0..29  -> ramps activate progressively
    n_ramp = 30
    ramp_c = np.arange(n_ramp, dtype=np.float32) / 30.0
    for j in range(n_ramp):
        W1[2 + j, 1, 1, 1] = 1.0
        b1[2 + j] = -float(ramp_c[j])

    # ---------------- design conv2 ----------------
    # out channels 0..29 : relu( ramp_j(t) * x )
    # out channels 30..59: relu( ramp_j(t) * (-x) )
    # out channels 60..63: relu(x), relu(-x), relu(ramp_last), relu(1)
    W2 = np.zeros((64, 32, 3, 3), dtype=np.float32)
    b2 = np.zeros((64,), dtype=np.float32)
    for j in range(n_ramp):
        # relu(ramp_j * x):  ramp_j channel (2+j) times x channel (0)
        W2[j, 2 + j, 1, 1] = 1.0
        W2[j, 0, 1, 1] = 1.0
        # relu(ramp_j * (-x)): ramp_j * relu(-x)  -> gives relu(-ramp_j*x)
        W2[30 + j, 2 + j, 1, 1] = 1.0
        W2[30 + j, 1, 1, 1] = 1.0
    # global channels
    W2[60, 0, 1, 1] = 1.0          # relu(x)
    W2[61, 1, 1, 1] = 1.0          # relu(-x)
    W2[62, 2 + n_ramp - 1, 1, 1] = 1.0   # relu(last ramp)
    W2[63, 2 + n_ramp - 1, 1, 1] = 1.0
    b2[63] = 1.0                   # constant 1 feature

    # ---------------- design conv3 ----------------
    # identity pass-through on the 64 conv2 channels (3x3 delta)
    W3 = np.zeros((64, 64, 3, 3), dtype=np.float32)
    b3 = np.zeros((64,), dtype=np.float32)
    for c in range(64):
        W3[c, c, 1, 1] = 1.0

    # ---------------- compute features & fit last layer ----------------
    Wo = np.asarray(w['out.weight'], dtype=np.float32).copy()
    bo = np.asarray(w['out.bias'], dtype=np.float32).copy()
    _, h3 = forward(x, t, W1, b1, W2, b2, W3, b3, Wo, bo)
    # h3: (N,64,16,16) nonneg
    Nf = h3.shape[0] * h3.shape[2] * h3.shape[3]
    F = h3.transpose(0, 2, 3, 1).reshape(Nf, 64).astype(np.float64)
    Y = y.transpose(0, 2, 3, 1).reshape(Nf, 1).astype(np.float64)
    Fb = np.concatenate([F, np.ones((Nf, 1))], axis=1)
    # ridge normal equations
    lam = 1e-3 * max(1.0, np.trace(Fb.T @ Fb) / 65.0)
    A = Fb.T @ Fb
    A[np.arange(65), np.arange(65)] += lam
    b = Fb.T @ Y
    sol = np.linalg.solve(A, b)
    Wo_new = sol[:64].reshape(1, 64, 1, 1).astype(np.float32)
    bo_new = sol[64:65].astype(np.float32)

    # ---------------- assemble ----------------
    w['conv1.weight'] = W1
    w['conv1.bias'] = b1
    w['conv2.weight'] = W2
    w['conv2.bias'] = b2
    w['conv3.weight'] = W3
    w['conv3.bias'] = b3
    w['out.weight'] = np.clip(Wo_new, -1000.0, 1000.0)
    w['out.bias'] = np.clip(bo_new, -1000.0, 1000.0)

    # ---------------- diagnostics ----------------
    pred, _ = forward(x, t, W1, b1, W2, b2, W3, b3, w['out.weight'], w['out.bias'])
    mse_new = float(np.mean((pred - y) ** 2))
    mse_old = float(np.mean((np.asarray(f['prediction'], dtype=np.float32) - y) ** 2))
    # reference: optimal per-timestep linear predictor alpha(t)*x
    ab = np.asarray(f['alphabar'], dtype=np.float64)
    vc = float(np.var(clean))
    alphas = np.sqrt(1.0 - ab) / (ab * vc + (1.0 - ab))
    tt = np.clip(t, 0, 31)
    lin = (alphas[tt] * x.transpose(0, 2, 3, 1)).reshape(y.shape)
    mse_lin = float(np.mean((lin - y) ** 2))
    diag = {'mse_old': mse_old, 'mse_new': mse_new,
            'mse_linear_ref': mse_lin,
            'mse_new_t0_7': float(np.mean((pred - y)[tt <= 7] ** 2)),
            'mse_new_t8_15': float(np.mean((pred - y)[(tt > 7) & (tt <= 15)] ** 2)),
            'mse_new_t16_23': float(np.mean((pred - y)[(tt > 15) & (tt <= 23)] ** 2)),
            'mse_new_t24_31': float(np.mean((pred - y)[tt > 23] ** 2)),
            'var_clean': vc, 'alphabar_last': float(ab[-1])}
    for k in w:
        w[k] = np.asarray(w[k], dtype=np.float32)
        w[k] = np.clip(np.nan_to_num(w[k]), -1000.0, 1000.0)
    return {'weights': w, 'diagnostics': diag}
