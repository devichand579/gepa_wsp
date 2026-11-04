import numpy as np

def edit(w, f):
    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def conv2d(x, W, b):
        # x: (N,C,H,W) float32 ; W: (O,C,3,3) ; returns (N,O,H,W)
        N, C, H, Wd = x.shape
        O = W.shape[0]
        xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
        # im2col: (N,H,W,C,3,3)
        s = xp.strides
        cols = np.lib.stride_tricks.as_strided(
            xp, shape=(N, H, Wd, C, 3, 3),
            strides=(s[0], s[2], s[3], s[1], s[2], s[3]), writeable=False)
        cols = cols.reshape(N * H * Wd, C * 9)
        out = cols @ W.reshape(O, C * 9).T
        out += b[None, :]
        return out.reshape(N, O, H, Wd)

    def relu(x):
        return np.maximum(x, 0.0, out=x)

    def forward_all(x, t, Ws):
        # x: (N,1,16,16) ; t: (N,)
        N = x.shape[0]
        tb = (t.astype(np.float32) / 31.0).reshape(N, 1, 1, 1)
        inp = np.concatenate([x, np.broadcast_to(tb, (N, 1, x.shape[2], x.shape[3]))], axis=1)
        h1 = relu(conv2d(inp, Ws['conv1.weight'], Ws['conv1.bias']))
        h2 = relu(conv2d(h1, Ws['conv2.weight'], Ws['conv2.bias']))
        h3 = relu(conv2d(h2, Ws['conv3.weight'], Ws['conv3.bias']))
        return h1, h2, h3

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------
    x = np.asarray(f['x'], dtype=np.float32)
    t = np.asarray(f['t']).astype(np.int64)
    y = np.asarray(f['y'], dtype=np.float32)
    N = x.shape[0]

    # ------------------------------------------------------------------
    # design the hidden stack
    #   conv1: 32 chans
    #     ch 0      : x + s_hi(t)              (t-dependent shift, high noise)
    #     ch 1      : x + s_lo(t)              (t-dependent shift, low noise)
    #     ch 2      : x                        (plain)
    #     ch 3      : -x
    #     ch 4..11  : local x patch basis (3x3 identity/edge/corner patches)
    #     ch 12..31 : t-embedding ramps:  ReLU(tau_k(t) - x)  (tau_k = t/31)
    #   conv2: 64 chans = [pass-through of conv1 chans 0..31] + [local x patch basis 32]
    #   conv3: 64 chans = [pass-through of conv2 chans 0..63]
    # ------------------------------------------------------------------
    W1 = np.zeros((32, 2, 3, 3), dtype=np.float32)
    b1 = np.zeros(32, dtype=np.float32)
    W2 = np.zeros((64, 32, 3, 3), dtype=np.float32)
    b2 = np.zeros(64, dtype=np.float32)
    W3 = np.zeros((64, 64, 3, 3), dtype=np.float32)
    b3 = np.zeros(64, dtype=np.float32)

    def set_center(W, out_c, in_c, val):
        W[out_c, in_c, 1, 1] = val

    # conv1 channel 0/1/2/3 : +-x with t shift
    set_center(W1, 0, 0, 1.0)
    set_center(W1, 1, 0, 1.0)
    set_center(W1, 2, 0, 1.0)
    set_center(W1, 3, 0, -1.0)
    b1[0] = 0.0
    b1[1] = 0.0
    b1[2] = 0.0
    b1[3] = 0.0
    # t-dependent shifts implemented via the t channel (channel 1 of the input)
    #   shift(t) = a*(t/31) + c  ->  w[.,1,1,1] = a , b = c
    W1[0, 1, 1, 1] = 0.0     # channel 0: x + s0(t)  (s0 built below)
    W1[1, 1, 1, 1] = 0.0
    # conv1 channel 4..11 : local 3x3 patch basis of x (identity + 8 neighbours)
    patch_basis = [(0, 0), (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for k in range(8):
        c = 4 + k
        dr, dc = patch_basis[k + 1]
        W1[c, 0, 1 + dr, 1 + dc] = 1.0
    # conv1 channel 12..31 : ReLU(tau_k(t) - x)  with tau_k = t/31 + c_k
    taus = np.array([-0.6, -0.4, -0.25, -0.15, -0.08, -0.03, 0.0, 0.03, 0.08, 0.15,
                     0.25, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], dtype=np.float32)
    for k, tk in enumerate(taus):
        c = 12 + k
        W1[c, 1, 1, 1] = 1.0          # + t/31
        W1[c, 0, 1, 1] = -1.0         # - x
        b1[c] = tk
    # conv1 channels 0/1 t-shift: use dedicated ramp channels to realise
    #   x + s(t) with s(t) = 0.5*(t/31) - 0.25  and  s(t) = -0.5*(t/31) + 0.25
    W1[0, 1, 1, 1] = 0.5
    b1[0] = -0.25
    W1[1, 1, 1, 1] = -0.5
    b1[1] = 0.25

    # conv2 : pass through conv1 channels 0..31 (identity spatial kernel)
    for c in range(32):
        set_center(W2, c, c, 1.0)
    # conv2 channels 32..63 : local 3x3 patch basis of x again (from conv1 ch2 = x)
    for k in range(32):
        c = 32 + k
        dr, dc = patch_basis[k % 9]
        W2[c, 2, 1 + dr, 1 + dc] = 1.0
    # conv3 : pass through conv2 channels 0..63
    for c in range(64):
        set_center(W3, c, c, 1.0)

    Ws = {'conv1.weight': W1, 'conv1.bias': b1,
          'conv2.weight': W2, 'conv2.bias': b2,
          'conv3.weight': W3, 'conv3.bias': b3}

    # ------------------------------------------------------------------
    # forward pass on training data -> features for the readout
    # ------------------------------------------------------------------
    h1, h2, h3 = forward_all(x, t, Ws)
    F = h3                                  # (N,64,16,16) hidden features

    # ------------------------------------------------------------------
    # ridge regression for the final 3x3 conv layer:
    #   pred(i,j) = sum_{c,di,dj} a[c,di,dj] * F[:,c,i+di,j+dj] + b
    # design matrix rows = (n, i, j), cols = 64*9 (+ bias)
    # ------------------------------------------------------------------
    Nf = F.shape[1]
    Hh = F.shape[2]
    Wd = F.shape[3]
    Fp = np.pad(F, ((0, 0), (0, 0), (1, 1), (1, 1)))
    s = Fp.strides
    cols = np.lib.stride_tricks.as_strided(
        Fp, shape=(N, Hh, Wd, Nf, 3, 3),
        strides=(s[0], s[2], s[3], s[1], s[2], s[3]), writeable=False)
    X = cols.reshape(N * Hh * Wd, Nf * 9).astype(np.float64)
    Y = y.reshape(N * Hh * Wd).astype(np.float64)
    X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)

    # split for lambda selection
    ntr = 1600
    Xtr, Ytr = X[:ntr], Y[:ntr]
    Xva, Yva = X[ntr:], Y[ntr:]

    def solve(A, B, lam):
        d = A.shape[0]
        G = A.T @ A
        G[np.diag_indices(d)] += lam
        return np.linalg.solve(G, A.T @ B)

    def mse(A, B, coef):
        R = A @ coef - B
        return float(np.mean(R * R))

    lams = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6]
    best = None
    for lam in lams:
        c = solve(Xtr, Ytr, lam)
        v = mse(Xva, Yva, c)
        if best is None or v < best[1]:
            best = (lam, v)
    lam = best[0]

    coef = solve(X, Y, lam)
    coef = np.clip(coef, -1e3, 1e3)

    Wo = coef[:Nf * 9].reshape(Nf, 3, 3).astype(np.float32)
    bo = np.array([coef[Nf * 9]], dtype=np.float32)

    # ------------------------------------------------------------------
    # assemble new weights
    # ------------------------------------------------------------------
    out = {}
    out['conv1.weight'] = W1.astype(np.float32)
    out['conv1.bias'] = b1.astype(np.float32)
    out['conv2.weight'] = W2.astype(np.float32)
    out['conv2.bias'] = b2.astype(np.float32)
    out['conv3.weight'] = W3.astype(np.float32)
    out['conv3.bias'] = b3.astype(np.float32)
    out['out.weight'] = Wo
    out['out.bias'] = bo

    # ------------------------------------------------------------------
    # verification: recompute predictions with the assembled weights
    # ------------------------------------------------------------------
    Ws2 = {'conv1.weight': out['conv1.weight'], 'conv1.bias': out['conv1.bias'],
           'conv2.weight': out['conv2.weight'], 'conv2.bias': out['conv2.bias'],
           'conv3.weight': out['conv3.weight'], 'conv3.bias': out['conv3.bias']}
    _, _, h3b = forward_all(x, t, Ws2)
    pred = conv2d(h3b, out['out.weight'], out['out.bias'])
    m = float(np.mean((pred - y) ** 2))

    # per-timestep mse
    per_t = {}
    for tt in range(32):
        msk = (t == tt)
        if msk.any():
            per_t[str(tt)] = float(np.mean((pred[msk] - y[msk]) ** 2))

    for k in out:
        out[k] = np.ascontiguousarray(out[k], dtype=np.float32)
        out[k] = np.clip(out[k], -1000.0, 1000.0)
        assert np.all(np.isfinite(out[k]))

    diagnostics = {
        'train_mse_new': m,
        'train_mse_old': float(np.mean((np.asarray(f['prediction'], dtype=np.float32) - y) ** 2)),
        'val_mse_lambda': best[1],
        'lambda': float(lam),
        'per_t_mse': per_t,
        'out_weight_absmax': float(np.max(np.abs(out['out.weight']))),
        'out_bias': float(out['out.bias'][0]),
    }
    return {'weights': out, 'diagnostics': diagnostics}
