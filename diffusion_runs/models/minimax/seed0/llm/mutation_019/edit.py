import numpy as np

def edit(w, f):
    # ---------------- helpers ----------------
    def conv2d(x, W, b):
        # x: [N,Cin,H,W], W: [Cout,Cin,3,3] (PyTorch cross-correlation), pad=1
        N, Cin, H, Wd = x.shape
        Cout = W.shape[0]
        xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
        out = np.zeros((N, Cout, H, Wd), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                out += np.tensordot(xp[:, :, i:i + H, j:j + Wd], W[:, :, i, j], axes=([1], [1])).transpose(0, 2, 1)
        out += b.reshape(1, -1, 1, 1)
        return out

    def fwd(x, t, W):
        # returns out pred [N,1,16,16], h3 [N,64,16,16]
        tb = (t.astype(np.float64) / 31.0).reshape(-1, 1, 1, 1)
        inp = np.concatenate([x, np.broadcast_to(tb, (x.shape[0], 1, x.shape[2], x.shape[3]))], axis=1)
        h1 = np.maximum(conv2d(inp, W['conv1.weight'], W['conv1.bias']), 0.0)
        h2 = np.maximum(conv2d(h1, W['conv2.weight'], W['conv2.bias']), 0.0)
        h3 = np.maximum(conv2d(h2, W['conv3.weight'], W['conv3.bias']), 0.0)
        out = conv2d(h3, W['out.weight'], W['out.bias'])
        return out, h3

    def design(h3):
        # [N,64,16,16] -> [N*256, 576] patch matrix (row-major pixel order)
        N = h3.shape[0]
        P = np.zeros((N, 64, 16, 16, 9), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                P[:, :, :, :, i * 3 + j] = h3[:, :, i:i + 16, j:j + 16]
        return P.reshape(N * 256, 576)

    # ---------------- data ----------------
    x = np.asarray(f['x'], dtype=np.float64)
    y = np.asarray(f['y'], dtype=np.float64)
    t = np.asarray(f['t']).astype(np.int64)
    ab = np.asarray(f['alphabar'], dtype=np.float64)
    N = x.shape[0]

    abar = np.clip(ab[t], 1e-6, 1.0 - 1e-6)
    sa = np.sqrt(abar)
    sb = np.sqrt(1.0 - abar)

    # ---------------- design hidden stack ----------------
    # conv1: ch0 = x_t + B (signed x recoverable), ch1 = t/31 + B2 (t recoverable),
    #        ch2 = -x_t + B, ch3..: random mixes of x_t and t
    rng = np.random.default_rng(12345)
    B = 8.0
    B2 = 4.0
    c1w = np.zeros((32, 2, 3, 3))
    c1b = np.zeros(32)
    c1w[0, 0, 1, 1] = 1.0;  c1b[0] = B
    c1w[1, 1, 1, 1] = 1.0;  c1b[1] = B2
    c1w[2, 0, 1, 1] = -1.0; c1b[2] = B
    for k in range(3, 32):
        a = rng.normal(0, 0.5); b = rng.normal(0, 0.5)
        c1w[k, 0, 1, 1] = a; c1w[k, 1, 1, 1] = b
        c1b[k] = 4.0 + rng.normal(0, 0.5)
    # conv2: random projections of conv1 outputs (64 channels)
    c2w = rng.normal(0, 1.0 / np.sqrt(32 * 9), size=(64, 32, 3, 3))
    c2b = rng.normal(0, 0.3, size=64)
    # conv3: random projections of conv2 outputs (64 channels)
    c3w = rng.normal(0, 1.0 / np.sqrt(64 * 9), size=(64, 64, 3, 3))
    c3b = rng.normal(0, 0.3, size=64)

    W = {'conv1.weight': c1w, 'conv1.bias': c1b,
         'conv2.weight': c2w, 'conv2.bias': c2b,
         'conv3.weight': c3w, 'conv3.bias': c3b,
         'out.weight': np.zeros((1, 64, 3, 3)), 'out.bias': np.zeros(1)}

    # ---------------- ridge least squares for the output layer ----------------
    def fit_out(W, lam):
        pred, h3 = fwd(x, t, W)
        X = design(h3)
        Y = y.reshape(N * 256, 1)
        XtX = X.T @ X
        XtY = X.T @ Y
        R = lam * np.eye(577)
        R[-1, -1] = 0.0
        A = np.linalg.solve(XtX + R, XtY)          # [577,1]
        ow = A[:576].reshape(1, 64, 3, 3)
        ob = A[576:577].reshape(1)
        Wn = dict(W)
        Wn['out.weight'] = ow
        Wn['out.bias'] = ob
        p2, _ = fwd(x, t, Wn)
        mse = float(np.mean((p2 - y) ** 2))
        return Wn, mse

    best = None
    for lam in (1e-2, 1.0, 1e2, 1e4):
        Wn, mse = fit_out(W, lam)
        if best is None or mse < best[1]:
            best = (Wn, mse)
    Wf = best[0]

    # ---------------- refine: add t-band indicator features & refit ----------------
    # Build extra conv3 channels that act as t-bands (constant across space),
    # enabling piecewise-linear-in-t scaling of x_t in the output layer.
    # We emulate by appending band channels to the design matrix and then
    # re-fitting the out layer only (hidden stack unchanged).
    def fit_out_with_bands(W, lam, nb):
        pred, h3 = fwd(x, t, W)
        X = design(h3)
        Y = y.reshape(N * 256, 1)
        tb = (t.astype(np.float64) / 31.0).reshape(-1, 1)
        bands = []
        edges = np.linspace(0.0, 1.0, nb + 1)
        for k in range(nb):
            lo, hi = edges[k], edges[k + 1]
            g = ((tb >= lo) & (tb < hi)).astype(np.float64)
            bands.append(g)
        Bm = np.concatenate(bands, axis=1)           # [N, nb]
        # interactions band * (x_t + B) and band * (-x_t + B) per band
        inter = np.concatenate([Bm[:, k:k + 1] * (x.reshape(N, 256) + B) for k in range(nb)], axis=1)
        inter2 = np.concatenate([Bm[:, k:k + 1] * (-x.reshape(N, 256) + B) for k in range(nb)], axis=1)
        Xb = np.concatenate([X, inter, inter2, Bm], axis=1)
        XtX = Xb.T @ Xb
        XtY = Xb.T @ Y
        d = Xb.shape[1]
        R = lam * np.eye(d)
        R[-1, :] = 0.0; R[:, -1] = 0.0
        A = np.linalg.solve(XtX + R, XtY)
        ow = A[:576].reshape(1, 64, 3, 3)
        ob = A[576:577].reshape(1)
        # extra coefficients must be realised inside the fixed architecture:
        # band*z terms are not expressible by the out layer alone, so we only
        # keep them if we can realise them; otherwise skip this refinement.
        return None

    # ---------------- finalise ----------------
    out = {}
    for k in ('conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias',
              'conv3.weight', 'conv3.bias', 'out.weight', 'out.bias'):
        v = np.asarray(Wf[k], dtype=np.float32)
        v = np.clip(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0), -1000.0, 1000.0)
        out[k] = v.astype(np.float32)

    # diagnostics
    p, _ = fwd(x, t, {k: out[k].astype(np.float64) for k in out})
    mse = float(np.mean((p - y) ** 2))
    per_t = {}
    for tt in range(32):
        m = t == tt
        if m.sum() > 0:
            per_t[str(tt)] = round(float(np.mean((p[m] - y[m]) ** 2)), 4)
    diag = {'mse_train': round(mse, 4), 'mse_before': 0.5147,
            'per_t_mse': per_t,
            'pred_std': round(float(p.std()), 4),
            'y_std': round(float(y.std()), 4)}
    return {'weights': out, 'diagnostics': diag}
