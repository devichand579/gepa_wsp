import numpy as np

def edit(w, f):
    # ---------------- helpers ----------------
    def conv2d(x, W, b):
        # x: [N,C,H,W], W: [O,C,3,3] cross-correlation, pad 1
        N, C, H, Wd = x.shape
        O = W.shape[0]
        xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
        out = np.zeros((N, O, H, Wd), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                out += np.tensordot(xp[:, :, i:i + H, j:j + Wd], W[:, :, i, j], axes=([1], [1])).transpose(0, 3, 1, 2)
        if b is not None:
            out += b.reshape(1, -1, 1, 1)
        return out

    def fwd(x, t, W1, b1, W2, b2, W3, b3, Wo, bo):
        tb = (t.astype(np.float32) / 31.0).reshape(-1, 1, 1, 1)
        inp = np.concatenate([x, np.broadcast_to(tb, (x.shape[0], 1, 16, 16))], axis=1)
        h1 = np.maximum(conv2d(inp, W1, b1), 0.0)
        h2 = np.maximum(conv2d(h1, W2, b2), 0.0)
        h3 = np.maximum(conv2d(h2, W3, b3), 0.0)
        return conv2d(h3, Wo, bo)

    def mse(p, y):
        return float(np.mean((p - y) ** 2))

    x = np.asarray(f['x'], dtype=np.float32)
    y = np.asarray(f['y'], dtype=np.float32)
    t = np.asarray(f['t']).astype(np.int64)
    N = x.shape[0]
    Xf = x.reshape(N, 256).astype(np.float64)
    Yf = y.reshape(N, 256).astype(np.float64)

    W1 = np.asarray(w['conv1.weight'], dtype=np.float32).copy()
    b1 = np.asarray(w['conv1.bias'], dtype=np.float32).copy()
    W2 = np.asarray(w['conv2.weight'], dtype=np.float32).copy()
    b2 = np.asarray(w['conv2.bias'], dtype=np.float32).copy()
    W3 = np.asarray(w['conv3.weight'], dtype=np.float32).copy()
    b3 = np.asarray(w['conv3.bias'], dtype=np.float32).copy()
    Wo = np.asarray(w['out.weight'], dtype=np.float32).copy()
    bo = np.asarray(w['out.bias'], dtype=np.float32).copy()

    base_mse = mse(fwd(x, t, W1, b1, W2, b2, W3, b3, Wo, bo), y)
    diag = {'base_mse': base_mse}

    # ---------------- (a) closed-form linear epsilon predictor ----------------
    # per-timestep ridge regression of y on x (with intercept), validated
    groups = [(t == k).nonzero()[0] for k in range(32)]
    alphas = [1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0]
    best = {}
    for k in range(32):
        idx = groups[k]
        if len(idx) < 20:
            continue
        Xg = Xf[idx]
        Yg = Yf[idx]
        n = len(idx)
        Xm = Xg.mean(0)
        Ym = Yg.mean(0)
        Xc = Xg - Xm
        Yc = Yg - Ym
        A = Xc.T @ Xc
        B = Xc.T @ Yc
        ntr = max(8, int(0.7 * n))
        tr = np.arange(ntr)
        va = np.arange(ntr, n)
        bestlam = None
        bestv = 1e18
        for lam in alphas:
            R = np.linalg.solve(A + lam * n * np.eye(256), B)
            pv = Xc[va] @ R + Ym
            v = float(np.mean((pv - Yg[va]) ** 2))
            if v < bestv:
                bestv = v
                bestlam = lam
        R = np.linalg.solve(A + bestlam * n * np.eye(256), B)
        best[k] = (R.astype(np.float32), Ym.astype(np.float32), bestlam, bestv)

    # ---------------- (b) local (spatially local) filter fit ----------------
    # design: for radius r, features = 3x3 neighbourhood of x (r=1) ... use
    # separable multi-scale: x at offsets in a (2r+1)^2 window, plus intercept.
    def build_design(Xf, r):
        Nn = Xf.shape[0]
        im = Xf.reshape(Nn, 16, 16)
        cols = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                sh = np.zeros_like(im)
                ys0 = slice(max(0, dy), 16 + min(0, dy))
                ys1 = slice(max(0, -dy), 16 + min(0, -dy))
                xs0 = slice(max(0, dx), 16 + min(0, dx))
                xs1 = slice(max(0, -dx), 16 + min(0, -dx))
                sh[:, ys0, xs1] = im[:, ys1, xs0]
                cols.append(sh.reshape(Nn, 256))
        cols.append(np.ones((Nn, 256), dtype=np.float64))
        return np.concatenate(cols, axis=1)

    # target: optimal linear predictor applied to x  (per timestep model)
    def lin_pred(idx):
        out = np.zeros((len(idx), 256), dtype=np.float64)
        for k in range(32):
            sel = idx[t[idx] == k]
            if len(sel) == 0:
                continue
            if k in best:
                R, Ym, lam, _ = best[k]
                out[sel] = Xf[sel] @ R + Ym
            else:
                out[sel] = 0.0
        return out

    rng = np.random.RandomState(0)
    perm = rng.permutation(N)
    ntr = int(0.75 * N)
    tr, va = perm[:ntr], perm[ntr:]

    # global local-filter fit (shared across timesteps) on the linear target
    best_r = 1
    best_lam = 1e-2
    best_val = 1e18
    for r in [1, 2, 3, 4]:
        D = build_design(Xf, r)
        ytar = lin_pred(np.arange(N))
        Dtr = D[tr]
        Ttr = ytar[tr]
        A = Dtr.T @ Dtr
        Bm = Dtr.T @ Ttr
        for lam in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
            R = np.linalg.solve(A + lam * len(tr) * np.eye(A.shape[0]), Bm)
            pv = D[va] @ R
            v = float(np.mean((pv - ytar[va]) ** 2))
            if v < best_val:
                best_val = v
                best_r = r
                best_lam = lam
    r = best_r
    lam = best_lam
    D = build_design(Xf, r)
    ytar = lin_pred(np.arange(N))
    A = D.T @ D
    Bm = D.T @ ytar
    Rglob = np.linalg.solve(A + lam * N * np.eye(A.shape[0]), Bm).astype(np.float32)
    diag['local_r'] = int(r)
    diag['local_val'] = float(best_val)

    # ---------------- (c) realise the local filter inside the CNN ----------------
    # conv1: 32 channels = shifted copies of x_t (9 of them) + t-constant channel
    # conv2: identity-ish selector (keeps shifts), conv3: identity, out: 3x3 mix
    # We implement out(p) = sum_{dy,dx} K(dy,dx) x_t(p+dy,dx) + c(t)
    # with K read off from Rglob (window of radius r -> use central 3x3 of it
    # by folding the outer taps into the 3-layer receptive field).
    # Simplify: use radius r=1 design (3x3) for direct implementation, but we
    # allow conv1..conv3 to widen the receptive field: we set conv1 channels to
    # shifted copies at offsets in {-2..2} via conv2 selectors, etc.
    # Practical approach: implement a 3x3 filter K directly (r=1) with conv1
    # providing shifted copies and out mixing them.
    K = np.zeros((3, 3), dtype=np.float32)
    # take the fitted 3x3 block of Rglob if available else identity-ish
    if r == 1:
        K = Rglob[:9].reshape(3, 3)
    else:
        # fold outer taps: use central 3x3 of the fitted window
        idx0 = (r - 1) * 3
        K = Rglob[idx0:idx0 + 9].reshape(3, 3)

    # t-dependent constant: use the per-timestep intercepts (mean of optimal
    # predictor) approximated by a linear function of t/31 (fit closed form)
    ts = (t.astype(np.float64) / 31.0)
    ytar_mean = ytar.mean(1)
    # linear fit of ytar_mean ~ [1, t/31]
    Mt = np.stack([np.ones(N), ts], 1)
    ct = np.linalg.lstsq(Mt, ytar_mean, rcond=None)[0].astype(np.float32)

    # build conv1: 9 channels of shifted x (offsets -1..1) + 1 channel constant
    W1n = np.zeros((32, 2, 3, 3), dtype=np.float32)
    b1n = np.zeros(32, dtype=np.float32)
    B = 30.0
    for k, (dy, dx) in enumerate([(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]):
        W1n[k, 0, 1 + dy, 1 + dx] = 1.0
        b1n[k] = B
    # channel 9: constant (from t channel) -> used for the t-dependent offset
    W1n[9, 1, 1, 1] = 1.0
    b1n[9] = B
    for k in range(10, 32):
        W1n[k, 0, 1, 1] = 0.0
        b1n[k] = B

    # conv2: select channels 0..8 -> channels 0..8 (identity in channel space),
    # plus channel 9 -> channel 9 (constant channel)
    W2n = np.zeros((64, 32, 3, 3), dtype=np.float32)
    b2n = np.zeros(64, dtype=np.float32)
    for k in range(9):
        W2n[k, k, 1, 1] = 1.0
        b2n[k] = B
    W2n[9, 9, 1, 1] = 1.0
    b2n[9] = B
    for k in range(10, 64):
        b2n[k] = B

    # conv3: identity on channels 0..9, others constant
    W3n = np.zeros((64, 64, 3, 3), dtype=np.float32)
    b3n = np.zeros(64, dtype=np.float32)
    for k in range(10):
        W3n[k, k, 1, 1] = 1.0
        b3n[k] = B
    for k in range(10, 64):
        b3n[k] = B

    # out: mix channels 0..8 with kernel K (3x3) and channel 9 with t-const
    Won = np.zeros((1, 64, 3, 3), dtype=np.float32)
    bon = np.zeros(1, dtype=np.float32)
    for k, (dy, dx) in enumerate([(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]):
        # conv1 channel k holds x_t shifted by (dy,dx); out kernel picks it
        Won[0, k, 1 - dy, 1 - dx] = K[dy + 1, dx + 1]
    Won[0, 9, 1, 1] = ct[1]
    bon[0] = ct[0]

    # sanity: forward pass of the constructed net
    pnew = fwd(x, t, W1n, b1n, W2n, b2n, W3n, b3n, Won, bon)
    m_new = mse(pnew, y)

    # ---------------- (d) refine final layer on recomputed features ----------
    if m_new < base_mse:
        # recompute features with the new trunk
        tb = (t.astype(np.float32) / 31.0).reshape(-1, 1, 1, 1)
        inp = np.concatenate([x, np.broadcast_to(tb, (N, 1, 16, 16))], axis=1)
        h1 = np.maximum(conv2d(inp, W1n, b1n), 0.0)
        h2 = np.maximum(conv2d(h1, W2n, b2n), 0.0)
        h3 = np.maximum(conv2d(h2, W3n, b3n), 0.0)
        # design matrix for out layer: rows = (n, p) ; cols = (channel, ky, kx)
        # build via patches of h3
        Nn = N
        hp = np.pad(h3, ((0, 0), (0, 0), (1, 1), (1, 1)))
        cols = []
        for ky in range(3):
            for kx in range(3):
                cols.append(hp[:, :, ky:ky + 16, kx:kx + 16].reshape(Nn, 64 * 256))
        Dm = np.concatenate(cols, axis=1).astype(np.float64)
        Ym2 = y.reshape(Nn, 256).astype(np.float64)
        # ridge LS with intercept
        A = Dm.T @ Dm
        Bm2 = Dm.T @ Ym2
        trm = tr
        lam2 = 1e-3
        R2 = None
        bestv = 1e18
        for l2 in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
            R2 = np.linalg.solve(A + l2 * len(trm) * np.eye(A.shape[0]), Bm2)
            pv = Dm[va] @ R2
            v = float(np.mean((pv - Ym2[va]) ** 2))
            if v < bestv:
                bestv = v
                bestR = R2
        R2 = bestR
        # assemble out weight: R2 shape [64*9, 256]
        Wo2 = np.zeros((1, 64, 3, 3), dtype=np.float32)
        for ky in range(3):
            for kx in range(3):
                blk = R2[ky * 3 + kx].reshape(64, 256)
                Wo2[0, :, ky, kx] = blk.T.astype(np.float32)
        p2 = fwd(x, t, W1n, b1n, W2n, b2n, W3n, b3n, Wo2, bo)
        m2 = mse(p2, y)
        if m2 < m_new:
            Won, bon, m_new = Wo2, bo, m2
            diag['refit_val'] = float(bestv)

    if m_new < base_mse:
        W1, b1, W2, b2, W3, b3, Wo, bo = W1n, b1n, W2n, b2n, W3n, b3n, Won, bon
        diag['new_mse'] = float(m_new)
    else:
        diag['new_mse'] = float(base_mse)

    # final safety: finite, bounded
    def clean(a):
        a = np.asarray(a, dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(a, -1000.0, 1000.0)

    out = {
        'conv1.weight': clean(W1), 'conv1.bias': clean(b1),
        'conv2.weight': clean(W2), 'conv2.bias': clean(b2),
        'conv3.weight': clean(W3), 'conv3.bias': clean(b3),
        'out.weight': clean(Wo), 'out.bias': clean(bo),
    }
    diag['mse_check'] = float(mse(fwd(x, t, out['conv1.weight'], out['conv1.bias'],
                                      out['conv2.weight'], out['conv2.bias'],
                                      out['conv3.weight'], out['conv3.bias'],
                                      out['out.weight'], out['out.bias']), y))
    return {'weights': out, 'diagnostics': diag}
