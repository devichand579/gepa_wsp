import numpy as np

# ------------------------------------------------------------------ utils
def _conv2d(x, W, b, pad=1):
    """cross-correlation, W[out,in,kh,kw], x[N,C,H,W] -> [N,O,H,W]"""
    N, C, H, Wd = x.shape
    O, C2, KH, KW = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    # im2col via stride tricks
    s = xp.strides
    cols = np.lib.stride_tricks.as_strided(
        xp,
        shape=(N, C, KH, KW, H, Wd),
        strides=(s[0], s[1], s[2], s[3], s[2], s[3]),
        writeable=False)
    cols = cols.reshape(N, C * KH * KW, H * Wd)
    Wm = W.reshape(O, C * KH * KW)
    out = np.einsum('oc,ncp->nop', Wm, cols, optimize=True) + b[None, :, None]
    return out.reshape(N, O, H, Wd)


def _forward(x, t, w):
    """returns (pred, feats) with feats = post-ReLU conv3 output"""
    N = x.shape[0]
    tau = (t.astype(np.float32) / 31.0).reshape(N, 1, 1, 1)
    inp = np.concatenate([x, np.broadcast_to(tau, (N, 1, x.shape[2], x.shape[3]))], axis=1)
    h1 = np.maximum(_conv2d(inp, w['conv1.weight'], w['conv1.bias']), 0.0)
    h2 = np.maximum(_conv2d(h1, w['conv2.weight'], w['conv2.bias']), 0.0)
    h3 = np.maximum(_conv2d(h2, w['conv3.weight'], w['conv3.bias']), 0.0)
    pred = _conv2d(h3, w['out.weight'], w['out.bias'])[:, 0]
    return pred, h3


def _ridge_solve(A, b, lam):
    """A: [n,d] (float32/float64), b: [n] -> x [d]"""
    d = A.shape[1]
    G = (A.T @ A).astype(np.float64)
    G.flat[:: d + 1] += lam
    r = (A.T @ b).astype(np.float64)
    try:
        return np.linalg.solve(G, r)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(G, r, rcond=None)[0]


def edit(w, f):
    w = {k: np.array(v, dtype=np.float32, copy=True) for k, v in w.items()}
    diag = {}

    x = np.asarray(f['x'], dtype=np.float32)
    y = np.asarray(f['y'], dtype=np.float32)
    t = np.asarray(f['t']).astype(np.int64)
    clean = np.asarray(f['clean'], dtype=np.float32)
    abar = np.asarray(f['alphabar'], dtype=np.float64)

    N = x.shape[0]
    H = x.shape[2]
    a = np.sqrt(abar)                      # [32]
    b = np.sqrt(np.maximum(1.0 - abar, 1e-12))

    # ---- per-pixel statistics of the clean images (permitted feedback data)
    mu = clean.mean(axis=0)[0]                       # [16,16]
    var = clean.var(axis=0)[0]                       # [16,16]
    sig2 = np.maximum(var, 1e-4)

    # ---- optimal per-timestep linear gain (Gaussian approximation)
    # k_t = b_t / (a_t^2 * sigma^2 + b_t^2)   (per pixel)
    k_t = b[:, None, None] / (a[:, None, None] ** 2 * sig2[None] + b[:, None, None] ** 2)
    k_t = np.clip(k_t, 0.0, 1.0)

    # ---- baseline: current network MSE (sanity, using own forward)
    try:
        p0, _ = _forward(x, t, w)
        diag['mse_before'] = float(np.mean((p0 - y) ** 2))
    except Exception:
        diag['mse_before'] = None

    # ================================================================
    # Analytic construction of the gain network
    #
    # We want, per pixel:  out = k_t * (x - a_t * mu)
    #
    # Multiplicative gain trick: a per-example constant tau shifts the
    # breakpoint of a ReLU ramp, so the *slope* of the resulting
    # piecewise-linear function depends on tau.
    #
    #   ramp_c(x) = relu(x + c) - relu(x - c) = 2c  for |x| < c
    # With a tau-dependent shift s(tau):
    #   relu(x - s + c) - relu(x - s - c)
    #   slope in |x - s| < c  ->  2 ; outside -> 0
    # We instead use a *scaled* ramp so that the active slope equals k_t:
    #   relu(k*x - s + c) - relu(k*x - s - c)  -> slope 2k in the window.
    # k*x is produced by conv1 (linear in x) and s,c by the tau channel.
    # ================================================================

    # --- choose the piecewise-linear schedule of k(tau) and shifts s(tau)
    # k(tau) is monotone-ish in t; we realise it with a set of segments.
    # Segment j covers tau in [u_j, u_{j+1}); within it we want gain k_j.
    # We realise the gain by a ramp whose *input* is scaled by k_j and whose
    # window is placed so that the whole (bounded) x-range is inside it.
    #
    # Instead of one segment per timestep we use a modest number of
    # segments (each realised by 2 conv1 channels) and then let the
    # closed-form last-layer fit refine the mapping.

    # bound on |x| for the ramp window (x is noisy image, |x| <= ~5)
    XMAX = 5.0

    # segments: choose breakpoints in tau and target gains
    nseg = 8
    edges = np.linspace(0.0, 1.0, nseg + 1)
    # target gain per segment = mean of k_t over timesteps in that segment
    seg_gain = np.zeros(nseg)
    for j in range(nseg):
        m = (t >= 0)  # placeholder
        ts = np.arange(32)
        sel = (ts / 31.0 >= edges[j]) & (ts / 31.0 <= edges[j + 1] + 1e-9)
        if sel.sum() == 0:
            sel = np.array([j * 4])
        seg_gain[j] = float(np.mean(k_t[sel]))
    seg_gain = np.clip(seg_gain, 1e-3, 1.0)

    # --- build conv1
    # channel layout:
    #   0..(nseg-1)      : ramp+ channels  relu(k_j*x + off_j)
    #   nseg..(2nseg-1)  : ramp- channels  relu(k_j*x - off_j)
    #   2nseg            : tau channel (constant per example, value tau)
    #   2nseg+1          : x passthrough (relu(x + XMAX))
    #   2nseg+2          : -x passthrough (relu(-x + XMAX))
    C1 = 2 * nseg + 3
    w1 = np.zeros((32, 2, 3, 3), dtype=np.float32)
    b1 = np.zeros(32, dtype=np.float32)

    # tau channel: value tau broadcast (weight 1 on channel1, bias 0)
    w1[2 * nseg, 1, 1, 1] = 1.0

    # x passthrough channels (identity-ish through ReLU)
    w1[2 * nseg + 1, 0, 1, 1] = 1.0
    b1[2 * nseg + 1] = XMAX
    w1[2 * nseg + 2, 0, 1, 1] = -1.0
    b1[2 * nseg + 2] = XMAX

    # ramp channels: for segment j we want relu(k_j*x - s_j + c_j)
    # where s_j = tau-dependent shift.  We cannot multiply k_j*tau, but we
    # can add a per-example offset: the conv1 pre-activation is
    #     k_j*x + w_tau*tau + b
    # so we set w_tau = -s_j (constant per segment) and b = c_j.
    # The window is then centred at x = s_j/k_j.  We place the window so
    # that it covers the whole relevant x range for that segment.
    for j in range(nseg):
        kj = seg_gain[j]
        # choose window centre so that the segment's gain applies where it matters
        # centre the window at 0 -> s_j = 0, c_j = XMAX/kj (so |k_j x| < c_j)
        c_j = XMAX / max(kj, 1e-3) + 1.0
        w1[j, 0, 1, 1] = kj
        b1[j] = c_j
        w1[nseg + j, 0, 1, 1] = kj
        b1[nseg + j] = -c_j

    # --- conv2: combine ramp pairs into per-segment gain channels
    # out2[j] = relu(k_j x + c_j) - relu(k_j x - c_j) = 2 k_j x  (|x|<XMAX)
    # we normalise later; keep magnitudes moderate.
    w2 = np.zeros((64, 32, 3, 3), dtype=np.float32)
    b2 = np.zeros(64, dtype=np.float32)
    for j in range(nseg):
        w2[j, j, 1, 1] = 1.0
        w2[j, nseg + j, 1, 1] = -1.0
    # also carry the tau channel and x passthrough into conv2 channels
    w2[2 * nseg, 2 * nseg, 1, 1] = 1.0
    w2[2 * nseg + 1, 2 * nseg + 1, 1, 1] = 1.0
    w2[2 * nseg + 2, 2 * nseg + 2, 1, 1] = 1.0

    # --- conv3: identity-ish on the 2*nseg gain channels + tau channel
    w3 = np.zeros((64, 64, 3, 3), dtype=np.float32)
    b3 = np.zeros(64, dtype=np.float32)
    for j in range(nseg):
        w3[j, j, 1, 1] = 1.0
    w3[2 * nseg, 2 * nseg, 1, 1] = 1.0

    w_new = {
        'conv1.weight': w1, 'conv1.bias': b1,
        'conv2.weight': w2, 'conv2.bias': b2,
        'conv3.weight': w3, 'conv3.bias': b3,
        'out.weight': np.zeros((1, 64, 3, 3), dtype=np.float32),
        'out.bias': np.zeros(1, dtype=np.float32),
    }

    # ================================================================
    # Closed-form fit of the last layer on the recomputed features
    # ================================================================
    # features h3: [N,64,16,16] -> design rows per pixel
    # out(p) = sum_{c,k,l} W[c,k,l] * h3[c, p+(k-1,l-1)] + bias
    # We build the normal equations directly (577 x 577) using the
    # shift-correlation structure, avoiding a huge patch matrix.

    def build_and_fit(wd, lam=1e-3, ridge_on_bias=True):
        _, h3 = _forward(x, t, wd)
        Nf, C3, Hf, Wf = h3.shape
        F = h3.reshape(Nf, C3, Hf * Wf)
        # padded feature map for shift access
        Fp = np.pad(F, ((0, 0), (0, 0), (1, 1)))
        # design columns: for each (c, dy, dx) -> Fp[c, p + (dy,dx)]
        # index arithmetic: p = i*Wf + j ; shifted index = (i+dy-1)*Wf + (j+dx-1)
        idx = np.arange(Hf * Wf)
        ii = idx // Wf
        jj = idx % Wf
        cols = []          # list of [N, C3, P] blocks? too big -> build Gram directly
        # Gram: G[(c,dy,dx),(c',dy',dx')] = sum_n sum_p Fp[c,p+s1]*Fp[c',p+s2]
        # We compute per (shift1, shift2) pair the channel-channel correlation.
        G = np.zeros((C3 * 9, C3 * 9), dtype=np.float64)
        rhs = np.zeros(C3 * 9, dtype=np.float64)
        yy = y.reshape(Nf, Hf * Wf)
        # precompute shifted views lazily
        shifts = [(dy - 1, dx - 1) for dy in range(3) for dx in range(3)]
        views = []
        for (dy, dx) in shifts:
            vi = np.clip(ii + dy, 0, Hf - 1)
            vj = np.clip(jj + dx, 0, Wf - 1)
            vidx = vi * Wf + vj
            views.append(Fp[:, :, vidx])       # [N, C3, P]
        for a in range(9):
            Va = views[a]                      # [N,C,P]
            for b in range(a, 9):
                Vb = views[b]
                # sum over n,p of Va[c]*Vb[c']  -> [C,C']
                blk = np.einsum('ncp,nep->ce', Va, Vb, optimize=True) / float(Nf * Hf * Wf)
                G[a * C3:(a + 1) * C3, b * C3:(b + 1) * C3] = blk
                if b != a:
                    G[b * C3:(b + 1) * C3, a * C3:(a + 1) * C3] = blk.T
                rb = np.einsum('ncp,np->c', Va, yy, optimize=True) / float(Nf * Hf * Wf)
                if a == 0:
                    rhs[0:C3] += rb
                if b == 0:
                    rhs[b * C3:(b + 1) * C3] += rb
        # solve with bias handled by augmenting (bias = extra column of ones)
        # G_aug = [[G, s],[s^T, n]], rhs_aug = [rhs, sum(y)]
        s = G.sum(axis=1)
        n = float(Nf * Hf * Wf)
        Ga = np.zeros((C3 * 9 + 1, C3 * 9 + 1), dtype=np.float64)
        Ga[:-1, :-1] = G
        Ga[:-1, -1] = s
        Ga[-1, :-1] = s
        Ga[-1, -1] = n
        ra = np.zeros(C3 * 9 + 1, dtype=np.float64)
        ra[:-1] = rhs
        ra[-1] = float(yy.sum())
        d = C3 * 9 + 1
        Ga.flat[:: d + 1] += lam * n / 1e4
        sol = _ridge_solve(Ga, ra, 0.0)
        Wo = sol[:-1].reshape(9, C3).T      # [C3, 9] -> (c, dy*3+dx)
        Wo = Wo.reshape(C3, 3, 3)
        bo = np.float32(sol[-1])
        return Wo.astype(np.float32), bo, G, rhs

    Wo, bo, G, rhs = build_and_fit(w_new, lam=1e-4)
    w_new['out.weight'] = Wo[None]
    w_new['out.bias'] = np.array([bo], dtype=np.float32)

    # evaluate
    try:
        p1, _ = _forward(x, t, w_new)
        diag['mse_after'] = float(np.mean((p1 - y) ** 2))
    except Exception:
        diag['mse_after'] = None

    # ================================================================
    # Fallback / refinement: if the analytic construction is poor, fall
    # back to a plain last-layer ridge fit on the ORIGINAL features.
    # ================================================================
    if diag.get('mse_after') is None or diag['mse_after'] > (diag.get('mse_before') or 1e9):
        # original features (parent) + closed-form last layer
        feats = np.asarray(f['features'], dtype=np.float32)
        Nf, C3, Hf, Wf = feats.shape
        F = feats.reshape(Nf, C3, Hf * Wf)
        Fp = np.pad(F, ((0, 0), (0, 0), (1, 1)))
        idx = np.arange(Hf * Wf)
        ii = idx // Wf
        jj = idx % Wf
        shifts = [(dy - 1, dx - 1) for dy in range(3) for dx in range(3)]
        views = []
        for (dy, dx) in shifts:
            vi = np.clip(ii + dy, 0, Hf - 1)
            vj = np.clip(jj + dx, 0, Wf - 1)
            views.append(Fp[:, :, vi * Wf + vj])
        G = np.zeros((C3 * 9, C3 * 9), dtype=np.float64)
        rhs = np.zeros(C3 * 9, dtype=np.float64)
        yy = y.reshape(Nf, Hf * Wf)
        for a in range(9):
            Va = views[a]
            for b in range(a, 9):
                Vb = views[b]
                blk = np.einsum('ncp,nep->ce', Va, Vb, optimize=True) / float(Nf * Hf * Wf)
                G[a * C3:(a + 1) * C3, b * C3:(b + 1) * C3] = blk
                if b != a:
                    G[b * C3:(b + 1) * C3, a * C3:(a + 1) * C3] = blk.T
                rb = np.einsum('ncp,np->c', Va, yy, optimize=True) / float(Nf * Hf * Wf)
                if a == 0:
                    rhs[0:C3] += rb
                if b == 0:
                    rhs[b * C3:(b + 1) * C3] += rb
        s = G.sum(axis=1)
        n = float(Nf * Hf * Wf)
        Ga = np.zeros((C3 * 9 + 1, C3 * 9 + 1), dtype=np.float64)
        Ga[:-1, :-1] = G
        Ga[:-1, -1] = s
        Ga[-1, :-1] = s
        Ga[-1, -1] = n
        ra = np.zeros(C3 * 9 + 1, dtype=np.float64)
        ra[:-1] = rhs
        ra[-1] = float(yy.sum())
        d = C3 * 9 + 1
        Ga.flat[:: d + 1] += 1e-4 * n / 1e4
        sol = _ridge_solve(Ga, ra, 0.0)
        Wo = sol[:-1].reshape(9, C3).T.reshape(C3, 3, 3).astype(np.float32)
        w_new = {k: np.array(v, dtype=np.float32, copy=True) for k, v in w.items()}
        w_new['out.weight'] = Wo[None]
        w_new['out.bias'] = np.array([np.float32(sol[-1])], dtype=np.float32)
        p2, _ = _forward(x, t, w_new)
        diag['mse_after'] = float(np.mean((p2 - y) ** 2))
        diag['mode'] = 'fallback_lastlayer_ridge'

    # ---- final safety: clip, finite, dtype
    out = {}
    for k, v in w_new.items():
        v = np.asarray(v, dtype=np.float32)
        if not np.all(np.isfinite(v)):
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        v = np.clip(v, -1000.0, 1000.0).astype(np.float32)
        out[k] = v

    # final verification with the returned weights
    try:
        pf, _ = _forward(x, t, out)
        diag['mse_final'] = float(np.mean((pf - y) ** 2))
        # per-timestep mse
        pt = []
        for tt in range(32):
            m = (t == tt)
            if m.sum() > 0:
                pt.append(float(np.mean((pf[m] - y[m]) ** 2)))
        diag['mse_by_t_first8'] = [round(v, 4) for v in pt[:8]]
        diag['mse_by_t_last8'] = [round(v, 4) for v in pt[-8:]]
    except Exception:
        pass

    return {"weights": out, "diagnostics": diag}
