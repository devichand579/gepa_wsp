import numpy as np

# ----------------------------------------------------------------- helpers
def _conv2d(x, W, b):
    """PyTorch cross-correlation, NCHW, pad=1, stride 1."""
    N, C, H, Wd = x.shape
    O, C2, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    # im2col via stride tricks
    s = xp.strides
    patches = np.lib.stride_tricks.as_strided(
        xp, shape=(N, C, kh, kw, H, Wd),
        strides=(s[0], s[1], s[2], s[3], s[2], s[3]), writeable=False)
    cols = patches.transpose(0, 4, 5, 1, 2, 3).reshape(N * H * Wd, C * kh * kw)
    out = cols @ W.reshape(O, C * kh * kw).T + b
    return out.reshape(N, H, Wd, O).transpose(0, 3, 1, 2)


def _forward(w, x, t):
    """Exact forward pass of the fixed architecture."""
    tb = (np.asarray(t, dtype=np.float64) / 31.0).reshape(-1, 1, 1, 1)
    inp = np.concatenate([x.astype(np.float64), np.broadcast_to(tb, (x.shape[0], 1, x.shape[2], x.shape[3]))], axis=1)
    h = np.maximum(_conv2d(inp, w['conv1.weight'], w['conv1.bias']), 0.0)
    h = np.maximum(_conv2d(h, w['conv2.weight'], w['conv2.bias']), 0.0)
    h = np.maximum(_conv2d(h, w['conv3.weight'], w['conv3.bias']), 0.0)
    return _conv2d(h, w['out.weight'], w['out.bias'])


def _mse(pred, y):
    return float(np.mean((pred - y) ** 2))


# ------------------------------------------------------- design matrix build
def _design(features, chunk=64):
    """Return (G, Y) with G: (n*256, 576) float32, Y: (n*256,) float32.

    Row order: example-major, then pixel-major (row-major over 16x16).
    Column order: c*9 + kr*3 + kc  (c = channel, kr/kc = kernel row/col),
    matching out.weight[0, c, kr, kc].
    """
    n = features.shape[0]
    F = features.astype(np.float32)
    Fp = np.pad(F, ((0, 0), (0, 0), (1, 1), (1, 1)))
    Gs, Ys = [], []
    for i in range(0, n, chunk):
        Fb = Fp[i:i + chunk]
        m = Fb.shape[0]
        # sliding windows -> (m, C, 16, 16, 3, 3)
        sw = np.lib.stride_tricks.sliding_window_view(Fb, (3, 3), axis=(2, 3))
        # (m, C, 16, 16, 3, 3) -> (m, 16, 16, C, 3, 3)
        sw = sw.transpose(0, 2, 3, 1, 4, 5)
        G = np.ascontiguousarray(sw).reshape(m * 256, 64 * 9)
        Gs.append(G)
    G = np.concatenate(Gs, axis=0)
    return G


# ------------------------------------------------------------------- edit
def edit(w, f):
    w = {k: np.array(v, dtype=np.float32, copy=True) for k, v in w.items()}
    diag = {}

    x = np.asarray(f['x'], dtype=np.float64)
    y = np.asarray(f['y'], dtype=np.float64)
    t = np.asarray(f['t']).astype(np.int64)
    clean = np.asarray(f['clean'], dtype=np.float64)
    feats = np.asarray(f['features'], dtype=np.float32)
    abar = np.asarray(f['alphabar'], dtype=np.float64)
    n = x.shape[0]

    # ---------------- (1) ridge least-squares readout on cached features
    G = _design(feats)                      # (n*256, 576) float32
    Y = y.reshape(n * 256).astype(np.float32)
    N = G.shape[0]
    A = (G.T @ G).astype(np.float64)
    s = G.sum(axis=0).astype(np.float64)
    rhs = (G.T @ Y).astype(np.float64)
    sy = float(Y.sum(dtype=np.float64))
    M = np.empty((577, 577), dtype=np.float64)
    M[:576, :576] = A
    M[:576, 576] = s
    M[576, :576] = s
    M[576, 576] = float(N)
    b = np.concatenate([rhs, [sy]])
    lam = 1e-6 * (np.trace(A) / 576.0 + 1e-12)
    reg = np.zeros(577)
    reg[:576] = lam
    reg[576] = lam * 1e-3
    try:
        theta = np.linalg.solve(M + np.diag(reg), b)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(M + np.diag(reg), b, rcond=None)[0]
    w_read = dict(w)
    w_read['out.weight'] = theta[:576].reshape(1, 64, 3, 3).astype(np.float32)
    w_read['out.bias'] = np.array([theta[576]], dtype=np.float32)
    del G, A, M, b, theta

    # ---------------- (2) linear predictor of eps from x_t (per timestep)
    # y = (x_t - sqrt(abar) * x0) / sqrt(1-abar);  regress on x_t with a
    # prior-mean correction term  m_t = E[x0] estimated from clean images.
    m_img = clean.mean(axis=0).reshape(-1)          # (256,)
    xf = x.reshape(n, 256)
    yf = y.reshape(n, 256)
    # per-timestep ridge regression of y on [x_t, 1]  (256+1 params, ~62 rows)
    # -> use pooled structure instead: y ~ c(t) * x_t + d(t) * m_img + b(t)
    coef = {}
    for tt in range(32):
        idx = np.where(t == tt)[0]
        if len(idx) < 8:
            continue
        Xt = xf[idx]
        Yt = yf[idx]
        # regress Yt (256 outputs) on [Xt (256), m_img (1), 1] with ridge
        Areg = np.zeros((258, 258))
        Areg[:256, :256] = Xt.T @ Xt
        Areg[:256, 256] = Xt.T @ m_img
        Areg[:256, 257] = Xt.sum(axis=0)
        Areg[256, :256] = Xt.T @ m_img
        Areg[256, 257] = float(m_img @ m_img)
        Areg[257, :256] = Xt.sum(axis=0)
        Areg[257, 256] = float(m_img @ m_img)
        Areg[257, 257] = float(len(idx))
        B = np.zeros((258, 256))
        B[:256] = Xt.T @ Yt
        B[256] = m_img @ Yt
        B[257] = Yt.sum(axis=0)
        lam = 1e-3 * (np.trace(Areg) / 258.0 + 1e-12)
        try:
            sol = np.linalg.solve(Areg + lam * np.eye(258), B)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(Areg + lam * np.eye(258), B, rcond=None)[0]
        coef[tt] = sol
    diag['n_timesteps_fitted'] = len(coef)

    # ---------------- (3) realise a linear map exactly with the conv stack
    # relu(z) - relu(-z) = z  =>  paired channels carry a linear signal.
    # We realise  eps_hat = A_t x_t + b_t  (per-pixel, 3x3 local taps only
    # available) using: conv1 -> 32 ch (16 pairs), conv2 -> 64 ch (32 pairs),
    # conv3 -> 64 ch (32 pairs), out -> 1 ch (difference of a pair).
    # Because the readout is convolutional, only 3x3 local linear maps can be
    # realised; a global (dense) linear map is approximated by a per-pixel
    # scalar model:  eps_hat = c(t) * x_t + d(t) * m_img + e(t).
    # Build the linear predictor in "local" form using the pooled fit above.

    # pooled per-timestep scalar model (robust): regress y on [x_t, m_img, 1]
    # using all timesteps with t-dependent coefficients shrunk toward global.
    # global fit first
    Xall = np.concatenate([xf, np.tile(m_img, (n, 1)), np.ones((n, 1))], axis=1)
    Ag = Xall.T @ Xall
    Bg = Xall.T @ yf
    lamg = 1e-4 * (np.trace(Ag) / Ag.shape[0] + 1e-12)
    sg = np.linalg.solve(Ag + lamg * np.eye(Ag.shape[0]), Bg)
    diag['global_mse_fit'] = _mse((Xall @ sg).reshape(n, 1, 16, 16), y)

    # per-timestep shrunk estimates of (c_t, d_t, e_t) via scalar projections
    # (projection of x_t and m_img onto the dominant eps direction is not
    #  available, so use the dense per-timestep solution above, shrunk).
    lin = {}   # tt -> (Cmat(256x256) too big) -> use diagonal + rank-1 form
    for tt, sol in coef.items():
        idx = np.where(t == tt)[0]
        Xt = xf[idx]
        Yt = yf[idx]
        pred = Xt @ sol[:256] + m_img @ sol[256] + sol[257]
        lin[tt] = (sol[:256], sol[256], sol[257], float(np.mean((pred - Yt) ** 2)))

    # The dense 256x256 map is not realisable by a 3x3 conv, so restrict to a
    # local (3x3) linear filter per timestep: fit  eps ~ sum_{d in 3x3} w_d x_t
    # with a shared 3x3 kernel per timestep (9 params) + bias, plus a prior
    # mean term.  This IS realisable exactly by the conv stack.
    def _fit_local(idx):
        # build patch matrix: (n, 256*9)
        xb = x[idx]
        xb_p = np.pad(xb, ((0, 0), (0, 0), (1, 1), (1, 1)))
        sw = np.lib.stride_tricks.sliding_window_view(xb_p, (3, 3), axis=(2, 3))
        sw = sw.transpose(0, 2, 3, 1, 4, 5).reshape(len(idx), 256, 9)
        P = sw.reshape(len(idx) * 256, 9)
        Yt = yf[idx].reshape(-1)
        A = P.T @ P
        b = P.T @ Yt
        lam = 1e-3 * (np.trace(A) / 9.0 + 1e-12)
        wv = np.linalg.solve(A + lam * np.eye(9), b)
        res = float(np.mean((P @ wv - Yt) ** 2))
        return wv, res

    local = {}
    for tt in range(32):
        idx = np.where(t == tt)[0]
        if len(idx) < 8:
            continue
        wv, res = _fit_local(idx)
        local[tt] = (wv, res)
    diag['local_fit_mse'] = float(np.mean([v[1] for v in local.values()]))

    # ---------------- (4) build the "exact linear" network for the local map
    # eps_hat(pixel) = sum_{d} w_d * x_t(pixel+d) + bias
    # Implement with paired ReLU channels:
    #   conv1: 16 pairs (c, c+1): weights +-v_c on x_t channel, bias 0
    #           -> relu gives r_c, relu(-r_c); r_c - relu(-r_c) = v_c . x
    #   conv2: 32 pairs: pre-act s_c = sum_{c1} a_{c,c1} (r_c1 - r'_c1)
    #           -> relu(s_c), relu(-s_c)
    #   conv3: 32 pairs: pre-act u_d = sum_c b_{d,c} (relu(s_c)-relu(-s_c))
    #           -> relu(u_d), relu(-u_d)
    #   out:   sum over pairs of (relu(u_d) - relu(-u_d)) * coef_d
    # Simplify: choose the chain so that the final linear functional is
    #   sum_d coef_d * u_d  with u_d = (3x3 filter) . x_t
    # We only need ONE 3x3 filter per timestep, so use a single "signal"
    # channel pair at each stage and put the 9 taps into conv3's 9 kernel
    # positions of one output channel (paired), then out sums the pair.
    def build_exact_linear(filt3, bias):
        """filt3: (3,3) applied to x_t ; returns weight dict implementing
        eps_hat = conv3x3(filt3, x_t) + bias exactly."""
        W = {}
        # conv1: 2 channels used: ch0 = +filt (on x_t), ch1 = -filt
        c1w = np.zeros((32, 2, 3, 3), dtype=np.float32)
        c1w[0, 0] = filt3
        c1w[1, 0] = -filt3
        c1b = np.zeros(32, dtype=np.float32)
        # conv2: ch0 pre-act = relu(c1_0) - relu(c1_1) = filt.x ; ch1 = -(that)
        c2w = np.zeros((64, 32, 3, 3), dtype=np.float32)
        c2w[0, 0] = 1.0
        c2w[0, 1] = -1.0
        c2w[1, 0] = -1.0
        c2w[1, 1] = 1.0
        c2b = np.zeros(64, dtype=np.float32)
        # conv3: ch0 pre-act = relu(c2_0) - relu(c2_1) = filt.x ; ch1 = -(that)
        c3w = np.zeros((64, 64, 3, 3), dtype=np.float32)
        c3w[0, 0] = 1.0
        c3w[0, 1] = -1.0
        c3w[1, 0] = -1.0
        c3w[1, 1] = 1.0
        c3b = np.zeros(64, dtype=np.float32)
        # out: relu(c3_0) - relu(c3_1) = filt.x
        ow = np.zeros((1, 64, 3, 3), dtype=np.float32)
        ow[0, 0] = 1.0
        ow[0, 1] = -1.0
        ob = np.array([bias], dtype=np.float32)
        W['conv1.weight'] = c1w
        W['conv1.bias'] = c1b
        W['conv2.weight'] = c2w
        W['conv2.bias'] = c2b
        W['conv3.weight'] = c3w
        W['conv3.bias'] = c3b
        W['out.weight'] = ow
        W['out.bias'] = ob
        return W

    # Because the timestep is an input channel (t/31) and the network is
    # shared across timesteps, a per-timestep filter cannot be realised by a
    # single set of weights.  Instead build ONE global linear predictor that
    # uses x_t and the t-channel:  eps_hat = c * x_t + d * (t/31) * x_t ...
    # -> not linear in x_t with t-dependent coefficient unless we use the
    # t channel multiplicatively (impossible).  So use a global linear map
    # fitted on all data (t-independent), which is the best we can realise
    # exactly, and additionally keep the readout-refit network which DOES
    # see t through conv1's second input channel.
    # Global local 3x3 filter (shared across t):
    xb_p = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    sw = np.lib.stride_tricks.sliding_window_view(xb_p, (3, 3), axis=(2, 3))
    sw = sw.transpose(0, 2, 3, 1, 4, 5).reshape(n, 256, 9)
    P = sw.reshape(n * 256, 9)
    Yt = y.reshape(-1)
    Agl = P.T @ P
    bgl = P.T @ Yt
    lamg2 = 1e-4 * (np.trace(Agl) / 9.0 + 1e-12)
    wglob = np.linalg.solve(Agl + lamg2 * np.eye(9), bgl)
    diag['global_local_mse'] = float(np.mean((P @ wglob - Yt) ** 2))

    w_exact = build_exact_linear(wglob, 0.0)

    # ---------------- (5) evaluate candidates exactly on the training data
    # (use a subsample for speed of the forward pass, then full for the best)
    def eval_weights(W, idx=None):
        if idx is None:
            idx = np.arange(n)
        pr = _forward(W, x[idx], t[idx])
        return _mse(pr, y[idx])

    sub = np.arange(0, n, 4)          # 500 examples for candidate scoring
    mse_read = eval_weights(w_read, sub)
    mse_exact = eval_weights(w_exact, sub)
    diag['mse_readout_sub'] = mse_read
    diag['mse_exact_sub'] = mse_exact
    diag['mse_current_sub'] = eval_weights(w, sub)

    # blend of the two predictions (both are valid networks -> use a
    # convex combination of the two *output* functions is not expressible;
    # instead pick the better one, with a small safety margin)
    if mse_exact < mse_read:
        best = w_exact
        diag['chosen'] = 'exact_linear'
    else:
        best = w_read
        diag['chosen'] = 'readout_ridge'

    # ---------------- (6) final full-data check & diagnostics
    full = eval_weights(best, np.arange(n))
    diag['mse_full'] = full
    diag['mse_current_full'] = eval_weights(w, np.arange(n))
    # per-timestep breakdown
    per_t = {}
    pr = _forward(best, x, t)
    for tt in range(32):
        idx = np.where(t == tt)[0]
        if len(idx):
            per_t[str(tt)] = round(float(np.mean((pr[idx] - y[idx]) ** 2)), 4)
    diag['per_timestep'] = per_t

    out = {k: np.asarray(v, dtype=np.float32) for k, v in best.items()}
    for k, v in out.items():
        v = np.clip(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0), -1000.0, 1000.0)
    return {"weights": out, "diagnostics": diag}
