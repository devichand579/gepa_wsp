import numpy as np

def edit(w, f):
    # ------------------------------------------------------------------
    # Target: implement the analytic epsilon-predictor  eps_hat = c(t)*x_t
    # inside the fixed conv architecture, then refine the last layer by
    # closed-form ridge least squares on the supplied training data.
    # ------------------------------------------------------------------
    x   = np.asarray(f['x'], dtype=np.float64)        # [N,1,16,16]
    t   = np.asarray(f['t'], dtype=np.int64)          # [N]
    y   = np.asarray(f['y'], dtype=np.float64)        # [N,1,16,16]
    ab  = np.asarray(f['alphabar'], dtype=np.float64)  # [32]
    N   = x.shape[0]

    # ---- per-timestep optimal linear coefficient (closed form) --------
    # x_t = sqrt(ab)*x0 + sqrt(1-ab)*eps  =>  eps = (x_t - sqrt(ab) x0)/sqrt(1-ab)
    # E[eps|x_t] ~ c(t) x_t with c(t)=sqrt(1-ab)/(ab*V0 + (1-ab)), V0=Var(x0).
    # V0 estimated from the clean images (per-timestep, using the given data).
    clean = np.asarray(f['clean'], dtype=np.float64)
    V0 = float(np.mean(clean.reshape(N, -1).var(axis=1)))   # per-example pixel variance
    V0 = max(V0, 1e-4)
    abt = np.clip(ab, 1e-8, 1.0 - 1e-8)
    c_t = np.sqrt(1.0 - abt) / (abt * V0 + (1.0 - abt))    # [32]

    # ---- design the network ------------------------------------------
    # conv1: 32 channels
    #   ch 0 : relu(+x_t)          (center tap)
    #   ch 1 : relu(-x_t)          (center tap)
    #   ch 2+k : relu(k - t) = h_k, k = 0..29   -> 30 step channels
    # conv2: for every timestep i we need the indicator g_i = h_i - h_{i-1}
    #   (h_{-1} = 0).  We build the products
    #       g_i * relu(x)      = relu(u + M*g_i - M) - relu(u - M)
    #       g_i * relu(-x)     = relu(v + M*g_i - M) - relu(v - M)
    #   but conv2 has only 64 output channels and we need 2 products per
    #   timestep (64 needed) -> we use the symmetry relu(-x) = -x + relu(x)
    #   to stay within budget:  conv2 produces, per timestep i:
    #       A_i = relu(u + M*g_i - M)   (u = relu(x))
    #       B_i = relu(u + M*g_i - M - M)?? ...
    #   -> instead we exploit:  g_i*x = g_i*relu(x) - g_i*relu(-x) and
    #      g_i*relu(-x) = g_i*(relu(x) - x)  -> needs x itself.
    #   Practical exact scheme (fits in 64 channels):
    #       conv2 out o = 2i   : relu( u + M*g_i - M )        (u=relu(x))
    #       conv2 out o = 2i+1 : relu( u + M*g_i - M ) with the *other*
    #                             branch realised as relu(v + M*g_i - M), v=relu(-x)
    #   We therefore need 32*2 = 64 channels -> exactly the layer size.
    #   conv3/out then linearly recombine to c(t)*x.

    # constants
    M = 6.0   # bound on |x_t| used by the relu-product identity

    W1 = np.zeros((32, 2, 3, 3), dtype=np.float64)
    b1 = np.zeros(32, dtype=np.float64)
    W1[0, 0, 1, 1] = 1.0            # +x_t
    W1[1, 0, 1, 1] = -1.0           # -x_t
    # gate channels h_k = relu(k - t):  t = 31 * tau, tau broadcast channel
    for k in range(30):
        ch = 2 + k
        W1[ch, 1, 1, 1] = -31.0     # -31*tau = -t
        b1[ch] = float(k)           # relu(k - t)

    # conv2: 64 outputs = 32 timesteps x 2 branches
    W2 = np.zeros((64, 32, 3, 3), dtype=np.float64)
    b2 = np.zeros(64, dtype=np.float64)
    for i in range(32):
        # branch '+': relu( u + M*g_i - M ), u = relu(x_t)  (conv1 ch 0)
        o = 2 * i
        W2[o, 0, 1, 1] = 1.0
        # g_i = h_i - h_{i-1};  h_i = conv1 ch (2+i) if i<30 else 0
        if i < 30:
            W2[o, 2 + i, 1, 1] += M
        if i >= 1:
            W2[o, 2 + i - 1, 1, 1] -= M
        b2[o] = -M
        # branch '-': relu( v + M*g_i - M ), v = relu(-x_t) (conv1 ch 1)
        o = 2 * i + 1
        W2[o, 1, 1, 1] = 1.0
        if i < 30:
            W2[o, 2 + i, 1, 1] += M
        if i >= 1:
            W2[o, 2 + i - 1, 1, 1] -= M
        b2[o] = -M

    # conv3 (64->64) and out (64->1): linear recombination.
    # We want, for a given timestep i, the output to equal c_i * x_t.
    # From conv2 we have (per pixel, for the active timestep i):
    #    A_i = relu(u + M*g_i - M) = u            (g_i=1, u<=M)
    #    B_i = relu(v + M*g_i - M) = v            (g_i=1, v<=M)
    # with u=relu(x), v=relu(-x), and u - v = x.
    # So c_i * x = c_i * (A_i - B_i)  -> conv3 selects A_i and B_i with
    # weights +c_i and -c_i and passes them through; out then sums.
    # conv3 has 64 outputs and must select per-timestep pairs; use
    # conv3 as identity on conv2 channels (64->64), then out (64->1)
    # applies the per-timestep coefficients c_i (only the active i
    # contributes because the other timesteps' channels are zero).
    W3 = np.zeros((64, 64, 3, 3), dtype=np.float64)
    b3 = np.zeros(64, dtype=np.float64)
    for o in range(64):
        W3[o, o, 1, 1] = 1.0

    Wo = np.zeros((1, 64, 3, 3), dtype=np.float64)
    bo = np.zeros(1, dtype=np.float64)
    for i in range(32):
        Wo[0, 2 * i, 1, 1] += c_t[i]
        Wo[0, 2 * i + 1, 1, 1] -= c_t[i]

    # ------------------------------------------------------------------
    # numpy forward (cross-correlation, pad 1)
    # ------------------------------------------------------------------
    def conv2d(inp, W, b):
        # inp [N,C,H,W], W [O,C,3,3]
        N, C, H, Wd = inp.shape
        O = W.shape[0]
        p = np.zeros((N, C, H + 2, Wd + 2), dtype=np.float64)
        p[:, :, 1:-1, 1:-1] = inp
        out = np.zeros((N, O, H, Wd), dtype=np.float64)
        for kr in range(3):
            for kc in range(3):
                out += np.einsum('nchw,oc->nohw', p[:, :, kr:kr + H, kc:kc + Wd], W[:, :, kr, kc])
        out += b.reshape(1, -1, 1, 1)
        return out

    def forward(xin, tvec, W1_, b1_, W2_, b2_, W3_, b3_, Wo_, bo_):
        tau = (tvec.astype(np.float64) / 31.0).reshape(-1, 1, 1, 1)
        inp = np.concatenate([xin, np.broadcast_to(tau, (xin.shape[0], 1, 16, 16))], axis=1)
        h1 = np.maximum(conv2d(inp, W1_, b1_), 0.0)
        h2 = np.maximum(conv2d(h1, W2_, b2_), 0.0)
        h3 = np.maximum(conv2d(h2, W3_, b3_), 0.0)
        return conv2d(h3, Wo_, bo_)

    # ------------------------------------------------------------------
    # closed-form refinement of the LAST layer only (ridge least squares)
    # on the supplied training data.  conv1..conv3 are frozen, so the
    # cached features are exactly reproduced by our forward pass.
    # ------------------------------------------------------------------
    # build conv3 output (h3) for all training examples, in chunks
    N = x.shape[0]
    h3_all = np.empty((N, 64, 16, 16), dtype=np.float32)
    chunk = 250
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        tau = (t[s:e].astype(np.float64) / 31.0).reshape(-1, 1, 1, 1)
        inp = np.concatenate([x[s:e], np.broadcast_to(tau, (e - s, 1, 16, 16))], axis=1)
        h1 = np.maximum(conv2d(inp, W1, b1), 0.0)
        h2 = np.maximum(conv2d(h1, W2, b2), 0.0)
        h3_all[s:e] = np.maximum(conv2d(h2, W3, b3), 0.0).astype(np.float32)

    # design matrix rows: per (example, pixel) -> 576 features (3x3x64) + bias
    # accumulate normal equations in chunks (never materialise the full matrix)
    D = 64 * 9
    A = np.zeros((D + 1, D + 1), dtype=np.float64)
    rhs = np.zeros(D + 1, dtype=np.float64)
    yy = y.reshape(N, -1).astype(np.float64)

    def im2col_rows(h3c):
        # h3c [n,64,16,16] -> [n*256, 576]
        n = h3c.shape[0]
        p = np.zeros((n, 64, 18, 18), dtype=np.float32)
        p[:, :, 1:-1, 1:-1] = h3c
        cols = np.empty((n, 64, 3, 3, 16, 16), dtype=np.float32)
        for kr in range(3):
            for kc in range(3):
                cols[:, :, kr, kc] = p[:, :, kr:kr + 16, kc:kc + 16]
        return cols.transpose(0, 4, 5, 1, 2, 3).reshape(n * 256, D)

    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        R = im2col_rows(h3_all[s:e]).astype(np.float64)
        b = yy[s:e].reshape(-1)
        R = np.concatenate([R, np.ones((R.shape[0], 1))], axis=1)
        A += R.T @ R
        rhs += R.T @ b

    # ridge for stability
    tr = np.trace(A[:D, :D]) / max(D, 1)
    lam = 1e-6 * max(tr, 1e-12)
    A[:D, :D] += lam * np.eye(D)
    sol = np.linalg.solve(A, rhs)
    Wo_new = sol[:D].reshape(1, 64, 3, 3)
    bo_new = np.array([sol[D]], dtype=np.float64)

    # ------------------------------------------------------------------
    # evaluate the analytic design and the refined design on training data
    # ------------------------------------------------------------------
    def mse_of(Wo_, bo_):
        tot = 0.0
        for s in range(0, N, chunk):
            e = min(N, s + chunk)
            p = forward(x[s:e], t[s:e], W1, b1, W2, b2, W3, b3, Wo_, bo_)
            tot += float(np.sum((p - y[s:e]) ** 2))
        return tot / (N * 256)

    mse_analytic = mse_of(Wo, bo)
    mse_refined = mse_of(Wo_new, bo_new)

    if np.isfinite(mse_refined) and mse_refined < mse_analytic:
        Wo_use, bo_use = Wo_new, bo_new
    else:
        Wo_use, bo_use = Wo, bo
        mse_refined = mse_analytic

    # ------------------------------------------------------------------
    # write back
    # ------------------------------------------------------------------
    w['conv1.weight'] = W1.astype(np.float32)
    w['conv1.bias'] = b1.astype(np.float32)
    w['conv2.weight'] = W2.astype(np.float32)
    w['conv2.bias'] = b2.astype(np.float32)
    w['conv3.weight'] = W3.astype(np.float32)
    w['conv3.bias'] = b3.astype(np.float32)
    w['out.weight'] = Wo_use.astype(np.float32)
    w['out.bias'] = bo_use.astype(np.float32)

    for k in w:
        w[k] = np.clip(np.asarray(w[k], dtype=np.float32), -1000.0, 1000.0)

    diagnostics = {
        'mse_analytic_train': round(float(mse_analytic), 5),
        'mse_refined_train': round(float(mse_refined), 5),
        'V0': round(float(V0), 5),
        'c_t_head': [round(float(v), 4) for v in c_t[:6]],
        'c_t_tail': [round(float(v), 4) for v in c_t[-6:]],
    }
    return {'weights': w, 'diagnostics': diagnostics}
