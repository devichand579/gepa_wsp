import numpy as np

def _conv3x3(x, W, b):
    # x: (N,C,H,W), W: (O,C,3,3) cross-correlation, pad 1
    N, C, H, Wd = x.shape
    O = W.shape[0]
    xp = np.zeros((N, C, H + 2, Wd + 2), dtype=np.float64)
    xp[:, :, 1:-1, 1:-1] = x
    out = np.zeros((N, O, H, Wd), dtype=np.float64)
    for kr in range(3):
        for kc in range(3):
            out += np.tensordot(xp[:, :, kr:kr + H, kc:kc + Wd], W[:, :, kr, kc], axes=([1], [1])).transpose(0, 3, 1, 2)
    return out + b[None, :, None, None]

def _forward(w, x, t):
    N = x.shape[0]
    u = (t.astype(np.float64) / 31.0)
    inp = np.concatenate([x.astype(np.float64), np.repeat(u[:, None, None, None], x.shape[2], axis=2).repeat(x.shape[3], axis=3)], axis=1)
    h1 = np.maximum(_conv3x3(inp, w['conv1.weight'], w['conv1.bias']), 0.0)
    h2 = np.maximum(_conv3x3(h1, w['conv2.weight'], w['conv2.bias']), 0.0)
    h3 = np.maximum(_conv3x3(h2, w['conv3.weight'], w['conv3.bias']), 0.0)
    o = _conv3x3(h3, w['out.weight'], w['out.bias'])
    return o

def edit(w, f):
    x = np.asarray(f['x'], dtype=np.float64)
    y = np.asarray(f['y'], dtype=np.float64)
    t = np.asarray(f['t']).astype(np.int64)
    N = x.shape[0]

    # ---------- target: pointwise affine fit in (x, u) with u-dependent coefficients ----------
    # basis for coefficient functions of u: [1, u, u^2]  ->  pred = sum_k g_k(u) * x + h(u)
    u = t.astype(np.float64) / 31.0
    B = np.stack([np.ones_like(u), u, u * u], axis=1)          # (N,3)
    # design: for each example, per-pixel regression of y on [x*B0, x*B1, x*B2, B0, B1, B2]
    # -> solve normal equations on all pixels of all examples
    nfeat = 6
    Xd = np.empty((N, 256, nfeat), dtype=np.float64)
    Xd[:, :, 0] = x[:, 0] * B[:, 0:1]
    Xd[:, :, 1] = x[:, 0] * B[:, 1:2]
    Xd[:, :, 2] = x[:, 0] * B[:, 2:3]
    Xd[:, :, 3] = B[:, 0:1]
    Xd[:, :, 4] = B[:, 1:2]
    Xd[:, :, 5] = B[:, 2:3]
    Xd = Xd.reshape(-1, nfeat)
    Yd = y[:, 0].reshape(-1)
    G = Xd.T @ Xd
    b = Xd.T @ Yd
    G[np.diag_indices(nfeat)] += 1e-6 * (np.trace(G) / nfeat + 1e-12)
    try:
        coef = np.linalg.solve(G, b)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(G, b, rcond=None)[0]
    # coef: [c0,c1,c2,d0,d1,d2] -> pred = (c0+c1 u+c2 u^2) x + (d0+d1 u+d2 u^2)
    c = coef[:3]
    d = coef[3:]

    def affine_pred(xx, tt):
        uu = tt.astype(np.float64) / 31.0
        a = c[0] + c[1] * uu + c[2] * uu * uu
        bb = d[0] + d[1] * uu + d[2] * uu * uu
        return a[:, None, None] * xx + bb[:, None, None]

    # ---------- design the network so it implements a hinge mixture approximating (a(u)x + b(u)) ----------
    # Architecture trick:
    #  conv1 ch0: pre = +x            -> relu(x)
    #  conv1 ch1: pre = -x            -> relu(-x)
    #  conv1 ch2: pre = c2*u + b2     -> constant (>=0)  (timestep signal)
    #  conv1 ch3: pre = -x + c3*u+b3  -> relu(-x + s(u))
    #  conv1 ch4: pre = x + c4*u+b4   -> relu(x + s(u))
    #  conv2 ch0: relu(x) - relu(-x) = x   (exact linear recovery of x)
    #  conv2 ch1: constant channel (timestep signal, linear in u)
    #  conv2 ch2..: relu(x + s(u)) - relu(-x + s(u)) = x + s(u)  (linear, u-dependent offset)
    #  conv3 channels: relu( x + s_j(u) ) and relu( -x + s_j(u) ) style hinges with u-dependent breakpoints
    #  out: linear combination of conv3 channels -> piecewise linear in x with u-dependent breakpoints
    #
    # We build the *implemented* function directly as a hinge mixture:
    #   pred = sum_j w_j * relu(x - d_j(u))   (plus optional constant term)
    # and then SOLVE for (w_j, d_j) coefficients by least squares on the training data.
    #
    # Parameterisation of d_j(u) = p_j + q_j * u  (affine in u), realised as:
    #   conv1 channel j: pre = x + (-p_j) + (-q_j)*u  -> relu(x - d_j(u))
    #   conv1 channel j': pre = -x + p_j + q_j*u     -> relu(-(x - d_j(u)))
    #   conv2 channel j: picks relu(x-d_j(u)) (identity on that channel)
    #   conv3 channel j: relu( conv2 ch j )  (already >=0, stays >=0)
    #   out: weight w_j on channel j
    # The hinge mixture with breakpoints p_j + q_j*u can represent a(u)*x + b(u) exactly if we allow
    # the "signed" pair trick: relu(x-d) - relu(-x+d) = x - d  (slope 1) and
    # relu(x-d) + relu(-x+d) = |x-d| (slope +-1).
    # => Using both signs we can realise arbitrary piecewise-linear slopes in x whose breakpoints
    #    move with u, i.e. a(u)*x + b(u) approximated by a fine grid of moving breakpoints.

    # Build a fine grid of moving breakpoints: for each of K base levels, two moving breakpoints.
    # We solve for the out-layer weights by least squares given fixed hinge features.
    K = 16                      # number of hinge pairs
    # breakpoints d_j(u) = p_j + q_j*u, chosen so that the union of breakpoints spans the x range
    xmin = float(np.percentile(x, 0.5)); xmax = float(np.percentile(x, 99.5))
    # base grid in x
    grid = np.linspace(xmin, xmax, K + 1)
    # u-dependence: allow each breakpoint to move with u (slope q_j), fitted later
    q = np.zeros(K)

    # Precompute hinge features for a candidate set of (p,q) and fit out weights.
    # We instead directly fit a *piecewise-linear* model: pred = sum_j w_j*relu(x - p_j - q_j*u)
    # with p_j fixed on the grid and q_j fitted by a small search-free linearization:
    #   treat each hinge as having unknown q_j; linearize relu(x - p_j - q_j u) around q_j=0:
    #   relu(z - q_j u) ~ relu(z) - q_j u * 1[z>0]  (subgradient, valid for small q_j)
    # Then the model is linear in (w_j, w_j q_j): pred = sum_j a_j relu(x-p_j) + sum_j b_j u 1[x>p_j]
    # which is linear in (a_j, b_j) -> closed form least squares. Then q_j = -b_j/a_j.
    nH = K
    Z = (x[:, 0] > grid[None, :])                     # (N,256,K) indicators
    R = np.maximum(x[:, 0][:, :, None] - grid[None, None, :], 0.0)   # (N,256,K) relu(x-p_j)
    # columns: relu terms (K) and u*indicator terms (K)
    cols = []
    for j in range(nH):
        cols.append(R[:, :, j].reshape(-1))
    for j in range(nH):
        cols.append((u[:, None] * Z[:, :, j]).reshape(-1))
    A = np.stack(cols, axis=1)
    Yv = y[:, 0].reshape(-1)
    Gm = A.T @ A
    bm = A.T @ Yv
    Gm[np.diag_indices(Gm.shape[0])] += 1e-8 * (np.trace(Gm) / Gm.shape[0] + 1e-12)
    try:
        sol = np.linalg.solve(Gm, bm)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(Gm, bm, rcond=None)[0]
    a_coef = sol[:nH]
    b_coef = sol[nH:]
    q_hat = np.where(np.abs(a_coef) > 1e-8, -b_coef / np.where(np.abs(a_coef) > 1e-8, a_coef, 1.0), 0.0)
    q_hat = np.clip(q_hat, -3.0, 3.0)

    # ---- realise the hinge mixture in the fixed architecture ----
    # channel layout of conv1 (32 channels):
    #   0: +x                          (relu(x))
    #   1: -x                          (relu(-x))
    #   2: constant u channel: pre = u  (t broadcast)  -> relu(u)=u
    #   3: constant 1 channel: pre = 1  -> relu(1)=1
    #   4 + 2j : pre = x - p_j - q_j*u        -> relu(x - d_j(u))
    #   5 + 2j : pre = -x + p_j + q_j*u       -> relu(-(x - d_j(u)))
    # conv2 (64 channels):
    #   0: relu(x) - relu(-x) = x                 (exact)
    #   1: u channel (linear in u)
    #   2: constant 1 channel
    #   3 + j: relu(x - d_j(u))                  (copy of conv1 ch 4+2j)
    #   4 + j: relu(-(x - d_j(u)))               (copy of conv1 ch 5+2j)
    # conv3 (64 channels): copy selected conv2 channels (identity, values >=0)
    # out: weights combining conv3 channels to produce sum_j a_j*relu(x-d_j(u)) + c0*x + c1*u + c2*1
    #
    # To keep the number of conv3 channels <= 64 we use nH <= 30 hinge pairs.
    nH = min(nH, 30)
    a_coef = a_coef[:nH]; q_hat = q_hat[:nH]
    grid_used = grid[:nH]

    W1 = np.zeros((32, 2, 3, 3)); b1 = np.zeros(32)
    W1[0, 0] = 1.0                      # +x
    W1[1, 0] = -1.0                     # -x
    W1[2, 1] = 1.0 / 9.0                # u channel (broadcast, 3x3 sum = 1)
    W1[3, 1] = 0.0; b1[3] = 1.0         # constant 1
    for j in range(nH):
        W1[4 + 2 * j, 0] = 1.0
        W1[4 + 2 * j, 1] = -q_hat[j] / 9.0
        b1[4 + 2 * j] = -grid_used[j]
        W1[5 + 2 * j, 0] = -1.0
        W1[5 + 2 * j, 1] = q_hat[j] / 9.0
        b1[5 + 2 * j] = grid_used[j]

    W2 = np.zeros((64, 32, 3, 3)); b2 = np.zeros(64)
    # conv2 ch0 = conv1 ch0 - conv1 ch1  -> x
    W2[0, 0] = 1.0; W2[0, 1] = -1.0
    # conv2 ch1 = conv1 ch2 -> u ; ch2 = conv1 ch3 -> 1
    W2[1, 2] = 1.0
    W2[2, 3] = 1.0
    for j in range(nH):
        W2[3 + j, 4 + 2 * j] = 1.0
        W2[4 + j, 5 + 2 * j] = 1.0

    W3 = np.zeros((64, 64, 3, 3)); b3 = np.zeros(64)
    # conv3 ch0 = conv2 ch0 (x) ; ch1 = conv2 ch1 (u) ; ch2 = conv2 ch2 (1)
    W3[0, 0] = 1.0
    W3[1, 1] = 1.0
    W3[2, 2] = 1.0
    # conv3 ch(3+j) = conv2 ch(3+j) (hinge +) ; ch(3+nH+j) = conv2 ch(4+j) (hinge -)
    for j in range(nH):
        W3[3 + j, 3 + j] = 1.0
        W3[3 + nH + j, 4 + j] = 1.0

    Wo = np.zeros((1, 64, 3, 3)); bo = np.zeros(1)
    # out = sum_j a_j * hinge_plus_j + c_x * x + c_u * u + c_1 * 1
    # (the affine part (c0+c1 u+c2 u^2)x + (d0+d1 u+d2 u^2) is added via the x/u/1 channels)
    for j in range(nH):
        Wo[0, 3 + j] = a_coef[j]
    # affine coefficients expressed with x, u, 1 channels (approximate u^2 by u on [0,1] -> use two u channels)
    # we have only one u channel in conv3; add a second u channel to reach u^2 accuracy
    # (conv3 ch1 = u). We instead use the hinge-mixture fit alone plus x/u/1 with best linear projection.
    # Fit: target = a_coef-hinges + alpha*x + beta*u + gamma  (least squares on residual)
    resid = Yv - A[:, :nH] @ a_coef
    X2 = np.stack([x[:, 0].reshape(-1), (u[:, None] * np.ones((1, 256))).reshape(-1), np.ones(N * 256)], axis=1)
    G2 = X2.T @ X2
    b2v = X2.T @ resid
    G2[np.diag_indices(3)] += 1e-8 * (np.trace(G2) / 3 + 1e-12)
    try:
        s2 = np.linalg.solve(G2, b2v)
    except np.linalg.LinAlgError:
        s2 = np.linalg.lstsq(G2, b2v, rcond=None)[0]
    Wo[0, 0] = s2[0]      # x
    Wo[0, 1] = s2[1]      # u
    Wo[0, 2] = s2[2]      # 1
    bo[0] = 0.0

    new_w = dict(w)
    new_w['conv1.weight'] = W1.astype(np.float32)
    new_w['conv1.bias'] = b1.astype(np.float32)
    new_w['conv2.weight'] = W2.astype(np.float32)
    new_w['conv2.bias'] = b2.astype(np.float32)
    new_w['conv3.weight'] = W3.astype(np.float32)
    new_w['conv3.bias'] = b3.astype(np.float32)
    new_w['out.weight'] = Wo.astype(np.float32)
    new_w['out.bias'] = bo.astype(np.float32)

    # ---------- evaluate candidates with our own forward pass ----------
    def mse_of(weights):
        pred = _forward(weights, x, t)
        return float(np.mean((pred - y) ** 2))

    cand = []
    cand.append(('hinge', new_w))
    # candidate 2: exact affine (a(u)x + b(u)) realised via x/u/1 channels only (no hinges)
    W1b = W1.copy(); b1b = b1.copy()
    W2b = W2.copy(); b2b = b2.copy()
    W3b = W3.copy(); b3b = b3.copy()
    Wob = Wo.copy(); bob = bo.copy()
    for j in range(nH):
        W3b[3 + j] = 0.0; W3b[3 + nH + j] = 0.0
        Wob[0, 3 + j] = 0.0
    # fit affine coefficients with u^2 approximated by u (least squares with basis x, x*u, u, u^2, 1)
    Bb = np.stack([np.ones_like(u), u, u * u], axis=1)
    Xa = np.empty((N * 256, 6))
    Xa[:, 0] = (x[:, 0] * Bb[:, 0:1]).reshape(-1)
    Xa[:, 1] = (x[:, 0] * Bb[:, 1:2]).reshape(-1)
    Xa[:, 2] = (x[:, 0] * Bb[:, 2:3]).reshape(-1)
    Xa[:, 3] = Bb[:, 0:1].repeat(256, axis=1).reshape(-1)
    Xa[:, 4] = Bb[:, 1:2].repeat(256, axis=1).reshape(-1)
    Xa[:, 5] = Bb[:, 2:3].repeat(256, axis=1).reshape(-1)
    Ga = Xa.T @ Xa; ba = Xa.T @ Yv
    Ga[np.diag_indices(6)] += 1e-8 * (np.trace(Ga) / 6 + 1e-12)
    try:
        sa = np.linalg.solve(Ga, ba)
    except np.linalg.LinAlgError:
        sa = np.linalg.lstsq(Ga, ba, rcond=None)[0]
    # x channel gets (sa0 + sa1 u + sa2 u^2) -> approximate with x and x*u via a hinge-free trick:
    # use conv3 ch0 = x, and additionally a channel x*u is not available; use hinge channels to emulate.
    # Instead: keep only constant-in-u x coefficient sa0 and put u-dependent part into u channels (approx).
    Wob[0, 0] = sa[0]
    Wob[0, 1] = sa[4] + sa[1] * 0.0
    bob[0] = sa[3] + sa[5] * 0.0
    cand.append(('affine_simple', {'conv1.weight': W1b.astype(np.float32), 'conv1.bias': b1b.astype(np.float32),
                                    'conv2.weight': W2b.astype(np.float32), 'conv2.bias': b2b.astype(np.float32),
                                    'conv3.weight': W3b.astype(np.float32), 'conv3.bias': b3b.astype(np.float32),
                                    'out.weight': Wob.astype(np.float32), 'out.bias': bob.astype(np.float32)}))

    best_name, best_w, best_mse = None, None, np.inf
    for nm, cw in cand:
        mm = mse_of(cw)
        if np.isfinite(mm) and mm < best_mse:
            best_name, best_w, best_mse = nm, cw, mm

    # also consider keeping the original weights if nothing beats it
    orig_mse = mse_of({k: np.asarray(v, dtype=np.float64) for k, v in w.items()})
    if orig_mse < best_mse:
        best_name, best_w, best_mse = 'original', {k: np.asarray(v, dtype=np.float32) for k, v in w.items()}, orig_mse

    # enforce constraints
    out = {}
    for k, v in best_w.items():
        arr = np.asarray(v, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, -1000.0, 1000.0).astype(np.float32)
        out[k] = arr

    diag = {'chosen': str(best_name), 'train_mse_after': float(best_mse),
            'train_mse_before': float(orig_mse),
            'affine_coef_x': [float(v) for v in c],
            'affine_coef_c': [float(v) for v in d],
            'n_hinges': int(nH)}
    return {'weights': out, 'diagnostics': diag}
