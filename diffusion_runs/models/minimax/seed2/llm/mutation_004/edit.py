import numpy as np

def edit(w, f):
    # ---------------- helpers ----------------
    def conv3x3(x, W, b):
        # x: [N,C,H,W], W: [O,C,3,3] -> [N,O,H,W] cross-correlation, pad 1
        N, C, H, Wd = x.shape
        O = W.shape[0]
        xp = np.pad(x, ((0,0),(0,0),(1,1),(1,1)))
        out = np.zeros((N, O, H, Wd), dtype=np.float64)
        for kr in range(3):
            for kc in range(3):
                patch = xp[:, :, kr:kr+H, kc:kc+Wd]        # [N,C,H,W]
                # patch: [N,C,H,W] -> contract C with W[o,c,kr,kc]
                out += np.tensordot(patch, W[:, :, kr, kc], axes=([1],[1])).transpose(1,0,2,3)
        return out + b.reshape(1, O, 1, 1)

    def forward(ww, x, t):
        N = x.shape[0]
        tb = (t.astype(np.float64) / 31.0).reshape(N, 1, 1, 1)
        inp = np.concatenate([x.astype(np.float64), np.broadcast_to(tb, (N,1,x.shape[2],x.shape[3]))], axis=1)
        h1 = np.maximum(conv3x3(inp, ww['conv1.weight'], ww['conv1.bias']), 0.0)
        h2 = np.maximum(conv3x3(h1, ww['conv2.weight'], ww['conv2.bias']), 0.0)
        h3 = np.maximum(conv3x3(h2, ww['conv3.weight'], ww['conv3.bias']), 0.0)
        o  = conv3x3(h3, ww['out.weight'], ww['out.bias'])
        return inp, h1, h2, h3, o

    # ---------------- data ----------------
    x   = np.asarray(f['x'], dtype=np.float64)          # [N,1,16,16]
    t   = np.asarray(f['t']).astype(np.int64)
    y   = np.asarray(f['y'], dtype=np.float64)          # [N,1,16,16]
    clean = np.asarray(f['clean'], dtype=np.float64)
    ab  = np.asarray(f['alphabar'], dtype=np.float64)   # [32]
    N = x.shape[0]

    ww = {k: np.asarray(v, dtype=np.float64).copy() for k, v in w.items()}

    # sanity: reproduce current prediction on the batch
    ids = np.asarray(f['batch.ids']).astype(np.int64)
    _,_,_,_, pred_b = forward(ww, x[ids], t[ids])
    ref_b = np.asarray(f['batch.prediction'] if 'batch.prediction' in f else f['prediction'][ids],
                       dtype=np.float64)
    sanity = float(np.mean((pred_b - ref_b)**2)) if ref_b.shape == pred_b.shape else float('nan')

    # ---------------- schedule ----------------
    T = len(ab)
    abar = np.clip(ab, 1e-6, 1.0 - 1e-6)
    sa = np.sqrt(abar); sb = np.sqrt(1.0 - abar)

    # ---------------- design: linear path from x_t to out ----------------
    # conv1: ch0 = ReLU(x_t + u0(t)), ch1 = ReLU(-x_t + u1(t)), ch2 = ReLU(x_t + u2(t))
    #         ch3 = ReLU(-x_t + u3(t)), ch4 = ReLU(x_t), ch5 = ReLU(-x_t)
    #         ch6 = ReLU(t/31 + c)  (t-only feature)
    # u_k(t) = s_k * (t/31) + b_k   -> chosen so that the ReLU "knee" sits at the
    # value that makes the readout able to realise the t-dependent gain.
    # We use a set of shifts spread over the range of x_t values.
    shifts = np.array([-1.2, -0.6, -0.2, 0.2, 0.6, 1.2])
    # channel layout
    #  0: +x with shift uA(t)   (uA = -g(t) * 0 ... set below)
    #  1: -x with shift uA(t)
    #  2: +x with shift uB(t)
    #  3: -x with shift uB(t)
    #  4: +x plain
    #  5: -x plain
    #  6: t-only
    #  7: +x with shift uC(t)
    #  8: -x with shift uC(t)
    #  9: +x with shift uD(t)
    # 10: -x with shift uD(t)
    # 11..: spare (zero)
    # For each pair (a,b) with sign s: channel = ReLU(s*x + u(t)) where u(t) = s_k*(t/31)+b_k

    # The gain we need: g(t) = 1/sb(t) (scale of x_t in eps).  We realise the
    # t-dependence of the *gain* by making the shift u(t) move the knee: with
    # channels ReLU(x+u) and ReLU(-x+u), the combination a*ReLU(x+u)+b*ReLU(-x+u)
    # equals (a+b)*x + const on |x|<u and saturates outside.  Using several
    # u values (driven by t) lets the fitted readout pick a t-dependent slope.
    # u(t) = -(t/31)*U  (moves knee from +U at t=0 to -U at t=31)
    U = 1.0
    # channel spec: (sign, u_scale)  u(t) = -u_scale*(t/31)
    spec = []
    for us in [0.0, 0.5, 1.0, 1.5]:
        spec.append((+1.0, us))
        spec.append((-1.0, us))
    spec.append((+1.0, 2.5))
    spec.append((-1.0, 2.5))
    n_spec = len(spec)   # 10
    # channel 10: t-only feature (constant per example) -> gives readout a t-dependent bias
    n_ch1 = 32

    W1 = np.zeros((n_ch1, 2, 3, 3))
    b1 = np.zeros(n_ch1)
    for k, (sg, us) in enumerate(spec):
        # x channel: delta at centre with weight sg
        W1[k, 0, 1, 1] = sg
        # t channel: all-ones kernel scaled so that sum = -us  (u(t) = -us*(t/31))
        W1[k, 1, :, :] = (-us) / 9.0
        b1[k] = 0.0
    # t-only channel (index n_spec): value = t/31 + 0.5 -> ReLU positive
    W1[n_spec, 1, :, :] = 1.0 / 9.0
    b1[n_spec] = 0.5

    # conv2: copy channels 0..n_spec to outputs 0..n_spec (delta kernel), rest zero
    W2 = np.zeros((64, n_ch1, 3, 3))
    b2 = np.zeros(64)
    for k in range(n_spec + 1):
        W2[k, k, 1, 1] = 1.0
    # conv3: copy channels 0..n_spec to outputs 0..n_spec
    W3 = np.zeros((64, 64, 3, 3))
    b3 = np.zeros(64)
    for k in range(n_spec + 1):
        W3[k, k, 1, 1] = 1.0

    # ---------------- ridge: x_t -> clean (per timestep group) ----------------
    # x_t = sa*clean + sb*eps  =>  E[clean|x_t] ~ linear map.  Use per-timestep
    # ridge regression on the 256-dim pixel vector (plus intercept).
    Xf = x.reshape(N, 256)
    Cf = clean.reshape(N, 256)
    tf = t
    x0hat = np.zeros((N, 256))
    for tt in range(T):
        m = tf == tt
        if m.sum() < 8:
            continue
        A = Xf[m]; B = Cf[m]
        G = A.T @ A + 1e-2 * np.eye(256) * (np.trace(A.T @ A) / 256.0 + 1e-12)
        try:
            sol = np.linalg.solve(G, A.T @ B)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(A, B, rcond=1e-6)[0]
        x0hat[m] = A @ sol
    # global fallback fit (all t) blended for small groups
    A = Xf; B = Cf
    G = A.T @ A + 1e-2 * np.eye(256) * (np.trace(A.T @ A) / 256.0 + 1e-12)
    try:
        solg = np.linalg.solve(G, A.T @ B)
    except np.linalg.LinAlgError:
        solg = np.linalg.lstsq(A, B, rcond=1e-6)[0]
    for tt in range(T):
        m = tf == tt
        if m.sum() < 8:
            x0hat[m] = Xf[m] @ solg

    # target epsilon implied by the x0 estimate (per example)
    sa_e = sa[t].reshape(N, 1, 1, 1)
    sb_e = sb[t].reshape(N, 1, 1, 1)
    eps_from_x0 = (x - sa_e * x0hat.reshape(N, 1, 16, 16)) / sb_e

    # ---------------- assemble candidate network & compute features ----------------
    # forward with the designed conv1/2/3 and the ORIGINAL out layer
    tmp = dict(ww)
    tmp['conv1.weight'] = W1; tmp['conv1.bias'] = b1
    tmp['conv2.weight'] = W2; tmp['conv2.bias'] = b2
    tmp['conv3.weight'] = W3; tmp['conv3.bias'] = b3
    inp, h1, h2, h3, _ = forward(tmp, x, t)

    # ---------------- closed-form fit of out layer on the training batch -------
    # design matrix rows: pixels of the batch; cols: 64*9 + 1
    # h3: [N,64,16,16] -> im2col columns
    def im2col(h):
        Nn, C, H, Wd = h.shape
        hp = np.pad(h, ((0,0),(0,0),(1,1),(1,1)))
        cols = np.empty((Nn, C*9, H*Wd), dtype=np.float64)
        idx = 0
        for kr in range(3):
            for kc in range(3):
                cols[:, idx*C:(idx+1)*C, :] = hp[:, :, kr:kr+H, kc:kc+Wd].reshape(Nn, C, H*Wd)
                idx += 1
        return cols

    cols = im2col(h3)                      # [N, 576, 256]
    Nn = cols.shape[0]
    # rows: example-pixel pairs
    A = np.transpose(cols, (0, 2, 1)).reshape(Nn*256, 576)
    A = np.concatenate([A, np.ones((Nn*256, 1))], axis=1)
    Tg = np.transpose(y.reshape(Nn, 1, 256), (0, 2, 1)).reshape(Nn*256)
    # target: use the model-implied epsilon (smooth, learnable) blended with the
    # true epsilon: we fit to the *conditional mean* proxy.
    Tg_model = np.transpose(eps_from_x0.reshape(Nn, 1, 256), (0, 2, 1)).reshape(Nn*256)
    # fit target: mixture of model-implied eps and true eps (weight on model)
    lam = 0.75
    Tfit = lam * Tg_model + (1.0 - lam) * Tg

    # ridge normal equations
    G = A.T @ A
    G[np.diag_indices_from(G)] += 1e-3 * np.trace(G) / G.shape[0]
    rhs = A.T @ Tfit
    try:
        sol = np.linalg.solve(G, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(A, Tfit, rcond=1e-8)[0]

    Wo = sol[:576].reshape(1, 64, 3, 3)
    bo = np.array([sol[576]])

    # ---------------- verify on all training data -------------------------
    new = dict(ww)
    new['conv1.weight'] = W1; new['conv1.bias'] = b1
    new['conv2.weight'] = W2; new['conv2.bias'] = b2
    new['conv3.weight'] = W3; new['conv3.bias'] = b3
    new['out.weight'] = Wo; new['out.bias'] = bo
    _,_,_,_, pred_new = forward(new, x, t)
    mse_new = float(np.mean((pred_new - y)**2))
    mse_old = float(np.mean((pred_new*0 - y)**2))

    # per-timestep mse (new)
    per_t = {}
    for tt in range(T):
        m = tf == tt
        per_t[str(tt)] = float(np.mean((pred_new[m] - y[m])**2))

    out = {}
    for k, v in w.items():
        arr = new[k].astype(np.float32)
        arr = np.clip(arr, -1000.0, 1000.0)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        out[k] = arr

    diagnostics = {
        "mse_new_full": mse_new,
        "mse_zero_baseline": mse_old,
        "mse_parent_batch_sanity": sanity,
        "per_timestep_mse": per_t,
        "n_train": int(N),
        "out_weight_norm": float(np.linalg.norm(Wo)),
    }
    return {"weights": out, "diagnostics": diagnostics}
