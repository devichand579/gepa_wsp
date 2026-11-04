import numpy as np

def edit(w, f):
    # ---------------- helpers ----------------
    def conv2d(x, W, b):
        # x: (N,Cin,H,W), W: (Cout,Cin,3,3) cross-correlation, pad 1
        N, Cin, H, Wd = x.shape
        Cout = W.shape[0]
        xp = np.zeros((N, Cin, H + 2, Wd + 2), dtype=np.float32)
        xp[:, :, 1:-1, 1:-1] = x
        out = np.zeros((N, Cout, H, Wd), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                out += np.einsum('nchw,oc->nohw', xp[:, :, i:i + H, j:j + Wd],
                                 W[:, :, i, j], optimize=True)
        out += b.reshape(1, -1, 1, 1)
        return out

    def forward(x, t, Wd):
        # x: (N,1,16,16), t: (N,)
        N = x.shape[0]
        tb = (t.astype(np.float32) / 31.0).reshape(N, 1, 1, 1)
        inp = np.concatenate([x, np.broadcast_to(tb, (N, 1, 16, 16))], axis=1)
        h1 = np.maximum(conv2d(inp, Wd['conv1.weight'], Wd['conv1.bias']), 0.0)
        h2 = np.maximum(conv2d(h1, Wd['conv2.weight'], Wd['conv2.bias']), 0.0)
        h3 = np.maximum(conv2d(h2, Wd['conv3.weight'], Wd['conv3.bias']), 0.0)
        out = conv2d(h3, Wd['out.weight'], Wd['out.bias'])
        return out

    def mse(pred, y):
        return float(np.mean((pred - y) ** 2))

    x = np.asarray(f['x'], dtype=np.float32)
    t = np.asarray(f['t']).astype(np.int64)
    y = np.asarray(f['y'], dtype=np.float32)
    feats = np.asarray(f['features'], dtype=np.float32)   # (N,64,16,16) parent conv3 output
    pred0 = np.asarray(f['prediction'], dtype=np.float32)
    ab = np.asarray(f['alphabar'], dtype=np.float32)

    Wd = {k: np.asarray(v, dtype=np.float32).copy() for k, v in w.items()}
    base_mse = mse(pred0, y)

    # ---------------- 1. re-linearize early layers (fixed architecture) ----------------
    # Goal: make conv1/conv2/conv3 act (near-)linearly so that conv3 features are an
    # affine function of x_t, giving the (linear) readout a real 7x7 filter capability.
    # Construction: conv1 pre-activation = s1*(x) + c1(t) with a large positive bias M1
    # so ReLU is in its linear regime -> h1 = M1 + u1.
    # conv2: bias cancels the constant M1 -> h2 = M2 + W2 u1 ; conv3: h3 = M3 + W3 W2 u1.
    # out: bias cancels the constant M3 -> output = linear function of x_t and t.
    rng = np.random.default_rng(0)
    M1, M2, M3 = 60.0, 40.0, 20.0

    # conv1: small random filters on the x channel, t channel gets a random (t-dependent) offset
    s1 = 0.05
    W1x = (rng.standard_normal((32, 1, 3, 3)) * s1).astype(np.float32)
    W1t = (rng.standard_normal((32, 1, 3, 3)) * 0.02).astype(np.float32)
    W1 = np.concatenate([W1x, W1t], axis=1).astype(np.float32)
    b1 = np.full((32,), M1, dtype=np.float32)

    # conv2: small random filters; bias cancels constant M1 contribution
    W2 = (rng.standard_normal((64, 32, 3, 3)) * (0.05 / np.sqrt(32))).astype(np.float32)
    # constant contribution of h1 (M1) through conv2 with zero padding is position dependent;
    # use the interior value for the bias and let the (small) border mismatch be absorbed later.
    b2 = np.full((64,), M2, dtype=np.float32) - M1 * W2.sum(axis=(1, 2, 3))

    # conv3: small random filters; bias cancels M2 contribution
    W3 = (rng.standard_normal((64, 64, 3, 3)) * (0.05 / np.sqrt(64))).astype(np.float32)
    b3 = np.full((64,), M3, dtype=np.float32) - M2 * W3.sum(axis=(1, 2, 3))

    # out: keep the readout to be fitted; bias cancels M3 constant
    b_out0 = np.zeros((1,), dtype=np.float32)

    Wnew = {
        'conv1.weight': W1, 'conv1.bias': b1,
        'conv2.weight': W2, 'conv2.bias': b2,
        'conv3.weight': W3, 'conv3.bias': b3,
        'out.weight': Wd['out.weight'].copy(), 'out.bias': b_out0.copy(),
    }

    # ---------------- 2. closed-form least squares for the last layer ----------------
    # out(pixel) = sum_{c,i,j} W[c,i,j] * feat[c, y+i, x+j] + b
    # Build normal equations over all training examples / pixels, then solve with ridge.
    N, C, H, Wd_ = feats.shape
    K = C * 9
    A = np.zeros((K + 1, K + 1), dtype=np.float64)
    rhs = np.zeros((K + 1,), dtype=np.float64)
    yy = y[:, 0]                                     # (N,16,16)

    # patch extraction (im2col) done in chunks to bound memory
    chunk = 100
    for s0 in range(0, N, chunk):
        s1_ = min(s0 + chunk, N)
        F = feats[s0:s1_]                            # (n,64,16,16)
        n = F.shape[0]
        # build patches: (n, 64, 3, 3, 16,16) -> (n*256, 576)
        P = np.zeros((n, 64, 3, 3, H, Wd_), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                P[:, :, i, j, :, :] = F[:, :, i:i + H, j:j + Wd_]
        P = P.transpose(0, 4, 5, 1, 2, 3).reshape(n * H * Wd_, K)
        T = yy[s0:s1_].reshape(n * H * Wd_)
        A[:K, :K] += (P.T.astype(np.float64) @ P.astype(np.float64))
        A[:K, K] += P.T.astype(np.float64) @ T.astype(np.float64)
        A[K, :K] = A[:K, K]
        A[K, K] += float(T.size)
        rhs[:K] += P.T.astype(np.float64) @ T.astype(np.float64)
        rhs[K] += float(T.sum())

    # ridge on the weight block (bias unpenalised)
    lam = 1e-3 * float(np.trace(A[:K, :K])) / max(K, 1)
    Areg = A.copy()
    Areg[:K, :K] += lam * np.eye(K)
    try:
        sol = np.linalg.solve(Areg, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(Areg, rhs, rcond=None)[0]

    Wo = sol[:K].reshape(1, C, 3, 3).astype(np.float32)
    bo = np.array([sol[K]], dtype=np.float32)

    # ---------------- 3. evaluate candidate weight sets on training data ----------------
    def build(out_w, out_b, use_new_early=True):
        Wc = dict(Wnew)
        if use_new_early:
            Wc['out.weight'] = out_w
            Wc['out.bias'] = out_b
        else:
            Wc = {k: Wd[k].copy() for k in Wd}
            Wc['out.weight'] = out_w
            Wc['out.bias'] = out_b
        return Wc

    def eval_weights(Wc, nmax=600):
        xx = x[:nmax]; tt = t[:nmax]; yv = y[:nmax]
        p = forward(xx, tt, Wc)
        return mse(p, yv)

    # candidate A: original early layers + fitted readout (features are valid for the parent)
    candA = build(Wo, bo, use_new_early=False)
    mseA = eval_weights(candA)

    # candidate B: re-linearized early layers + fitted readout
    candB = build(Wo, bo, use_new_early=True)
    mseB = eval_weights(candB)

    # candidate C: re-linearized early layers, analytic readout = optimal per-timestep
    # scalar gain applied to x_t (implemented as conv1 x-filter + out readout)
    # -> use the fitted readout but ALSO add a direct x_t path is impossible (out sees only
    #    conv3 features), so instead fit the readout on the re-linearized features directly.
    # Recompute features with the new early layers, then refit the readout.
    def make_features(Wc, nmax=2000):
        Nn = min(nmax, N)
        tb = (t[:Nn].astype(np.float32) / 31.0).reshape(Nn, 1, 1, 1)
        inp = np.concatenate([x[:Nn], np.broadcast_to(tb, (Nn, 1, 16, 16))], axis=1)
        h1 = np.maximum(conv2d(inp, Wc['conv1.weight'], Wc['conv1.bias']), 0.0)
        h2 = np.maximum(conv2d(h1, Wc['conv2.weight'], Wc['conv2.bias']), 0.0)
        h3 = np.maximum(conv2d(h2, Wc['conv3.weight'], Wc['conv3.bias']), 0.0)
        return h3

    def fit_readout(F, Y, ridge=1e-4):
        n, Cc, Hh, Ww = F.shape
        Kk = Cc * 9
        AA = np.zeros((Kk + 1, Kk + 1), dtype=np.float64)
        rr = np.zeros((Kk + 1,), dtype=np.float64)
        ch = 100
        for s0 in range(0, n, ch):
            s1_ = min(s0 + ch, n)
            Fc = F[s0:s1_]
            nn = Fc.shape[0]
            P = np.zeros((nn, Cc, 3, 3, Hh, Ww), dtype=np.float32)
            for i in range(3):
                for j in range(3):
                    P[:, :, i, j, :, :] = Fc[:, :, i:i + Hh, j:j + Ww]
            P = P.transpose(0, 4, 5, 1, 2, 3).reshape(nn * Hh * Ww, Kk)
            T = Y[s0:s1_].reshape(nn * Hh * Ww)
            AA[:Kk, :Kk] += (P.T.astype(np.float64) @ P.astype(np.float64))
            AA[:Kk, Kk] += P.T.astype(np.float64) @ T.astype(np.float64)
            AA[Kk, :Kk] = AA[:Kk, Kk]
            AA[Kk, Kk] += float(T.size)
            rr[:Kk] += P.T.astype(np.float64) @ T.astype(np.float64)
            rr[Kk] += float(T.sum())
        lm = ridge * float(np.trace(AA[:Kk, :Kk])) / max(Kk, 1)
        AA[:Kk, :Kk] += lm * np.eye(Kk)
        sol = np.linalg.solve(AA, rr)
        return sol[:Kk].reshape(1, Cc, 3, 3).astype(np.float32), np.array([sol[Kk]], dtype=np.float32)

    Fnew = make_features(candB)
    Wo2, bo2 = fit_readout(Fnew, y)
    candC = build(Wo2, bo2, use_new_early=True)
    mseC = eval_weights(candC)

    # ---------------- 4. pick the best candidate on the training data ----------------
    cands = [(mseA, candA), (mseB, candB), (mseC, candC)]
    cands.sort(key=lambda z: z[0])
    best_mse, bestW = cands[0]

    # safety: never return something worse than the parent on the training set
    if best_mse > base_mse:
        bestW = {k: Wd[k].copy() for k in Wd}
        best_mse = base_mse

    # clip & sanitize
    out = {}
    for k, v in bestW.items():
        v = np.asarray(v, dtype=np.float32)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        v = np.clip(v, -1000.0, 1000.0).astype(np.float32)
        out[k] = v

    # diagnostics
    per_t = {}
    try:
        pfin = forward(x, t, out)
        for tt in range(32):
            m = t == tt
            if m.any():
                per_t[str(tt)] = round(float(np.mean((pfin[m] - y[m]) ** 2)), 5)
    except Exception:
        pass

    diag = {
        'base_mse': round(base_mse, 5),
        'mse_candA_origEarly_fittedReadout': round(float(mseA), 5),
        'mse_candB_linearEarly_fittedReadout': round(float(mseB), 5),
        'mse_candC_linearEarly_refitReadout': round(float(mseC), 5),
        'selected_mse_train': round(float(best_mse), 5),
        'per_t_mse': per_t,
    }
    return {'weights': out, 'diagnostics': diag}
