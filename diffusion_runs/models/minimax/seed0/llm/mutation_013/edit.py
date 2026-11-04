import numpy as np

# ---------------------------------------------------------------- helpers
def _conv(x, W, b, pad=1):
    """Cross-correlation conv, PyTorch layout W[out,in,kh,kw]."""
    N, C, H, Wd = x.shape
    O, Ci, KH, KW = W.shape
    if pad:
        xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    else:
        xp = x
    out = np.zeros((N, O, H, Wd), dtype=np.float64)
    for kh in range(KH):
        for kw in range(KW):
            out += np.einsum('nihw,oi->nohw', xp[:, :, kh:kh + H, kw:kw + Wd], W[:, :, kh, kw])
    out += b.reshape(1, O, 1, 1)
    return out

def _relu(x):
    return np.maximum(x, 0.0)

def _forward(x, t, w):
    """Exact numpy re-implementation of the fixed architecture."""
    N = x.shape[0]
    s = (t.astype(np.float64) / 31.0).reshape(N, 1, 1, 1)
    inp = np.concatenate([x.astype(np.float64), np.broadcast_to(s, (N, 1, x.shape[2], x.shape[3]))], axis=1)
    h1 = _relu(_conv(inp, w['conv1.weight'], w['conv1.bias']))
    h2 = _relu(_conv(h1, w['conv2.weight'], w['conv2.bias']))
    h3 = _relu(_conv(h2, w['conv3.weight'], w['conv3.bias']))
    o = _conv(h3, w['out.weight'], w['out.bias'])
    return o

# ---------------------------------------------------------------- edit
def edit(w, f):
    w = {k: np.array(v, dtype=np.float32, copy=True) for k, v in w.items()}
    diag = {}

    x = np.asarray(f['x'], dtype=np.float64)
    y = np.asarray(f['y'], dtype=np.float64)
    t = np.asarray(f['t']).astype(np.int64)
    clean = np.asarray(f['clean'], dtype=np.float64)
    abar = np.asarray(f['alphabar'], dtype=np.float64)
    N = x.shape[0]
    ts = t.reshape(-1, 1, 1, 1)

    # -------------------------------------------------- 1. verify schedule
    a = abar[t].reshape(-1, 1, 1, 1)
    sa = np.sqrt(a)
    sb = np.sqrt(np.maximum(1.0 - a, 1e-12))
    recon = sa * clean + sb * y
    resid = float(np.sqrt(np.mean((recon - x) ** 2)))
    diag['schedule_residual_rms'] = resid
    diag['abar0'] = float(abar[0]); diag['abar31'] = float(abar[-1])

    # -------------------------------------------------- 2. local linear fit
    # target: eps_hat(p) = c(t) + sum_o K(t,o) * x(p+o),  o in 5x5 window
    # fitted per timestep by ridge regression (closed form).
    R = 2                                   # half width of the 5x5 window
    P = 16
    offs = [(dy, dx) for dy in range(-R, R + 1) for dx in range(-R, R + 1)]
    M = len(offs)

    def window(a_):
        """a_: (N,P,P) -> (N, M, P, P) shifted copies (zero padded)."""
        ap = np.pad(a_, ((0, 0), (R, R), (R, R)))
        out = np.empty((a_.shape[0], M, P, P), dtype=np.float64)
        for i, (dy, dx) in enumerate(offs):
            out[:, i] = ap[:, R + dy:R + dy + P, R + dx:R + dx + P]
        return out

    Xw = window(x)                                   # (N,M,P,P)
    Xf = Xw.reshape(N * M * P * P, M)
    Yf = y.reshape(N * P * P, 1)

    lam = 1e-3 * Xf.shape[0] / 1000.0
    G = Xf.T @ Xf + lam * np.eye(M)
    Kt = np.zeros((32, M))
    ct = np.zeros(32)
    for k in range(32):
        idx = np.where(t == k)[0]
        if idx.size < 20:
            continue
        Xk = Xf.reshape(N, M, P * P)[idx].reshape(-1, M)
        Yk = Yf.reshape(N, P * P)[idx].reshape(-1, 1)
        Gk = Xk.T @ Xk + lam * np.eye(M)
        bk = Xk.T @ Yk
        sol = np.linalg.solve(Gk, bk)
        Kt[k] = sol[:, 0]
        ct[k] = float(np.mean(Yk - Xk @ sol))

    # training MSE of the local-linear model (the achievable floor here)
    pred_lin = np.empty_like(y)
    for k in range(32):
        idx = np.where(t == k)[0]
        if idx.size == 0:
            continue
        Xk = Xw[idx]
        pred_lin[idx] = np.einsum('nmpp,m->npp', Xk, Kt[k]) + ct[k]
    mse_lin = float(np.mean((pred_lin - y) ** 2))
    diag['local_linear_mse'] = mse_lin
    diag['baseline_mse'] = float(np.mean((np.asarray(f['prediction'], dtype=np.float64) - y) ** 2))

    # -------------------------------------------------- 3. build the network
    # Design: implement  out(p) = sum_o K(o) * x(p+o) + c   with a
    # per-timestep gain g(t) applied to the *whole* filter, realised by
    # conv1 producing  a = u + (g-1)*s  and  -a, so that
    # ReLU(a) - ReLU(-a) = a  and the out layer can subtract the pair.
    # Because g(t) enters through the s-channel, the effective filter is
    # K(t) = g(t) * Kbase  -> we fit Kbase = K(t)/g(t) and put the gain
    # into conv1.  We realise the gain through a *hinge* in s so that the
    # exact sign trick still works:
    #   a = u + (g(t)-1)*s   (u = x channel, s = t/31 broadcast)
    #   pair (ReLU(a), ReLU(-a)) -> difference = a  (exact)
    #   out(p) = sum_o W[o] * (a(p+o) - (g-1)s )  -> recovers sum_o K(o) u
    # The constant (g-1)*s part is spatially uniform and is removed by
    # subtracting the corresponding constant channel, giving exact
    # g(t) * (Kbase * u)(p) + c(t).
    #
    # Concretely:
    #  conv1 channel 0 : a  = u + (g(t)-1)*s        (3x3 identity on u)
    #  conv1 channel 1 : b  = -a
    #  conv2 channel 0 : copy of conv1 ch0 (identity, ReLU-safe)
    #  conv2 channel 1 : copy of conv1 ch1
    #  conv3 channel 0 : conv2ch0 - conv2ch1  = a   (linear in x and s)
    #  conv3 channel 1 : conv2ch1 - conv2ch0  = -a
    #  out             : sum_o W[o]*(conv3ch0(p+o) - conv3ch1(p+o)) + bias
    #                    -> sum_o W[o] * a(p+o)  (linear in x, plus s term)
    # The s-term of a is (g(t)-1)*s, constant in space; the out layer sums
    # W[o] over its 3x3 taps, giving a *spatially uniform* s contribution.
    # We therefore put the whole per-timestep variation of the filter into
    # the *spatial* coefficients by using the t-channel multiplicatively
    # through the hinge: instead of one global gain we build, for every
    # spatial offset o, its own pair of channels whose difference equals
    # K(t,o) * u(p+o) exactly, by using the identity
    #      K(t,o)*u = (1/2)[ (u + K u) - (u - K u) ] ... (needs products)
    # -> instead we exploit that the *whole* filter can be time-varying if
    #    we build the network from the *time-dependent* hinge trick:
    #    a_o = u + (K_o(t)-1)*s ;  ReLU(a_o)-ReLU(-a_o) = a_o
    #    out = sum_o W_o * a_o  with W_o chosen so that the s-part cancels:
    #    W_o * (K_o(t)-1)*s summed over o must be 0 for all t  -> not
    #    generally possible, so we instead subtract the s-part explicitly
    #    with a dedicated constant channel:
    #    conv3 channel 2 : constant 1 (bias only)  -> out can subtract
    #    the s-part exactly.
    # This gives an EXACT implementation of  out(p) = sum_o K(t,o) x(p+o)+c(t)
    # with per-timestep coefficients, as long as the s-part of each a_o is
    # linear in s (it is: K_o(t) enters linearly through s only if K_o(t)
    # is affine in s).  We therefore make K_o(t) affine in s by fitting
    # K(t,o) = k0_o + k1_o * s  (least squares over t), which is an
    # excellent approximation for the smooth schedule dependence.

    # --- fit affine-in-s coefficients:  K(t,o) = k0_o + k1_o * s(t)
    s_all = (t.astype(np.float64) / 31.0)
    # design matrix over timesteps for each offset
    A_ts = np.stack([np.ones(32), s_all], axis=1)          # (32,2)
    # weighted least squares over all pixels of all timesteps
    # K(t,o) = k0_o + k1_o*s_t ; residual r = y - c(t) - sum_o K(t,o) x(p+o)
    # Solve jointly for [k0_o, k1_o, c(t)] by alternating closed-form
    # (block coordinate descent on a convex problem -> deterministic).
    k0 = np.zeros(M); k1 = np.zeros(M); cvec = np.zeros(32)
    # init from per-t fit
    for k in range(32):
        if np.any(t == k):
            pass
    # initialise k0,k1 from the per-timestep fits via least squares in s
    for oi in range(M):
        ks = []; ss = []
        for k in range(32):
            if np.any(t == k):
                ks.append(Kt[k, oi]); ss.append(s_all[k])
        if len(ks) >= 2:
            A = np.stack([np.ones(len(ks)), np.array(ss)], axis=1)
            sol, *_ = np.linalg.lstsq(A, np.array(ks), rcond=None)
            k0[oi], k1[oi] = sol
        elif len(ks) == 1:
            k0[oi] = ks[0]; k1[oi] = 0.0
    cvec = ct.copy()

    # --- refine jointly (block coordinate descent, convex, closed-form)
    for it in range(6):
        # (a) update c(t) given k0,k1
        for k in range(32):
            idx = np.where(t == k)[0]
            if idx.size == 0:
                continue
            Xk = Xw[idx]
            Kk = k0[None, :] + k1[None, :] * s_all[k]
            base = np.einsum('nmpp,m->npp', Xk, Kk)
            cvec[k] = float(np.mean(y[idx] - base))
        # (b) update k0,k1 given c(t)
        # residual for pixel n: y - c(t_n) = sum_o (k0_o + k1_o s_n) x_{n,o}
        # normal equations per offset-pair with shared design
        # build big design: for each (n,pixel) row, columns = x_{n,o} and s_n*x_{n,o}
        # too big -> solve per offset using aggregated moments
        # moments: for each n: sum_p x_{n,o} x_{n,o'} etc.
        # use the 5x5 structure: compute per-example outer products of the
        # 25 shifted images (25x25) and aggregate.
        # (b1) compute per-example Gram G_n (M x M) and rhs b_n (M x 2)
        Xf2 = Xw.reshape(N, M, P * P)
        # G_n = sum_p x x^T ; b_n = sum_p x * [y-c, s*(y-c)]
        # memory: N x M x M = 2000*625 = 1.25M floats ok
        Gsum = np.einsum('nmp,nkp->nmk', Xf2, Xf2)            # (N,M,M)
        r1 = (y.reshape(N, P * P) - cvec[t][:, None])          # (N,P*P)
        r2 = r1 * s_all[:, None]
        b1 = np.einsum('nmp,np->nm', Xf2, r1)
        b2 = np.einsum('nmp,np->nm', Xf2, r2)
        # accumulate weighted by 1 and s
        # Solve for [k0;k1] minimizing sum_n || X_n (k0 + s_n k1) - r1_n ||^2
        # normal equations: (G11) k0 + (G12) k1 = b1 ; (G12)^T k0 + (G22) k1 = b2
        G11 = Gsum.sum(axis=0)
        G12 = (Gsum * s_all[:, None, None]).sum(axis=0)
        G22 = (Gsum * (s_all ** 2)[:, None, None]).sum(axis=0)
        B1 = b1.sum(axis=0)
        B2 = b2.sum(axis=0)
        # solve 2M x 2M system
        Z = np.zeros((2 * M, 2 * M))
        Z[:M, :M] = G11
        Z[:M, M:] = G12
        Z[M:, :M] = G12.T
        Z[M:, M:] = G22
        rhs = np.concatenate([B1, B2])
        Z += 1e-6 * np.trace(Z) / (2 * M) * np.eye(2 * M)
        sol = np.linalg.solve(Z, rhs)
        k0 = sol[:M]; k1 = sol[M:]

    # final c(t) update
    for k in range(32):
        idx = np.where(t == k)[0]
        if idx.size == 0:
            continue
        Xk = Xw[idx]
        Kk = k0[None, :] + k1[None, :] * s_all[k]
        cvec[k] = float(np.mean(y[idx] - np.einsum('nmpp,m->npp', Xk, Kk)))

    # MSE of the affine-in-s local linear model
    pred_aff = np.empty_like(y)
    for k in range(32):
        idx = np.where(t == k)[0]
        if idx.size == 0:
            continue
        Kk = k0[None, :] + k1[None, :] * s_all[k]
        pred_aff[idx] = np.einsum('nmpp,m->npp', Xw[idx], Kk) + cvec[k]
    mse_aff = float(np.mean((pred_aff - y) ** 2))
    diag['affine_s_local_linear_mse'] = mse_aff

    # -------------------------------------------------- 4. realise in the net
    # Effective filter of the built network: out(p) = sum_o K(t,o) x(p+o) + c(t)
    # with K(t,o) = k0_o + k1_o * s.
    # Construction (exact, using ReLU(a)-ReLU(-a)=a):
    #   conv1 ch0: a = u + (k1_o)*s ... but k1_o differs per offset -> we
    #   need one channel per offset with its own s-gain, then the out layer
    #   (3x3 conv over conv3 channels) must place them at the right spatial
    #   offsets.  conv3 channels are 3x3-conv of conv2 channels, so a
    #   conv3 channel at position p sees conv2 at p-1..p+1.  We use centre
    #   taps only, so conv3 ch j at position p = conv2 ch j at position p.
    #   Then the out layer (3x3 over conv3) at position p sums conv3 ch j at
    #   p-1..p+1 -> so conv3 channel j must carry the shifted signal.
    #   Therefore: conv3 channel j (for offset o) = conv2 channel j shifted
    #   by o.  Shifting is done with the conv3 3x3 kernel (a delta at the
    #   right tap).  conv2 channel j = copy of conv1 channel j (identity).
    #   conv1 channel j = u shifted by o, plus s-gain k1_o * s.
    #   With conv1 3x3 we can shift u by at most 1 -> total reachable
    #   offsets are 3x3 (R=1), not 5x5.  So we restrict the fitted filter
    #   to a 3x3 window (R=1) and refit.
    R = 1
    offs = [(dy, dx) for dy in range(-R, R + 1) for dx in range(-R, R + 1)]
    M = len(offs)

    def window3(a_):
        ap = np.pad(a_, ((0, 0), (R, R), (R, R)))
        out = np.empty((a_.shape[0], M, P, P), dtype=np.float64)
        for i, (dy, dx) in enumerate(offs):
            out[:, i] = ap[:, R + dy:R + dy + P, R + dx:R + dx + P]
        return out

    Xw = window3(x)
    Xf = Xw.reshape(N * M * P * P, M)
    Xf2 = Xw.reshape(N, M, P * P)
    lam = 1e-4 * max(1.0, Xf.shape[0] / 1000.0)

    # per-timestep ridge fit (3x3 window) -> reference floor
    Kt = np.zeros((32, M)); ct = np.zeros(32)
    for k in range(32):
        idx = np.where(t == k)[0]
        if idx.size < 20:
            continue
        Xk = Xf2[idx].reshape(-1, M); Yk = y[idx].reshape(-1, 1)
        sol = np.linalg.solve(Xk.T @ Xk + lam * np.eye(M), Xk.T @ Yk)
        Kt[k] = sol[:, 0]; ct[k] = float(np.mean(Yk - Xk @ sol))
    pred3 = np.empty_like(y)
    for k in range(32):
        idx = np.where(t == k)[0]
        if idx.size == 0:
            continue
        pred3[idx] = np.einsum('nmpp,m->npp', Xw[idx], Kt[k]) + ct[k]
    diag['local_linear_mse_3x3'] = float(np.mean((pred3 - y) ** 2))

    # affine-in-s joint fit on the 3x3 window
    k0 = np.zeros(M); k1 = np.zeros(M); cvec = np.zeros(32)
    for oi in range(M):
        ks = []; ss = []
        for k in range(32):
            if np.any(t == k):
                ks.append(Kt[k, oi]); ss.append(s_all[k])
        if len(ks) >= 2:
            A = np.stack([np.ones(len(ks)), np.array(ss)], axis=1)
            sol, *_ = np.linalg.lstsq(A, np.array(ks), rcond=None)
            k0[oi], k1[oi] = sol
        elif len(ks) == 1:
            k0[oi] = ks[0]
    for it in range(8):
        for k in range(32):
            idx = np.where(t == k)[0]
            if idx.size == 0:
                continue
            Kk = k0[None, :] + k1[None, :] * s_all[k]
            cvec[k] = float(np.mean(y[idx] - np.einsum('nmpp,m->npp', Xw[idx], Kk)))
        Gsum = np.einsum('nmp,nkp->nmk', Xf2, Xf2)
        r1 = y.reshape(N, P * P) - cvec[t][:, None]
        b1 = np.einsum('nmp,np->nm', Xf2, r1)
        b2 = np.einsum('nmp,np->nm', Xf2, r1 * s_all[:, None])
        G11 = Gsum.sum(axis=0)
        G12 = (Gsum * s_all[:, None, None]).sum(axis=0)
        G22 = (Gsum * (s_all ** 2)[:, None, None]).sum(axis=0)
        Z = np.zeros((2 * M, 2 * M))
        Z[:M, :M] = G11; Z[:M, M:] = G12; Z[M:, :M] = G12.T; Z[M:, M:] = G22
        rhs = np.concatenate([b1.sum(axis=0), b2.sum(axis=0)])
        Z += 1e-8 * np.trace(Z) / (2 * M) * np.eye(2 * M)
        sol = np.linalg.solve(Z, rhs)
        k0 = sol[:M]; k1 = sol[M:]
    for k in range(32):
        idx = np.where(t == k)[0]
        if idx.size == 0:
            continue
        Kk = k0[None, :] + k1[None, :] * s_all[k]
        cvec[k] = float(np.mean(y[idx] - np.einsum('nmpp,m->npp', Xw[idx], Kk)))
    pred_aff3 = np.empty_like(y)
    for k in range(32):
        idx = np.where(t == k)[0]
        if idx.size == 0:
            continue
        Kk = k0[None, :] + k1[None, :] * s_all[k]
        pred_aff3[idx] = np.einsum('nmpp,m->npp', Xw[idx], Kk) + cvec[k]
    diag['affine_s_local_linear_mse_3x3'] = float(np.mean((pred_aff3 - y) ** 2))

    # -------------------------------------------------- 5. write the weights
    # Channel plan (32 conv1 channels, 64 conv2, 64 conv3, 1 out):
    #   conv1 ch 2j   : a_j  = shift_j(u) + k1_j * s      (3x3 delta + s-gain)
    #   conv1 ch 2j+1 : -a_j                              (negated)
    #   conv2 ch 2j   : copy of conv1 ch 2j   (identity, >=0 after ReLU)
    #   conv2 ch 2j+1 : copy of conv1 ch 2j+1
    #   conv3 ch 2j   : conv2ch(2j) - conv2ch(2j+1)  shifted by delta_j
    #   conv3 ch 2j+1 : conv2ch(2j+1) - conv2ch(2j)  shifted by delta_j
    #   out           : 3x3 conv over conv3 with per-offset weights k0_j
    #                   plus a constant channel to cancel the s-part.
    # The s-part of a_j is k1_j * s (spatially uniform).  In conv3 we get
    # (a_j - (-a_j)) = 2*a_j  -> contains 2*k1_j*s.  The out layer sums
    # k0_j * (that) -> s-part = 2*s*sum_j k0_j*k1_j (spatially uniform).
    # We cancel it with a dedicated conv3 constant channel (value 1) whose
    # out weight is set to -2*sum_j k0_j*k1_j / (number of taps) ... we
    # instead simply subtract the uniform s-part by adding a conv3 channel
    # that equals -2*sum_j k0_j k1_j * s (a constant channel with s-gain),
    # and give it out-weight 1 at the centre tap.
    n1 = 32; n2 = 64; n3 = 64
    W1 = np.zeros((n1, 2, 3, 3)); b1v = np.zeros(n1)
    W2 = np.zeros((n2, n1, 3, 3)); b2v = np.zeros(n2)
    W3 = np.zeros((n3, n2, 3, 3)); b3v = np.zeros(n3)
    Wo = np.zeros((1, n3, 3, 3)); bo = np.zeros(1)

    # map: conv1 channel pair j -> spatial offset (dy,dx) of the 3x3 window
    # out layer 3x3 taps: out(p) = sum_{tap} Wo[ch, tap] * conv3[ch](p+tap)
    # we want conv3[2j](p) = a_j(p) and out to use centre tap with k0_j.
    # a_j(p) = shift_j(u)(p) + k1_j*s ; shift by delta_j in conv1's 3x3.
    # conv3[2j] = conv2[2j] - conv2[2j+1] = a_j - (-a_j) = 2*a_j
    # so out weight for conv3[2j] must be k0_j/2 at the centre tap.
    # conv3[2j+1] = -2*a_j  (unused in out, kept for structure)
    for j, (dy, dx) in enumerate(offs):
        cA = 2 * j; cB = 2 * j + 1
        # conv1 ch cA : u shifted by (dy,dx) + k1_j * s
        W1[cA, 0, 1 - dy, 1 - dx] = 1.0
        W1[cA, 1, 1, 1] = k1[j]
        # conv1 ch cB : -(that)
        W1[cB, 0, 1 - dy, 1 - dx] = -1.0
        W1[cB, 1, 1, 1] = -k1[j]
        # conv2 copies (identity, centre tap)
        W2[cA, cA, 1, 1] = 1.0
        W2[cB, cB, 1, 1] = 1.0
        # conv3 : difference
        W3[cA, cA, 1, 1] = 1.0
        W3[cA, cB, 1, 1] = -1.0
        W3[cB, cB, 1, 1] = 1.0
        W3[cB, cA, 1, 1] = -1.0
        # out : centre tap on conv3[cA] with weight k0_j/2
        Wo[0, cA, 1, 1] += 0.5 * k0[j]

    # constant channel to cancel the uniform s-part  2*s*sum_j k0_j*k1_j
    # conv3 channel 63 : constant 1  (bias 1, zero weights)
    W3[63, :, :, :] = 0.0
    b3v[63] = 1.0
    s_coeff = 2.0 * float(np.sum(k0 * k1))
    # conv3 channel 62 : constant -s_coeff*s  (s-gain via conv1? conv3 has no
    # s input) -> instead use conv1 ch 62/63 pair?  Simpler: give conv3 ch 63
    # a bias of 1 and let the out layer add a *constant*; the s-part must be
    # produced by a channel that depends on s.  Use conv1 ch 62 = s (weight 1
    # on s channel, bias 0) and conv3 ch 62 = copy of conv2 ch 62 (identity).
    W1[30, 1, 1, 1] = 1.0
    W2[60, 30, 1, 1] = 1.0
    W3[62, 60, 1, 1] = 1.0
    Wo[0, 62, 1, 1] += -s_coeff
    # (the constant-1 channel 63 is not needed; keep it harmless)
    W3[63, :, :, :] = 0.0
    b3v[63] = 0.0

    # per-timestep constant c(t): must be spatially uniform and t-dependent.
    # c(t) is not affine in s in general -> approximate with affine fit.
    cs = []; ss = []
    for k in range(32):
        cs.append(cvec[k]); ss.append(s_all[k])
    A = np.stack([np.ones(32), np.array(ss)], axis=1)
    csol, *_ = np.linalg.lstsq(A, np.array(cs), rcond=None)
    c0, c1 = float(csol[0]), float(csol[1])
    # add c0 via out bias, c1*s via the s-channel path (conv3 ch 62 already
    # carries s; give it an extra weight for c1)
    Wo[0, 62, 1, 1] += c1
    bo[0] = c0

    w['conv1.weight'] = W1.astype(np.float32)
    w['conv1.bias'] = b1v.astype(np.float32)
    w['conv2.weight'] = W2.astype(np.float32)
    w['conv2.bias'] = b2v.astype(np.float32)
    w['conv3.weight'] = W3.astype(np.float32)
    w['conv3.bias'] = b3v.astype(np.float32)
    w['out.weight'] = Wo.astype(np.float32)
    w['out.bias'] = bo.astype(np.float32)

    # -------------------------------------------------- 6. verify
    for k in w:
        v = np.asarray(w[k], dtype=np.float32)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        v = np.clip(v, -1000.0, 1000.0)
        w[k] = v.astype(np.float32)

    # forward check on a subset (exact architecture emulation)
    sub = np.arange(0, N, max(1, N // 200))
    pred = _forward(x[sub], t[sub], w)
    mse_check = float(np.mean((pred - y[sub][:, None, :, :]) ** 2))
    diag['mse_check_200'] = mse_check
    diag['mse_check_2000'] = float(np.mean((_forward(x, t, w)[:, 0] - y) ** 2))
    diag['pred_std'] = float(np.std(pred))
    diag['k0_absmax'] = float(np.max(np.abs(k0)))
    diag['k1_absmax'] = float(np.max(np.abs(k1)))
    diag['c_range'] = [float(np.min(cvec)), float(np.max(cvec))]

    return {"weights": w, "diagnostics": diag}
