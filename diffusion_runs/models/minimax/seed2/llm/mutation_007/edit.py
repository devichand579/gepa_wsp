import numpy as np

def edit(w, f):
    # ---------------- data ----------------
    x = np.asarray(f["x"], np.float64)            # [N,1,16,16] noisy x_t
    y = np.asarray(f["y"], np.float64)            # [N,1,16,16] target epsilon
    t = np.asarray(f["t"], np.int64).reshape(-1)  # [N]
    clean = np.asarray(f["clean"], np.float64)    # [N,1,16,16]
    abar = np.asarray(f["alphabar"], np.float64).reshape(-1)  # [32]

    N = x.shape[0]
    xf = x.reshape(N, -1)
    yf = y.reshape(N, -1)

    # ---------------- per-timestep closed-form linear regression ----------
    # y ~ A(t)*x + B(t) + C(t)*t   (per pixel, shared across pixels)
    # ridge-regularised least squares on 3 features per example
    lam = 1e-6
    A = np.zeros(32); B = np.zeros(32); C = np.zeros(32)
    for tt in range(32):
        m = (t == tt)
        n = int(m.sum())
        if n < 3:
            continue
        Xt = xf[m]; Yt = yf[m]
        tf = (t[m].astype(np.float64) / 31.0)
        F = np.stack([Xt, np.ones(n), np.full(n, tf[0])], axis=2)  # [n,P,3]
        # normal equations with ridge on (A,B) only
        G = np.einsum('npi,npj->ij', F, F)
        rhs = np.einsum('npi,np->i', F, Yt)
        R = np.diag([lam, lam, 0.0])
        try:
            coef = np.linalg.solve(G + R, rhs)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(G + R, rhs, rcond=None)[0]
        A[tt], B[tt], C[tt] = coef[0], coef[1], coef[2]

    # smooth A(t) across t (it is a smooth function of t) to reduce noise
    # simple 1-2-1 smoothing on the interior, endpoints kept
    As = A.copy()
    As[1:-1] = 0.25 * A[:-2] + 0.5 * A[1:-1] + 0.25 * A[2:]
    A = As

    # ---------------- express target as A(t)*x + B(t) + C(t)*t -----------
    # network output must equal  A(t)*x + B(t) + C(t)*(t/31)
    # we realise:  out = g1*(x - k*(t/31)) + g0*(t/31) + b0
    # choose k so that the required constant part is representable:
    #   A(t)*x + B(t) + C(t)*tau  with tau=t/31
    #   => g1 = A(t), and we need g1*(-k*tau) + g0*tau + b0 = B(t) + C(t)*tau
    # choose k = 1 (so the conv2/conv3 pair carries x - tau):
    #   then need g0*tau + b0 - A(t)*tau = B(t) + C(t)*tau
    #   => g0 = C(t) + A(t) - B(t)/tau  (tau-dependent -> absorb via per-t
    #      affine fit of B(t) in tau).  Fit B(t) ~ b0 + b1*tau by least squares.
    tau_all = np.arange(32) / 31.0
    Mb = np.stack([np.ones(32), tau_all], axis=1)
    try:
        bb = np.linalg.lstsq(Mb, B, rcond=None)[0]
    except Exception:
        bb = np.array([B.mean(), 0.0])
    b0, b1 = float(bb[0]), float(bb[1])
    # residual of the B(t) fit is folded into the constant channel via C(t)
    # (C(t) is free per t, so we can absorb: need g0(t)*tau + b0 - A(t)*tau
    #  = B(t) + C(t)*tau  ->  g0(t) = C(t) + A(t) + (B(t)-b0-b1*tau)/tau )
    # to keep g0 a function of tau only, fit g0(t) by least squares on tau.
    with np.errstate(divide='ignore', invalid='ignore'):
        g0_raw = C + A + np.where(tau_all > 0, (B - b0 - b1 * tau_all) / np.maximum(tau_all, 1e-12), 0.0)
    g0_raw[0] = C[0] + A[0] + (B[0] - b0) / (1.0 / 31.0)
    Mg = np.stack([np.ones(32), tau_all], axis=1)
    try:
        gg = np.linalg.lstsq(Mg, g0_raw, rcond=None)[0]
    except Exception:
        gg = np.array([g0_raw.mean(), 0.0])
    g0_0, g0_1 = float(gg[0]), float(gg[1])

    # final mapping (per t):  out = A(t)*x + g0(t)*tau + b0
    # where g0(t) = g0_0 + g0_1*tau  (affine in tau)

    # ---------------- build the fixed conv stack -------------------------
    # channel plan
    # conv1 (2->32):
    #   ch0: x            (weight on x-channel: centre tap 1)
    #   ch1: -x
    #   ch2: tau = t/31   (weight on t-channel: one tap 1)
    #   ch3: -tau
    #   ch4: 1            (bias 1)
    #   ch5: -1
    # conv2 (32->64):
    #   ch0: x - tau      (from conv1 ch0 with +1, ch2 with -1)
    #   ch1: -(x - tau)   (from ch1 with +1, ch3 with +1)
    #   ch2: tau          (from ch2 with +1)
    #   ch3: -tau         (from ch3 with +1)
    #   ch4: 1            (from ch4 with +1)
    #   ch5: -1           (from ch5 with +1)
    # conv3 (64->64):
    #   ch0: x - tau      (from conv2 ch0 +1, ch2 -1)
    #   ch1: -(x - tau)   (from conv2 ch1 +1, ch3 +1)
    #   ch2: tau          (from conv2 ch2 +1)
    #   ch3: -tau         (from conv2 ch3 +1)
    #   ch4: 1            (from conv2 ch4 +1)
    #   ch5: -1           (from conv2 ch5 +1)
    # out (64->1):
    #   uses conv3 ch0 (x-tau) with weight A(t)?? -> A depends on t, not constant!
    #
    # The t-dependent gain A(t) on x is the hard part.  We realise it with a
    # per-timestep gain by making the *pair* channels carry (x - tau) and then
    # using the fact that the required output is
    #     A(t)*x + g0(t)*tau + b0
    #   = A(t)*(x - tau) + (g0(t) + A(t))*tau + b0
    # The gain A(t) is still t-dependent.  We approximate A(t) exactly by
    # expressing the output as a sum over t-blocks is impossible (fixed depth),
    # so instead we use a *linear-in-t* gain approximation:
    #     A(t) ~ a0 + a1*tau
    # and then out = a0*(x-tau) + a1*tau*(x-tau) + ... : needs a product.
    #
    # Instead we use the exact piecewise structure: A(t) is a smooth function
    # of t, and the *product* tau*(x-tau) is not available.  So we fall back to
    # an exact representation using the ReLU "hinge" trick on tau:
    #   A(t) = sum_j  c_j * relu(tau - d_j)   (piecewise-linear in tau)
    # and the product with x is still needed -> not possible.
    #
    # FINAL: use the sign-pair trick to build a *t-independent* gain on x and
    # put the t-dependence into the *value* of a channel that is multiplied by
    # x through a ReLU hinge on x itself:
    #   relu(x - c) - relu(x - c) ... still no product.
    #
    # => We therefore use the exact closed-form *per-timestep* affine predictor
    #    implemented via 32-step piecewise linear gain realised by a linear
    #    combination of shifted-and-clipped x channels.  Concretely:
    #    conv1 emits x and -x (channels 0,1) plus tau,-tau,1,-1.
    #    conv2 emits relu-ready channels  (x - tau) and -(x - tau) and tau,-tau,1,-1
    #    conv3 emits  (x - tau) and -(x - tau) and tau,-tau,1,-1
    #    out combines:  out = w_x * (x - tau) + w_t * tau + w_c * 1
    #    with w_x CONSTANT.  The t-dependent gain is therefore approximated by
    #    the best single constant gain (least squares over all t) plus a
    #    t-dependent offset.  This is the *only* t-independent-gain solution.
    #
    # We instead obtain the t-dependent gain by exploiting that the *pair*
    # trick lets us form ANY piecewise-linear function of (x, tau) with 3
    # ReLU layers when the function is a sum of terms  relu(a*x + b*tau + c)
    # and  -relu(a*x + b*tau + c).  A(t)*x is bilinear in (x,tau) and is NOT
    # in that class.  Therefore we approximate the bilinear gain by a
    # piecewise-linear-in-x function whose coefficient varies with tau:
    #   A(t)*x  ~  sum_j  relu(x - c_j) * ... (no product)  -> impossible.
    #
    # CONCLUSION: exact t-dependent gain requires a product; we obtain it
    # with the classic ReLU product identity using the extra depth of the
    # stack:
    #   x*tau = 0.5*[(x+tau)^2 - (x-tau)^2]  needs squares (depth 2 more).
    # Instead we use the *exact* identity valid on the discrete tau grid:
    #   tau*x = sum_{k} tau_k * (x - x_{k}) ... no.
    #
    # Practical resolution: the gain A(t) is applied by *re-parameterising the
    # target*: we fit a single global affine map  y ~ a*x + b*tau + c  and also
    # a per-timestep map; we then choose the global map that minimises the
    # *actual* training MSE (computed exactly), and additionally exploit the
    # fact that the network can represent a per-timestep map through the
    # *bias* path when the gain is close to a constant.

    # ---- compute the best t-independent-gain affine predictor -------------
    # features per example: [x, 1, tau]
    F = np.stack([xf, np.ones((N, 1)), (t.astype(np.float64) / 31.0)[:, None]], axis=2)
    G = np.einsum('npi,npj->ij', F, F)
    rhs = np.einsum('npi,np->i', F, yf)
    R = np.diag([lam, lam, lam])
    try:
        coef = np.linalg.solve(G + R, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(G + R, rhs, rcond=None)[0]
    a_g, b_g, c_g = float(coef[0]), float(coef[1]), float(coef[2])

    # ---- build weights ---------------------------------------------------
    W = {k: np.array(v, dtype=np.float32, copy=True) for k, v in w.items()}

    def zero_like(key):
        return np.zeros_like(W[key], dtype=np.float64)

    c1w = zero_like("conv1.weight"); c1b = zero_like("conv1.bias")
    c2w = zero_like("conv2.weight"); c2b = zero_like("conv2.bias")
    c3w = zero_like("conv3.weight"); c3b = zero_like("conv3.bias")
    ow = zero_like("out.weight"); ob = zero_like("out.bias")

    # conv1
    c1w[0, 0, 1, 1] = 1.0          # ch0 = x
    c1w[1, 0, 1, 1] = -1.0         # ch1 = -x
    c1w[2, 1, 0, 0] = 1.0          # ch2 = tau
    c1w[3, 1, 0, 0] = -1.0         # ch3 = -tau
    c1b[4] = 1.0                   # ch4 = 1
    c1b[5] = -1.0                  # ch5 = -1

    # conv2: build (x - tau), -(x-tau), tau, -tau, 1, -1
    c2w[0, 0, 1, 1] = 1.0
    c2w[0, 2, 0, 0] = -1.0
    c2w[1, 1, 1, 1] = 1.0
    c2w[1, 3, 0, 0] = 1.0
    c2w[2, 2, 0, 0] = 1.0
    c2w[3, 3, 0, 0] = 1.0
    c2w[4, 4, 0, 0] = 1.0
    c2w[5, 5, 0, 0] = 1.0

    # conv3: same structure, from conv2 channels
    c3w[0, 0, 1, 1] = 1.0
    c3w[0, 2, 0, 0] = -1.0
    c3w[1, 1, 1, 1] = 1.0
    c3w[1, 3, 0, 0] = 1.0
    c3w[2, 2, 0, 0] = 1.0
    c3w[3, 3, 0, 0] = 1.0
    c3w[4, 4, 0, 0] = 1.0
    c3w[5, 5, 0, 0] = 1.0

    # out:  out = a_g*(x - tau) + (a_g + c_g)*tau + b_g
    # conv3 ch0 = relu(x-tau) (>=0 when x>=tau), ch1 = relu(-(x-tau)).
    # Using the pair: relu(u) - relu(-u) = u  ->  out needs a signed
    # combination, but out has no ReLU afterwards, so we can use
    #   a_g * (relu(u) - relu(-u))  -> requires negative weight on ch1: fine.
    ow[0, 0, 1, 1] = a_g
    ow[0, 1, 1, 1] = -a_g
    ow[0, 2, 0, 0] = (a_g + c_g)
    ow[0, 3, 0, 0] = -(a_g + c_g)
    ow[0, 4, 0, 0] = b_g
    ow[0, 5, 0, 0] = -b_g
    ob[0] = 0.0

    W["conv1.weight"] = c1w.astype(np.float32)
    W["conv1.bias"] = c1b.astype(np.float32)
    W["conv2.weight"] = c2w.astype(np.float32)
    W["conv2.bias"] = c2b.astype(np.float32)
    W["conv3.weight"] = c3w.astype(np.float32)
    W["conv3.bias"] = c3b.astype(np.float32)
    W["out.weight"] = ow.astype(np.float32)
    W["out.bias"] = ob.astype(np.float32)

    # ---------------- diagnostics (exact forward on training set) ---------
    def conv3x3(inp, wt, bs):
        # inp [N,C,H,W], wt [O,C,3,3] cross-correlation, pad 1
        N, C, H, Wd = inp.shape
        O = wt.shape[0]
        ip = np.pad(inp, ((0, 0), (0, 0), (1, 1), (1, 1)))
        out = np.zeros((N, O, H, Wd))
        for i in range(3):
            for j in range(3):
                out += np.tensordot(ip[:, :, i:i + H, j:j + Wd],
                                    wt[:, :, i, j], axes=([1], [1]))
        return out + bs.reshape(1, -1, 1, 1)

    tau = (t.astype(np.float64) / 31.0).reshape(-1, 1, 1, 1)
    inp = np.concatenate([x, np.broadcast_to(tau, x.shape)], axis=1)
    h = np.maximum(conv3x3(inp, c1w, c1b), 0.0)
    h = np.maximum(conv3x3(h, c2w, c2b), 0.0)
    h = np.maximum(conv3x3(h, c3w, c3b), 0.0)
    pred = conv3x3(h, ow, ob)
    mse_new = float(np.mean((pred - y) ** 2))
    mse_old = float(np.mean((np.asarray(f["prediction"], np.float64) - y) ** 2))

    # per-timestep MSE of the new predictor
    per_t = {}
    for tt in range(32):
        m = (t == tt)
        if m.sum() > 0:
            per_t[str(tt)] = round(float(np.mean((pred[m] - y[m]) ** 2)), 4)

    # reference: best per-timestep affine predictor (not representable exactly)
    ref = float(np.mean([np.mean((y[t == tt] -
                    (A[tt] * xf[t == tt] + B[tt] + C[tt] * (tt / 31.0))) ** 2)
                    for tt in range(32) if (t == tt).sum() > 0]))

    for k in W:
        W[k] = np.clip(np.nan_to_num(W[k]), -1000.0, 1000.0).astype(np.float32)

    return {"weights": W,
            "diagnostics": {"mse_old": round(mse_old, 5),
                            "mse_new": round(mse_new, 5),
                            "mse_per_timestep_ref": round(ref, 5),
                            "gain_global": round(a_g, 4),
                            "intercept": round(b_g, 4),
                            "tau_coef": round(c_g, 4),
                            "per_timestep_mse": per_t,
                            "clean_mean": round(float(clean.mean()), 4),
                            "clean_var": round(float(clean.var()), 4)}}
