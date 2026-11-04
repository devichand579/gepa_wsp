import numpy as np

def _conv2d(x, W, b):
    """PyTorch-style cross-correlation, stride 1, pad 1. x:[N,C,H,W] -> [N,O,H,W]"""
    N, C, H, Wd = x.shape
    O = W.shape[0]
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    out = np.zeros((N, O, H, Wd), dtype=np.float64)
    for kr in range(3):
        for kc in range(3):
            patch = xp[:, :, kr:kr + H, kc:kc + Wd]           # [N,C,H,W]
            out += np.tensordot(patch, W[:, :, kr, kc], axes=([1], [1])).transpose(0, 3, 1, 2)
    return out + b.reshape(1, -1, 1, 1)

def _forward(w, x, t):
    """Exact re-implementation of the fixed architecture (float64)."""
    N = x.shape[0]
    tchan = (t.astype(np.float64) / 31.0).reshape(N, 1, 1, 1)
    inp = np.concatenate([x.astype(np.float64), np.broadcast_to(tchan, (N, 1, x.shape[2], x.shape[3]))], axis=1)
    h1 = np.maximum(_conv2d(inp, w["conv1.weight"], w["conv1.bias"]), 0.0)
    h2 = np.maximum(_conv2d(h1, w["conv2.weight"], w["conv2.bias"]), 0.0)
    h3 = np.maximum(_conv2d(h2, w["conv3.weight"], w["conv3.bias"]), 0.0)
    o = _conv2d(h3, w["out.weight"], w["out.bias"])
    return o, h1, h2, h3

def edit(w, f):
    w = {k: np.array(v, dtype=np.float64) for k, v in w.items()}

    x = np.asarray(f["x"], dtype=np.float64)            # [N,1,16,16]
    y = np.asarray(f["y"], dtype=np.float64)            # [N,1,16,16]
    t = np.asarray(f["t"]).astype(np.int64).ravel()      # [N]
    ab = np.asarray(f["alphabar"], dtype=np.float64)     # [32]
    N = x.shape[0]

    # ---------- sanity: reproduce supplied traces / prediction with current weights ----------
    diag = {}
    try:
        pred, h1, h2, h3 = _forward(w, x[:128], t[:128])
        diag["trace_conv1_maxerr"] = float(np.max(np.abs(h1 - np.asarray(f["layer.conv1.output"], dtype=np.float64))))
        diag["trace_conv2_maxerr"] = float(np.max(np.abs(h2 - np.asarray(f["layer.conv2.output"], dtype=np.float64))))
        diag["trace_conv3_maxerr"] = float(np.max(np.abs(h3 - np.asarray(f["layer.conv3.output"], dtype=np.float64))))
        diag["trace_out_maxerr"] = float(np.max(np.abs(pred - np.asarray(f["layer.out.output"], dtype=np.float64))))
        diag["pred_maxerr"] = float(np.max(np.abs(pred - np.asarray(f["prediction"][:128], dtype=np.float64))))
    except Exception as e:  # pragma: no cover
        diag["trace_check_error"] = str(e)

    # ---------- closed-form per-timestep scalar filter c_t :  y ~ c_t * x ----------
    xs = x[:, 0].reshape(N, -1)          # [N,256]
    ys = y[:, 0].reshape(N, -1)
    ts = np.unique(t)
    c_t = {}
    sxx_t, sxy_t, n_t = {}, {}, {}
    for tt in ts:
        m = (t == tt)
        X = xs[m]; Y = ys[m]
        sxx = float(np.sum(X * X)); sxy = float(np.sum(X * Y))
        a = float(ab[tt])
        den = a * sxx + (1.0 - a) * sxx + 1e-12
        c_t[tt] = sxy / den
        sxx_t[tt] = sxx; sxy_t[tt] = sxy; n_t[tt] = int(m.sum())

    # ---------- project c_t onto the achievable affine family c_t = alpha + (t/31)*beta ----------
    tt_all = np.array(sorted(ts), dtype=np.float64)
    s = tt_all / 31.0
    cvec = np.array([c_t[int(tt)] for tt in tt_all])
    wts = np.array([n_t[int(tt)] for tt in tt_all], dtype=np.float64)
    # weighted least squares for alpha, beta
    A = np.stack([np.ones_like(s), s], axis=1)
    Aw = A * wts[:, None]
    coef, *_ = np.linalg.lstsq(Aw.T @ A, Aw.T @ cvec, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])

    # ---------- per-timestep constant (bias) pattern: mean of (y - c_t x) ----------
    # network can realise  (t/31)*p + q  (constant spatial pattern, affine in t/31)
    resid_mean = {}
    for tt in ts:
        m = (t == tt)
        resid_mean[tt] = float(np.mean(ys[m] - c_t[int(tt)] * xs[m]))
    rvec = np.array([resid_mean[int(tt)] for tt in tt_all])
    coef2, *_ = np.linalg.lstsq(Aw.T @ A, Aw.T @ rvec, rcond=None)
    p0, q0 = float(coef2[0]), float(coef2[1])   # r_t = p0 + (t/31)*q0

    # ---------- design the network ----------
    # channel plan
    #  conv1 out channels: 0 -> carries x (identity), 1 -> carries t-plane pattern, 2.. -> padding/aux
    #  conv2: identity on channel 0 and 1 (centre tap), rest zero
    #  conv3: identity on channel 0 and 1, rest zero
    #  out : centre tap on channel 0 (x) and channel 1 (t-pattern)
    W1 = np.zeros((32, 2, 3, 3))
    W1[0, 0, 1, 1] = 1.0                      # x channel, centre tap
    W1[1, 1, 1, 1] = 1.0                      # t channel, centre tap -> interior 1, border 1/3 (edge) / 1/6 (corner)
    b1 = np.zeros(32)

    W2 = np.zeros((64, 32, 3, 3))
    for o in range(64):
        W2[o, 0, 1, 1] = 1.0                  # pass x-channel
        W2[o, 1, 1, 1] = 1.0                  # pass t-pattern channel
    b2 = np.zeros(64)

    W3 = np.zeros((64, 64, 3, 3))
    for o in range(64):
        W3[o, 0, 1, 1] = 1.0
        W3[o, 1, 1, 1] = 1.0
    b3 = np.zeros(64)

    Wo = np.zeros((1, 64, 3, 3))
    Wo[0, 0, 1, 1] = 1.0                      # x channel
    Wo[0, 1, 1, 1] = 1.0                      # t-pattern channel
    bo = np.zeros(1)

    # ---------- choose magnitudes so that every ReLU stays inactive (all pre-activations > 0) ----------
    xmax = float(np.max(np.abs(xs)))
    # conv1 pre-activation (channel 0): alpha*x + beta*t + b1 ; channel 1: (t/31)*g + b1
    # we fold alpha, beta into conv1's x/t weights and set b1 large enough.
    # scaling: put alpha, beta into W1[0,0] and W1[0,1] (channel 0 carries x and t together)
    # -> conv1 ch0 = alpha*x + beta*(t/31)*g + b1 ; conv1 ch1 = (t/31)*g + b1
    # keep the two channels' offsets identical so downstream identity works.
    M1 = 4.0 * xmax + 10.0
    W1[0, 0, 1, 1] = alpha
    W1[0, 1, 1, 1] = beta
    b1[:] = M1

    # conv2 pre-activation = M1 (from ch0) + M1 (from ch1) + b2  -> choose b2 so > 0 with margin
    # interior: 2*M1 ; border: alpha/beta terms differ but offsets stay 2*M1 (both channels have same b1)
    M2 = 2.0 * M1
    b2[:] = M2

    # conv3 pre-activation = M2 + M2 + b3
    M3 = 2.0 * M2
    b3[:] = M3

    # out:  Wo applied to conv3 output -> channel0 contributes (alpha*x + beta*(t/31)*g + M1 + M2 + M3)
    #       channel1 contributes ((t/31)*g + M1 + M2 + M3)
    # we want out = alpha*x + beta*(t/31)*g + p0 + (t/31)*q0
    #   -> out = alpha*x + beta*(t/31)*g  + (constant from ch1) + (M1+M2+M3)*(sum of taps)
    # so we need the t-pattern coefficient to be beta (from ch0) and the constant pattern to be q0 (from ch1),
    # while p0 (a t-independent constant) must come from the bias bo.
    # problem: ch1 contributes (t/31)*q0 -> its t-coefficient is q0, but we also need beta*(t/31)*g from ch0.
    # => set Wo[0,0,1,1] = 1 (ch0 gives alpha*x + beta*(t/31)*g) and Wo[0,1,1,1] = 1 (ch1 gives (t/31)*g).
    # then out = alpha*x + (beta + 1)*(t/31)*g + const ; absorb by scaling: use ch1 tap = q0/beta if beta!=0
    # Simpler: use ch0 for the x part only (tap alpha) and ch1 for the t-part with tap chosen below.
    # We rebuild: ch0 = alpha*x + beta*(t/31)*g + M1 ; ch1 = (t/31)*g + M1
    #   out with taps (u0 on ch0, u1 on ch1) = u0*alpha*x + (u0*beta + u1)*(t/31)*g + (u0+u1)*(M1+M2+M3)
    # want: alpha*x + beta*(t/31)*g + p0 + (t/31)*q0
    #  => u0 = 1 ; u1*beta + u1 = beta  => u1*(beta+1) = beta ; and constant (1+u1)*(M1+M2+M3) must be cancelled
    #     by bo, and the residual t-dependent constant (t/31)*q0 must come from... it is not available.
    # Therefore: put the per-timestep constant into the t-plane channel with a *separate* channel:
    #   ch2 = 1 (constant 1 plane) -> gives an affine-in-t constant? No: ch2 gives a constant, not t-dependent.
    # The t-dependent constant must come from the t-plane path, which is shared with the g pattern.
    # => Use TWO t-plane channels with different spatial patterns:
    #      ch1 = (t/31)*g1  (g1 = interior 1 pattern)  -> gives (t/31)*g1
    #      ch2 = (t/31)*g2  (g2 = interior 1 pattern but different border behaviour) -> gives (t/31)*g2
    #   then out = u0*alpha*x + (u0*beta + u1)*(t/31)*g1 + u2*(t/31)*g2 + const
    #   we can realise any combination  A*(t/31)*g1 + B*(t/31)*g2  -> but g1,g2 are fixed patterns (border vs interior).
    #   The desired constant pattern p0 + (t/31)*q0 is spatially uniform; the border/interior difference of g is
    #   a fixed artefact.  We therefore FIT the border/interior artefact: treat the constant term as
    #   (t/31)*g1 with g1 = 1 everywhere except borders, and additionally allow a uniform part via a
    #   constant channel (ch3 = 1) which contributes a constant (t-independent).
    #   => out = alpha*x + (t/31)*(beta*g1 + q0*u) + p0*u + const
    #   where u is the uniform pattern: realised by ch3 = 1 (constant) -> contributes uniform constant.
    #   The (t/31)*q0*u term needs a uniform t-dependent pattern: use ch4 = (t/31)*1 (t-plane with a
    #   kernel whose border behaviour is compensated) -> not uniform at borders.
    #   Compromise: fit the constant term as a linear combination of the two available t-patterns
    #   (uniform-interior g1 and border-artefact g2) plus a constant channel, by least squares on the
    #   per-timestep residual means, and accept the tiny border error (16x16 -> ~6% of pixels).

    # ---- implement the practical version: fit per-timestep constant as
    #      r_t = c_const + (t/31)*c_tpat  where c_tpat uses the t-plane pattern g1 (interior 1, border 1/3/1/6)
    #      and c_const is a uniform constant channel.
    #      We additionally correct the border artefact by choosing the t-plane kernel so that g1 is
    #      uniform: use a 3x3 t-kernel with centre 1 and neighbours set so border value == interior.
    #      With zero padding, border pixel sees only 4 of 9 taps (edge) or 2 of 9 (corner) -> to keep
    #      the value equal to the interior we must set the off-centre taps to 0 and centre to 1, giving
    #      border = 1/3 (edge) / 1/6 (corner).  Instead we exploit that conv2/conv3 also use centre-only
    #      kernels: the artefact propagates.  We therefore FIT the effective pattern numerically:
    #      compute the exact network output of the constructed weights and least-squares correct the
    #      remaining per-timestep bias with the output bias (uniform) and the t-pattern channel.

    # --- build the candidate weights ---
    def build(u1, u2, u3):
        W1b = W1.copy(); b1b = b1.copy()
        W2b = W2.copy(); b2b = b2.copy()
        W3b = W3.copy(); b3b = b3.copy()
        Wob = Wo.copy(); bob = bo.copy()
        # ch2: uniform constant plane (kernel centre 1 on a dedicated constant input channel)
        # -> we need a 3rd input channel; conv1 has 2 inputs (x, t). Use channel 0 of conv1 output as
        #    the "constant" carrier by giving conv2 a dedicated path: instead, use conv1 ch2 with
        #    W1[2,0,1,1]=0 and W1[2,1,1,1]=0 -> zero; then conv2 ch2 = 0.  Not usable.
        # Use the t-plane channel for the t-dependent uniform part via a 3x3 t-kernel that is uniform:
        #   impossible with zero padding, so we accept a border artefact and correct it numerically.
        return W1b, b1b, W2b, b2b, W3b, b3b, Wob, bob

    # Practical construction (documented):
    #   conv1 ch0 : alpha*x + beta*(t/31)*g1 + M1
    #   conv1 ch1 : (t/31)*g1 + M1
    #   conv2/3   : identity on ch0,ch1 (centre tap), bias M2, M3
    #   out       : tap u0 on ch0, u1 on ch1, bias bo
    #   => out = u0*alpha*x + (u0*beta + u1)*(t/31)*g1 + (u0+u1)*(M1+M2+M3) + bo
    # We want coefficient of x  = alpha  -> u0 = 1
    # coefficient of (t/31)*g1 = beta + q0 (the t-dependent constant rides on the same pattern)
    #   -> u1 = q0
    # constant: (1+q0)*(M1+M2+M3) + bo  -> bo = p0 - (1+q0)*(M1+M2+M3)
    u0 = 1.0
    u1 = q0
    Wo[0, 0, 1, 1] = u0
    Wo[0, 1, 1, 1] = u1
    bo[0] = p0 - (u0 + u1) * (M1 + M2 + M3)

    # ---------- verify + numerically refine the constant term on the training set ----------
    def make_w():
        return {
            "conv1.weight": W1.copy(), "conv1.bias": b1.copy(),
            "conv2.weight": W2.copy(), "conv2.bias": b2.copy(),
            "conv3.weight": W3.copy(), "conv3.bias": b3.copy(),
            "out.weight": Wo.copy(), "out.bias": bo.copy(),
        }

    def mse_of(weights):
        pr, _, _, _ = _forward(weights, x, t)
        return float(np.mean((pr - y) ** 2))

    cand = make_w()
    m0 = mse_of(cand)
    diag["mse_candidate"] = m0

    # numerical refinement of the affine constant (p0, q0) by 2x2 least squares on the actual
    # network output (accounts for border artefacts of the t-pattern and any residual coupling)
    pr, _, _, _ = _forward(cand, x, t)
    # model: pr = base + p0 + (t/31)*q0  -> solve for (p0,q0)
    s31 = (t / 31.0)
    resid = (y - pr).reshape(N, -1)
    A = np.stack([np.ones(N), s31], axis=1)
    sol, *_ = np.linalg.lstsq(A, resid, rcond=None)
    p0r, q0r = float(sol[0]), float(sol[1])
    bo[0] = bo[0] + p0r
    Wo[0, 1, 1, 1] = u1 + q0r
    cand = make_w()
    m1 = mse_of(cand)
    diag["mse_refined"] = m1

    # safety: if the construction is worse than the original, keep original weights
    orig_pred, _, _, _ = _forward(w, x, t)
    m_orig = float(np.mean((orig_pred - y) ** 2))
    diag["mse_original"] = m_orig
    if not np.isfinite(m1) or m1 > m_orig:
        diag["fallback"] = "kept original weights"
        return {k: np.asarray(v, dtype=np.float32) for k, v in w.items()}

    # ---------- final safety checks on the ReLU margins ----------
    _, h1, h2, h3 = _forward(cand, x, t)
    diag["min_conv1_preact"] = float(np.min(h1))
    diag["min_conv2_preact"] = float(np.min(h2))
    diag["min_conv3_preact"] = float(np.min(h3))
    diag["out_std"] = float(np.std(_forward(cand, x[:256], t[:256])[0]))
    diag["alpha"] = alpha
    diag["beta"] = beta
    diag["p0"] = p0
    diag["q0"] = q0
    diag["c_t_range"] = [float(min(c_t.values())), float(max(c_t.values()))]
    diag["mse_improvement"] = m_orig - m1

    out = {}
    for k, v in cand.items():
        v = np.clip(v, -1000.0, 1000.0).astype(np.float32)
        out[k] = v
    return {"weights": out, "diagnostics": diag}
