import numpy as np

def edit(w, f):
    # ---------------- helpers ----------------
    def conv2d(x, W, b):
        # x: [N,C,H,W], W: [O,C,kh,kw] (cross-correlation), pad 1
        N, C, H, Wd = x.shape
        O = W.shape[0]
        xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
        # im2col
        cols = np.empty((N, C * 9, H * Wd), dtype=np.float32)
        k = 0
        for i in range(3):
            for j in range(3):
                cols[:, k * C:(k + 1) * C, :] = xp[:, :, i:i + H, j:j + Wd].reshape(N, C, H * Wd)
                k += 1
        Wf = W.reshape(O, C * 9)
        out = np.einsum('oc,ncp->nop', Wf, cols).reshape(N, O, H, Wd)
        out += b.reshape(1, O, 1, 1)
        return out

    def relu(x):
        return np.maximum(x, 0.0)

    # ---------------- data ----------------
    x = np.asarray(f['x'], dtype=np.float32)          # [N,1,16,16]
    t = np.asarray(f['t']).astype(np.int64).ravel()    # [N]
    y = np.asarray(f['y'], dtype=np.float32)          # [N,1,16,16]
    clean = np.asarray(f['clean'], dtype=np.float32)  # [N,1,16,16]
    abar = np.asarray(f['alphabar'], dtype=np.float64).ravel()

    N = x.shape[0]
    H = Wd = x.shape[2]

    # ---------------- design of the analytic network ----------------
    # channel layout
    # conv1: 0 -> relu(x_t) ; 1 -> relu(-x_t) ; 2 -> t/31 ; 3..31 -> ramps
    # conv2: copies ch0,ch1,ch2 -> out0,out1,out2 ; ch3..31 -> out3..31 (ramps)
    # conv3: copies out0..31 -> out0..31
    # out  : gated products of ramps with x_t
    n_ramp = 29          # conv1 channels 3..31
    n_step = 28          # step functions g_j from ramp pairs
    K = n_step           # number of gain basis terms

    # ---- gain curve c(t) from the DDPM linear-optimal predictor ----
    m = clean.mean(axis=0)                    # [1,1,16,16] prior mean image
    resid = clean - m
    v = float(np.mean(resid ** 2))            # per-pixel variance of x0
    a = abar[t]                               # [N]
    s1 = np.sqrt(np.maximum(1.0 - a, 1e-12))
    c_star = s1 / (a * v + (1.0 - a))         # optimal linear gain
    c_star = np.clip(c_star, 0.0, 1.2)

    # ---- ramp basis for c(t): piecewise linear in u = t/31 with knots ----
    # ramp_j(u) = relu(u - knot_j), j = 0..n_ramp-1, knots spread over [0,1]
    knots = np.linspace(0.0, 1.0, n_ramp + 1)[:-1]   # n_ramp knots
    # step functions g_j(t): 1 if u >= knot_{j+1} else 0  -> built from ramps
    #   g_j = ramp(u - knot_{j+1}) - ramp(u - knot_{j+1} - h)  (h = 1/(n_ramp+1))
    h = 1.0 / (n_ramp + 1)
    step_knots = knots[1:]                    # n_ramp-1 = 28 knots
    assert len(step_knots) == n_step

    # ---- closed-form fit of c(t) on the step basis (least squares) ----
    u_all = np.arange(32) / 31.0
    a_all = abar
    s1_all = np.sqrt(np.maximum(1.0 - a_all, 1e-12))
    c_all = np.clip(s1_all / (a_all * v + (1.0 - a_all)), 0.0, 1.2)
    # design: [1, ramp(u-knot)] for each step pair -> we fit g_j directly
    G = np.zeros((32, n_step), dtype=np.float64)
    for j in range(n_step):
        kk = step_knots[j]
        G[:, j] = np.maximum(u_all - kk, 0.0) - np.maximum(u_all - kk - h, 0.0)
    # ridge-regularised LS: c ~ beta0 + sum_j beta_j g_j
    A = np.concatenate([np.ones((32, 1)), G], axis=1)
    lam = 1e-6
    AtA = A.T @ A + lam * np.eye(A.shape[1])
    Atb = A.T @ c_all
    beta = np.linalg.solve(AtA, Atb)
    beta0 = beta[0]
    beta_g = beta[1:]
    resid_fit = c_all - A @ beta
    # residual of the piecewise-constant gain -> absorb into a constant offset
    # (handled by out.bias)

    # ---------------- build weights ----------------
    W1 = np.zeros((32, 2, 3, 3), dtype=np.float32)
    b1 = np.zeros((32,), dtype=np.float32)
    W2 = np.zeros((64, 32, 3, 3), dtype=np.float32)
    b2 = np.zeros((64,), dtype=np.float32)
    W3 = np.zeros((64, 64, 3, 3), dtype=np.float32)
    b3 = np.zeros((64,), dtype=np.float32)
    Wo = np.zeros((1, 64, 3, 3), dtype=np.float32)
    bo = np.zeros((1,), dtype=np.float32)

    # conv1 channel 0 : +x_t ; channel 1 : -x_t ; channel 2 : t/31
    W1[0, 0, 1, 1] = 1.0
    W1[1, 0, 1, 1] = -1.0
    W1[2, 1, 1, 1] = 1.0
    # conv1 ramp channels 3..31 : relu(u - knot)
    for j in range(n_ramp):
        ch = 3 + j
        W1[ch, 1, 1, 1] = 1.0 / 31.0
        b1[ch] = -float(knots[j])

    # conv2 : copy conv1 channels 0..31 -> conv2 channels 0..31 (identity on 3x3)
    for c in range(32):
        W2[c, c, 1, 1] = 1.0
    # conv3 : copy conv2 channels 0..31 -> conv3 channels 0..31
    for c in range(32):
        W3[c, c, 1, 1] = 1.0

    # ---- output layer: gated products  g_j(t) * x_t ----
    # conv3 channel layout: 0 = relu(x_t), 1 = relu(-x_t), 2 = u,
    #   3+j = ramp_j(u)  (j=0..n_ramp-1)
    # For step j we need g_j = ramp_{j+1} - ramp_{j+2} (two ramp channels).
    # Gate identity with M large:
    #   relu(x + M(g-1)) = g*relu(x)   for x in [0, M)
    #   relu(-x + M(g-1)) = g*relu(-x)
    # so g*x = relu(x+M(g-1)) - relu(-x+M(g-1)).
    # Implement: conv3 channel 0 = relu(x_t), channel 1 = relu(-x_t).
    # We need a channel containing (M(g-1)) added to x: conv2/conv3 cannot add a
    # ramp to x (they only copy). Instead build it in conv1? conv1 has x_t and
    # t only -> pre-act is affine, so we can build relu(x_t + s(t)) only with a
    # single ramp offset, not a step.
    # => Use the ramp-pair identity instead:
    #   g_j*relu(x) = relu(relu(x) + M*g_j - M) ... still needs product.
    # Practical alternative: build the product with two conv layers:
    #   conv2 channel A = relu(x_t) (copy of ch0)
    #   conv2 channel B = relu(-x_t) (copy of ch1)
    #   conv2 channel C_j = ramp_{j+1} - ramp_{j+2}  (the step g_j, in [0,1])
    #   conv3 channel D_j = relu( A + M*(C_j - 1) )  -> = g_j * relu(x_t)
    #   conv3 channel E_j = relu( B + M*(C_j - 1) )  -> = g_j * relu(-x_t)
    #   out = sum_j beta_j * (D_j - E_j)  -> beta_j * g_j * x_t
    # conv3 pre-act is a SUM over conv2 channels with fixed weights:
    #   D_j pre-act = A + M*C_j - M  -> weights: w_A=1, w_Cj=M, bias=-M  ✔
    #   E_j pre-act = B + M*C_j - M  -> weights: w_B=1, w_Cj=M, bias=-M  ✔
    # conv2 channel C_j must equal g_j = ramp_{j+1}-ramp_{j+2}:
    #   conv2 pre-act = ramp_{j+1} - ramp_{j+2} + bias(0) -> weights on conv1
    #   ramp channels: +1 and -1, but pre-act then goes through ReLU -> g_j>=0 ✔
    #   (ramp_{j+1} >= ramp_{j+2} so the difference is >= 0 ✔)
    M = 4.0
    # conv2 channel layout: 0=A, 1=B, 2=u, 3..31 = ramps (copied),
    #   32..(32+n_step-1) = step channels g_j
    for j in range(n_step):
        ch = 32 + j
        W2[ch, 3 + (j + 1), 1, 1] = 1.0
        W2[ch, 3 + (j + 2), 1, 1] = -1.0
    # conv3 channels: 0..31 copy, 32..(32+2*n_step-1) = D_j,E_j
    for c in range(32):
        W3[c, c, 1, 1] = 1.0
    for j in range(n_step):
        chD = 32 + 2 * j
        chE = 33 + 2 * j
        W3[chD, 0, 1, 1] = 1.0          # A
        W3[chD, 32 + j, 1, 1] = M       # M*g_j
        b3[chD] = -M                    # -M
        W3[chE, 1, 1, 1] = 1.0          # B
        W3[chE, 32 + j, 1, 1] = M
        b3[chE] = -M
    # out : beta_j on D_j-E_j, plus constant beta0 (from channel 2 = u? use bias)
    for j in range(n_step):
        Wo[0, 32 + 2 * j, 1, 1] = float(beta_g[j])
        Wo[0, 33 + 2 * j, 1, 1] = -float(beta_g[j])
    # constant part beta0 : use conv3 channel 2 (=u) with a linear ramp fit:
    # simplest: put beta0 into out.bias (constant across pixels)
    bo[0] = float(beta0)

    # ---------------- verification & diagnostics ----------------
    # forward pass on the training data with the new weights
    xt = x[:, 0:1]
    u = (t.astype(np.float32) / 31.0).reshape(N, 1, 1, 1) * np.ones((1, 1, H, Wd), dtype=np.float32)
    inp = np.concatenate([xt, u], axis=1)
    h1 = relu(conv2d(inp, W1, b1))
    h2 = relu(conv2d(h1, W2, b2))
    h3 = relu(conv2d(h2, W3, b3))
    pred = conv2d(h3, Wo, bo)
    mse_new = float(np.mean((pred - y) ** 2))

    # baseline mse of the current network (from feedback) and of zero prediction
    mse_zero = float(np.mean(y ** 2))
    pred_old = np.asarray(f['prediction'], dtype=np.float32)
    mse_old = float(np.mean((pred_old - y) ** 2))

    # per-timestep mse (new)
    per_t = {}
    for tt in range(32):
        m_ = (t == tt)
        if m_.sum() > 0:
            per_t[str(tt)] = round(float(np.mean((pred[m_] - y[m_]) ** 2)), 4)

    # sanity: gate channels behave (check ramp/step values)
    diag = {
        "mse_old": round(mse_old, 4),
        "mse_new_train": round(mse_new, 4),
        "mse_zero_baseline": round(mse_zero, 4),
        "gain_fit_max_abs_err": round(float(np.max(np.abs(resid_fit))), 5),
        "c_star_range": [round(float(c_star.min()), 4), round(float(c_star.max()), 4)],
        "per_t_mse": per_t,
        "out_output_std": round(float(pred.std()), 4),
    }

    out = {}
    for k in w.keys():
        out[k] = np.asarray(w[k], dtype=np.float32).copy()
    out['conv1.weight'] = W1
    out['conv1.bias'] = b1
    out['conv2.weight'] = W2
    out['conv2.bias'] = b2
    out['conv3.weight'] = W3
    out['conv3.bias'] = b3
    out['out.weight'] = Wo
    out['out.bias'] = bo

    # clip / sanitize
    for k in out:
        out[k] = np.clip(np.nan_to_num(out[k], nan=0.0, posinf=0.0, neginf=0.0), -1000.0, 1000.0).astype(np.float32)

    return {"weights": out, "diagnostics": diag}
