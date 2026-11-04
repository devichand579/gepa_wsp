import numpy as np


def edit(w, f):
    """Closed-form reconstruction of the epsilon-predictor.

    Hypothesis: the current net is untrained (MSE 1.0058 > 1.0 = MSE of predicting
    zero).  We rebuild the fixed architecture as an (almost) affine map of (x_t, t)
    that implements a Wiener filter for epsilon plus a pixel-wise quadratic
    correction, all coefficients obtained by closed-form least squares on the
    supplied training feedback.  No gradients, no iterative optimisation.
    """
    rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ helpers
    def conv2d(x, W, b):
        """x: (N,C,H,W) ; W: (O,C,3,3) cross-correlation, pad 1."""
        N, C, H, Wd = x.shape
        O = W.shape[0]
        xp = np.zeros((N, C, H + 2, Wd + 2), dtype=np.float32)
        xp[:, :, 1:-1, 1:-1] = x
        cols = np.empty((N, C, 9, H, Wd), dtype=np.float32)
        for kr in range(3):
            for kc in range(3):
                cols[:, :, kr * 3 + kc] = xp[:, :, kr:kr + H, kc:kc + Wd]
        out = np.einsum('ockhw,nchw->nohw', W.astype(np.float32), cols, optimize=True)
        out += b.astype(np.float32).reshape(1, O, 1, 1)
        return out

    def boxsum(x, k):
        """sum over a k x k window, 'same' size, stride 1, zero padded."""
        N, C, H, Wd = x.shape
        xp = np.zeros((N, C, H + k - 1, Wd + k - 1), dtype=np.float32)
        xp[:, :, :H, :Wd] = x
        cs = np.cumsum(np.cumsum(xp, axis=2), axis=3)
        cs = np.pad(cs, ((0, 0), (0, 0), (1, 0), (1, 0)))
        out = (cs[:, :, k - 1:k - 1 + H, k - 1:k - 1 + Wd]
               - cs[:, :, 0:H, k - 1:k - 1 + Wd]
               - cs[:, :, k - 1:k - 1 + H, 0:Wd]
               + cs[:, :, 0:H, 0:Wd])
        return out

    def smooth(x, k):
        return boxsum(x, k) / float(k * k)

    # ------------------------------------------------------------------ data
    x = np.asarray(f['x'], dtype=np.float32)            # (N,1,16,16)
    y = np.asarray(f['y'], dtype=np.float32)            # (N,1,16,16)
    t = np.asarray(f['t'], dtype=np.int64).ravel()       # (N,)
    clean = np.asarray(f['clean'], dtype=np.float32)     # (N,1,16,16)
    ab = np.asarray(f['alphabar'], dtype=np.float64).ravel()
    N = x.shape[0]
    H = Wd = x.shape[2]
    P = H * Wd

    xf = x.reshape(N, P).astype(np.float64)
    yf = y.reshape(N, P).astype(np.float64)
    cf = clean.reshape(N, P).astype(np.float64)

    abar = np.clip(ab[t], 1e-6, 1.0 - 1e-6)
    s = np.sqrt(abar)[:, None]
    n = np.sqrt(1.0 - abar)[:, None]

    # ------------------------------------------------- spatial covariance of x0
    C0 = np.zeros((P, P), dtype=np.float64)
    for a in range(0, N, 250):
        b = min(N, a + 250)
        blk = cf[a:b]
        C0 += blk.T @ blk
    C0 /= float(N)
    m0 = C0.mean(axis=1)                      # spatial mean profile of clean images
    C0c = C0 - np.outer(m0, m0)
    C0c = 0.5 * (C0c + C0c.T)

    # --------------------------------------------- per-timestep linear (Wiener)
    # eps_hat = A_t x + b_t  with A_t = n_t (s^2 C0 + n^2 I)^{-1}
    # We realise A_t as a filter applied to a *normalised* image channel, so the
    # per-timestep scalar 1/n_t is folded into the output layer (t-dependent).
    I_P = np.eye(P)
    A_by_t = {}
    for tt in range(32):
        idx = np.where(t == tt)[0]
        if idx.size == 0:
            continue
        st = float(s[idx[0], 0])
        nt = float(n[idx[0], 0])
        M = (st * st) * C0c + (nt * nt) * I_P
        try:
            A = nt * np.linalg.inv(M)
        except np.linalg.LinAlgError:
            A = nt * np.linalg.pinv(M)
        A_by_t[tt] = A

    # effective (pixel-wise) filter of the whole net: out = (1/n_t) * A_t x
    # -> we implement A_t via a *fixed* spatial filter bank and let the output
    #    layer pick per-timestep coefficients (t enters through conv1 ch1).
    # Because A_t = n_t (s^2 C0c + n^2 I)^{-1} depends on t nonlinearly, we
    #    approximate it with a low-rank + diagonal form that is exactly
    #    representable by 3x3 convolutions:
    #       A_t x = sum_{j<=K} u_j(t) * (v_j . x) * e_j  +  d(t) * x
    #    (e_j = spatial basis functions, realised as 3x3 conv kernels).
    # We use an eigendecomposition of C0c and keep the leading K modes.
    evals, evecs = np.linalg.eigh(C0c)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    K = 12
    lam = np.maximum(evals[:K], 0.0)
    U = evecs[:, :K]                                  # (P,K)

    # spatial basis functions U must be realised by 3x3 convs -> smooth them
    #    and keep the smooth part (high-frequency modes are not representable).
    Us = U.reshape(H, Wd, K).copy()
    for j in range(K):
        Us[:, :, j] = smooth(Us[:, :, j][None, None], 3)[0, 0]
    # re-orthonormalise the smoothed basis (QR) for stability
    Q, _ = np.linalg.qr(Us.reshape(P, K))
    Us = Q

    # per-timestep coefficients of the low-rank part and the diagonal part
    coef = {}
    for tt in range(32):
        if tt not in A_by_t:
            continue
        st = float(np.sqrt(ab[tt]))
        nt = float(np.sqrt(1.0 - ab[tt]))
        # A_t = n (s^2 C0c + n^2 I)^{-1};  low-rank approx via Woodbury on the
        # smoothed subspace:
        # (s^2 C0c + n^2 I)^{-1} ~ sum_j u_j u_j^T /(s^2 lam_j + n^2) + (I-UU^T)/n^2
        d = 1.0 / (st * st * lam + nt * nt) if len(lam) else np.zeros(0)
        # diagonal (isotropic) part applied to x directly
        diag = 1.0 / nt
        coef[tt] = (d, diag)

    # ------------------------------------------- pixel-wise quadratic features
    # xbar_t = mean over pixels of x for each example (per timestep mean)
    xbar = xf.mean(axis=1)
    # per-timestep mean of xbar
    tb_mean = np.zeros(32)
    for tt in range(32):
        idx = t == tt
        if idx.any():
            tb_mean[tt] = xbar[idx].mean()
    xb = (xbar - tb_mean[t])[:, None]
    q = xb * xb                                  # (N,1) quadratic feature
    qc = q - q.mean(axis=0, keepdims=True)

    # ------------------------------------------------- closed-form ridge fit
    # Design matrix columns (per pixel, per example):
    #   1) x          (linear, spatial)
    #   2) low-rank projections  (K channels)
    #   3) constant   (per timestep intercept -> handled by per-t bias)
    #   4) quadratic feature q
    # We fit, per timestep, a linear model:
    #   y ~ sum_j c_j * (u_j . x) + d * x + g * q + b
    # where (u_j . x) is a projection of x onto the smoothed basis.
    # All of these are produced by conv layers; the output layer holds the
    # per-timestep coefficients via the t-channel of conv1 (piecewise linear
    # interpolation of the 32 coefficient vectors).
    # To keep the fit tractable we use the *pixel-level* least squares with the
    # spatial structure of the target retained through the basis functions.

    # compute projections of x onto the smoothed basis (per example)
    proj = np.empty((N, K), dtype=np.float64)
    for j in range(K):
        uj = Us[:, j].reshape(H, Wd)
        proj[:, j] = (xf * uj.reshape(1, P)).sum(axis=1)

    # per-timestep ridge regression on the *pixel* level:
    #   features per example: [proj (K), xbar_centered, q_centered, 1]
    #   target: y (P dims)  -> we need per-pixel coefficients, so instead we
    #   regress each pixel's y on the *spatial* regressors: for pixel p,
    #   y_p ~ sum_j c_j * (u_j . x) + d * x_p + g * q + b
    # Solve jointly for all pixels: this is a huge design matrix; instead we
    # exploit that the coefficients (c_j, d, g, b) are shared across pixels
    # while the regressors differ per pixel -> we can accumulate normal
    # equations by looping over pixels (256) with per-timestep blocks.
    # Cost: 32 timesteps x 256 pixels x (K+3)^2  -> fine.

    Kreg = K + 3
    ridge = 1e-6
    coef_by_t = {}
    diag_by_t = {}
    for tt in range(32):
        idx = np.where(t == tt)[0]
        if idx.size < 8:
            continue
        ni = idx.size
        # regressors per example: proj (K), xbar_c, q_c, 1
        R = np.empty((ni, Kreg), dtype=np.float64)
        R[:, :K] = proj[idx]
        R[:, K] = xb[idx, 0]
        R[:, K + 1] = qc[idx, 0]
        R[:, K + 2] = 1.0
        # normal equations accumulated over pixels
        A = np.zeros((Kreg, Kreg))
        B = np.zeros((Kreg, P))
        for p in range(P):
            # per-pixel regressors: same R but with x_p added
            Rp = np.empty((ni, Kreg + 1), dtype=np.float64)
            Rp[:, :K] = R[:, :K]
            Rp[:, K] = xf[idx, p]
            Rp[:, K + 1] = R[:, K]
            Rp[:, K + 2] = R[:, K + 1]
            Rp[:, K + 3] = R[:, K + 2]
            A += Rp.T @ Rp
            B += Rp.T @ yf[idx, p]
        A += ridge * np.trace(A) / Kreg * np.eye(Kreg)
        sol = np.linalg.solve(A, B)      # (Kreg, P)
        coef_by_t[tt] = sol[:K]           # per-mode coefficients (K,P)
        diag_by_t[tt] = sol[K]            # per-pixel diagonal coefficient (P,)
        # store intercept and quadratic coefficient (scalars) for later
        coef_by_t[tt + 100] = np.array([sol[K + 1, 0], sol[K + 2, 0]])

    # --------------------------------------------- build the network weights
    # Channel plan (conv1 -> 32 channels):
    #   ch 0 : x_t                       (identity 3x3, centre tap)
    #   ch 1 : t/31                      (constant over space)
    #   ch 2..K+1 : smoothed projections  sum_p u_j(p) x(p)  (3x3 kernels)
    #   ch K+2 : squared centred x        (via ReLU squaring cell)
    #   ch K+3 : t/31 again (spare)       -> we use it for the intercept ramp
    # conv2 (32->64): copies channels 0..K+3 into 64 channels (identity)
    # conv3 (64->64): copies channels 0..K+3 into 64 channels (identity)
    # out  (64->1): 3x3 kernel with per-channel weights; but the per-timestep
    #   coefficients differ per t -> we cannot use a single static out kernel.
    #   SOLUTION: put the per-timestep coefficient selection in conv1/conv2/conv3
    #   is impossible (they are shared across t as well).
    #   => Instead: use the t-channel to *gate* the coefficients through the
    #   ReLU nonlinearity: for each mode j we create a channel that equals
    #   c_j(t) * (u_j . x) by making conv1 output = c_j(t) * (u_j.x) directly.
    #   But c_j(t) is t-dependent and conv1 weights are static.
    #   => Use the following trick: conv1 channel for mode j outputs
    #      (u_j . x) * 0 + B  ... no.
    #   Practical resolution: the per-timestep coefficients are *piecewise
    #   linear in t*; we approximate them by a rank-1 structure in t:
    #      c_j(t) ~= c_j^{(0)} + c_j^{(1)} * (t/31)
    #   and realise the product (u_j.x) * (t/31) inside conv2 by a
    #   ReLU-based multiplication cell (two channels: a = u_j.x + BIG,
    #   b = (t/31) + BIG; then relu(a+b-2BIG) is not a product).
    #   Instead we use the *affine* structure: because the whole net can be made
    #   affine (large biases), products are impossible -> we therefore use a
    #   different, fully legitimate route: fold the t-dependence into the
    #   *spatial* filters by using a per-timestep channel that is a *constant*
    #   and combining it with x through the final 3x3 convolution is impossible
    #   as well (conv is linear in x).
    #   FINAL DESIGN: use the ReLU non-linearity deliberately to obtain the
    #   product: for a channel pair we can build
    #       relu(A + B) - relu(A - B) = 2*min(A,B)   (not a product)
    #   but we can build a *quadratic* in x via the squaring cell, and a
    #   *bilinear* in (x, t) via:  relu(u.x + s*t + C) trick does not multiply.
    #   => Therefore we restrict the t-dependence to the *scalar* output
    #      scaling and intercept, which IS possible: the out layer computes
    #      sum over channels of w_c * f_c where f_c are t-independent spatial
    #      features, and the t-dependence enters through the *input* channel
    #      t/31 which is spatially constant: out = sum_c w_c f_c + sum_c w'_c * t
    #      -> the intercept can be t-dependent (linear in t) but the slope
    #      cannot.
    #   We therefore make the *slope* t-independent by normalising the target:
    #      define  z = eps / n_t   (target scaled by 1/sqrt(1-abar_t)).
    #      Then the predictor for z is  z_hat = (x - s*m0)/n_t ... still t-dep.
    #   Instead: work with the *whitened* variable:
    #      u = x / n_t  and  v = eps / n_t  ... same issue.
    #   Practical compromise (still fully closed form):
    #      fit a single set of spatial filter coefficients valid for all t on
    #      the *normalised* problem:
    #          eps_hat = a(t) * (F x) + b(t)
    #      where F is a fixed spatial filter (3x3 representable) and a(t), b(t)
    #      are scalars per timestep.  a(t) and b(t) enter the network through
    #      the t-channel: out = a(t)*Fx + b(t) requires a product of a(t) with
    #      Fx  -> not representable.
    #   => FINAL: use the *scaling inside the input normalisation*: we cannot
    #      scale x by a per-t scalar either (conv1 weights are static).
    #   => Therefore: accept a t-independent slope but choose the target
    #      parameterisation that makes the slope t-independent: predict
    #          eps_hat = (1/n_t) * (x - s_t * m0)   requires 1/n_t scaling.
    #   The ONLY way to get a t-dependent scalar gain in a static CNN is to
    #      use the ReLU nonlinearity as a *switch* (different linear regions for
    #      different t).  We exploit exactly that: the t/31 channel is
    #      constant in space, so a ReLU of (t/31 - c) is a step function of t
    #      (0 or 1).  Using a bank of 32 such step functions we can realise ANY
    #      piecewise-constant-in-t function of the features:
    #          step_j(t) = relu(t/31 - c_j) - relu(t/31 - c_j - eps)
    #      and then out = sum_j step_j(t) * (per-timestep filter applied to x).
    #      But the per-timestep filter applied to x still needs a per-timestep
    #      spatial filter -> we need 32 different spatial filters, each gated.
    #      That is 32 * (number of spatial features) channels -> too many for
    #      64 channels?  We can afford 64 channels: use 32 gates and 32
    #      spatial features?  But then the filter must be shared across t.
    #   => Compromise that is exactly representable and near-optimal:
    #      realise the *spatial* filter bank with K+2 channels (shared across t)
    #      and the *t-dependent gain* with 32 gate channels, and combine them
    #      in the final 3x3 output convolution?  A 3x3 conv is linear: it cannot
    #      multiply a gate by a feature.
    #   => Accept: t-dependent GAIN is impossible in a static CNN without
    #      multiplication.  So we build the network to output
    #          eps_hat = F(x) + b(t)
    #      i.e. a t-INDEPENDENT spatial filter plus a t-dependent intercept
    #      (intercept is easy: it is a linear function of the t-channel).
    #      To make the t-independent filter optimal we fit it on the
    #      *normalised* target  eps * n_t  ... let's simply fit F and the
    #      intercepts by least squares over ALL timesteps simultaneously with
    #      the target  eps_hat = F(x) + b(t)  (b(t) piecewise-linear in t via
    #      32 gate channels -> exactly representable).
    #      The optimal such F is a single spatial Wiener filter shared across
    #      t; we compute it in closed form from the data.

    # ---- closed form: shared spatial filter F (3x3 representable) + per-t bias
    # Model: y_p = sum_{q in 3x3 nbhd} F[q] * x_q + b(t)
    # Solve for F (9 params) and b(t) (32 params) jointly by ridge LS.
    # Build the 3x3 neighbourhood design for x (N*P rows, 9 cols) - do it
    # per timestep block to keep memory bounded.
    def nbhd_feats(idx):
        xx = xf[idx]                                   # (ni,P)
        ni = xx.shape[0]
        cols = np.empty((ni, P, 9), dtype=np.float64)
        img = xx.reshape(ni, H, Wd)
        pad = np.zeros((ni, H + 2, Wd + 2))
        pad[:, 1:-1, 1:-1] = img
        for kr in range(3):
            for kc in range(3):
                cols[:, :, kr * 3 + kc] = pad[:, kr:kr + H, kc:kc + Wd].reshape(ni, P)
        return cols

    # joint design: rows = (example, pixel); unknowns: F(9) + b(32)
    NU = 9 + 32
    A = np.zeros((NU, NU))
    B = np.zeros(NU)
    for tt in range(32):
        idx = np.where(t == tt)[0]
        if idx.size == 0:
            continue
        cols = nbhd_feats(idx)                        # (ni,P,9)
        ni = idx.size
        yv = yf[idx]                                  # (ni,P)
        # normal equations contributions
        # sum over pixels and examples of cols^T cols  (9x9)
        # sum_p cols[:,p,a]*cols[:,p,b]
        Cf = cols.reshape(ni * P, 9)
        A[:9, :9] += Cf.T @ Cf
        # cross with bias: sum_p cols[:,p,a] * 1
        A[:9, 9 + tt] += Cf.sum(axis=0)
        A[9 + tt, :9] += Cf.sum(axis=0)
        A[9 + tt, 9 + tt] += float(ni * P)
        B[:9] += Cf.T @ yv.reshape(-1)
        B[9 + tt] += yv.sum()
    A += 1e-4 * np.trace(A) / NU * np.eye(NU)
    sol = np.linalg.solve(A, B)
    F9 = sol[:9].reshape(3, 3)
    bias_t = sol[9:]

    # ---- add a pixel-wise quadratic correction (t-independent gain on q)
    # Model: y_p = sum_q F[q] x_q + b(t) + g * q_p  (q = centred xbar^2)
    # Solve jointly (F, b(t), g).
    NU2 = 9 + 32 + 1
    A2 = np.zeros((NU2, NU2))
    B2 = np.zeros(NU2)
    for tt in range(32):
        idx = np.where(t == tt)[0]
        if idx.size == 0:
            continue
        cols = nbhd_feats(idx)
        ni = idx.size
        yv = yf[idx].reshape(-1)
        qv = np.repeat(qc[idx, 0], P)
        Cf = cols.reshape(ni * P, 9)
        A2[:9, :9] += Cf.T @ Cf
        A2[:9, 9 + tt] += Cf.sum(axis=0)
        A2[9 + tt, :9] += Cf.sum(axis=0)
        A2[9 + tt, 9 + tt] += float(ni * P)
        A2[:9, 32] += Cf.T @ qv
        A2[32, :9] += qv @ Cf
        A2[32, 9 + tt] += qv.sum()
        A2[9 + tt, 32] += qv.sum()
        A2[32, 32] += float(qv.size)
        B2[:9] += Cf.T @ yv
        B2[9 + tt] += yv.sum()
        B2[32] += qv @ yv
    A2 += 1e-4 * np.trace(A2) / NU2 * np.eye(NU2)
    sol2 = np.linalg.solve(A2, B2)
    F9 = sol2[:9].reshape(3, 3)
    bias_t = sol2[9:41]
    gcoef = float(sol2[41])

    # ------------------------------------------------------------- build net
    # conv1: 32 -> channels
    W1 = np.zeros((32, 2, 3, 3), dtype=np.float32)
    b1 = np.zeros(32, dtype=np.float32)
    BIG = 8.0
    # ch0: x_t (identity 3x3: centre tap =1)  -> affine regime via bias
    W1[0, 0, 1, 1] = 1.0
    b1[0] = BIG
    # ch1: t/31 constant
    W1[1, 1, 1, 1] = 1.0
    b1[1] = BIG
    # ch2: x_t again (for the quadratic cell we need x itself)
    W1[2, 0, 1, 1] = 1.0
    b1[2] = BIG
    # ch3: t/31 again
    W1[3, 1, 1, 1] = 1.0
    b1[3] = BIG
    # remaining channels: zero (unused)
    # conv2: 32 -> 64 : copy ch0..ch3 into out channels 0..3, rest zero
    W2 = np.zeros((64, 32, 3, 3), dtype=np.float32)
    b2 = np.zeros(64, dtype=np.float32)
    for c in range(4):
        W2[c, c, 1, 1] = 1.0
    # conv3: 64 -> 64 : copy channels 0..3 into 0..3
    W3 = np.zeros((64, 64, 3, 3), dtype=np.float32)
    b3 = np.zeros(64, dtype=np.float32)
    for c in range(4):
        W3[c, c, 1, 1] = 1.0

    # out: 64 -> 1, 3x3 kernel.  We need:
    #   out = sum_q F[q] * x_q  +  g * q_p  +  b(t)
    # The spatial part: out.weight[:, ch0, :, :] = F9  (applied to ch0 = x)
    # The quadratic part: needs q_p = (xbar_c)^2 at each pixel -> we build it
    #   inside the network with a ReLU squaring cell:
    #       conv3 output ch_q = relu( (x + BIG)^2 ... ) -- not available.
    #   Instead: realise the quadratic term as a *pixel-wise* function of the
    #   local 3x3 neighbourhood of x, using the identity
    #       (a+b)^2 - (a-b)^2 = 4ab
    #   which is a product of two channels -> needs a multiplication.
    #   We CAN multiply using ReLU with the "gate" trick:
    #       relu(u + v - BIG) - relu(u - v + BIG)  ... not a product either.
    #   Practical approach: implement the quadratic term as a *linear* function
    #   of the local second moments that ARE linear in x? Not possible.
    #   => Use a different, exactly-representable nonlinear feature: the
    #      *local mean* of x (a 3x3 box filter) is linear; its square is not.
    #   => Use ReLU as a soft-threshold to obtain a piecewise-linear function
    #      of the local mean:  relu(m - c) is piecewise linear in m.  A sum of
    #      such terms gives a piecewise-linear approximation of q(m).  This is
    #      exactly representable: conv1 gives local mean (3x3 box kernel),
    #      conv2 applies relu(m - c_j) for several c_j (via bias), conv3
    #      combines them, out applies the coefficients.
    #   We therefore approximate the optimal per-pixel function of the local
    #      mean with a piecewise-linear (hinge) basis and fit the coefficients
    #      by closed-form least squares on the training data.
    #
    #  Implementation of the hinge basis:
    #    conv1 ch4..ch11 : local 3x3 box mean of x  (same kernel, different bias
    #                       -> different hinge thresholds after relu)
    #    conv2 copies them; relu gives hinge features h_j = relu(m - c_j)
    #    conv3 copies; out combines with fitted coefficients.
    #  We fit the hinge coefficients jointly with F9 and b(t) by ridge LS.

    # --- build hinge features and refit everything jointly
    # local mean feature: 3x3 box filter of x
    box = np.ones((3, 3), dtype=np.float32) / 9.0
    # hinge thresholds on the local mean (m in roughly [-1,1])
    knots = np.array([-0.6, -0.4, -0.2, -0.0, 0.2, 0.4, 0.6], dtype=np.float32)
    nk = len(knots)
    # conv1 channels: 0:x, 1:t, 2:x, 3:t, 4..(4+nk-1): local mean with bias
    W1 = np.zeros((32, 2, 3, 3), dtype=np.float32)
    b1 = np.zeros(32, dtype=np.float32)
    W1[0, 0, 1, 1] = 1.0
    b1[0] = BIG
    W1[1, 1, 1, 1] = 1.0
    b1[1] = BIG
    W1[2, 0, 1, 1] = 1.0
    b1[2] = BIG
    W1[3, 1, 1, 1] = 1.0
    b1[3] = BIG
    for j in range(nk):
        W1[4 + j, 0] = box
        b1[4 + j] = BIG - float(knots[j])
    # conv2: copy channels 0..(4+nk-1)
    nch = 4 + nk
    W2 = np.zeros((64, 32, 3, 3), dtype=np.float32)
    b2 = np.zeros(64, dtype=np.float32)
    for c in range(nch):
        W2[c, c, 1, 1] = 1.0
    # conv3: copy channels 0..(nch-1) into 0..(nch-1)
    W3 = np.zeros((64, 64, 3, 3), dtype=np.float32)
    b3 = np.zeros(64, dtype=np.float32)
    for c in range(nch):
        W3[c, c, 1, 1] = 1.0

    # ---- assemble design matrix for the joint fit:
    #   columns: 9 (spatial F on x) + 32 (b(t)) + nk (hinge coeffs)
    #   hinge feature h_j(p) = relu(m_p - knot_j) where m_p = local mean of x
    #   (the BIG offset cancels in the fit because we also fit an intercept per t)
    mloc = np.empty((N, P), dtype=np.float64)
    for a in range(0, N, 250):
        b = min(N, a + 250)
        blk = x[a:b]
        sm = smooth(blk, 3).reshape(b - a, P).astype(np.float64)
        mloc[a:b] = sm
    NU3 = 9 + 32 + nk
    A3 = np.zeros((NU3, NU3))
    B3 = np.zeros(NU3)
    for tt in range(32):
        idx = np.where(t == tt)[0]
        if idx.size == 0:
            continue
        cols = nbhd_feats(idx)
        ni = idx.size
        yv = yf[idx].reshape(-1)
        Cf = cols.reshape(ni * P, 9)
        Hm = np.maximum(mloc[idx].reshape(-1, 1) - knots.reshape(1, nk), 0.0)
        A3[:9, :9] += Cf.T @ Cf
        A3[:9, 9 + tt] += Cf.sum(axis=0)
        A3[9 + tt, :9] += Cf.sum(axis=0)
        A3[9 + tt, 9 + tt] += float(ni * P)
        A3[:9, 41:] += Cf.T @ Hm
        A3[41:, :9] += Hm.T @ Cf
        A3[41:, 9 + tt] += Hm.sum(axis=0)
        A3[9 + tt, 41:] += Hm.sum(axis=0)
        A3[41:, 41:] += Hm.T @ Hm
        B3[:9] += Cf.T @ yv
        B3[9 + tt] += yv.sum()
        B3[41:] += Hm.T @ yv
    A3 += 1e-4 * np.trace(A3) / NU3 * np.eye(NU3)
    sol3 = np.linalg.solve(A3, B3)
    F9 = sol3[:9].reshape(3, 3)
    bias_t = sol3[9:41]
    hcoef = sol3[41:]

    # ------------------------------------------------------------ assemble out
    # out.weight[0, ch, kr, kc]:
    #   ch 0 : F9  (spatial filter applied to x)
    #   ch 4+j : hcoef[j] * delta_centre  (hinge features are already centred
    #            conv outputs; the 3x3 kernel picks the centre pixel)
    #   ch 1,3 : t/31 channels -> used for the per-timestep bias.  We need
    #            b(t) for 32 timesteps but only 2 t-channels exist.  Instead we
    #            realise b(t) via the *hinge* mechanism on the t channel:
    #            add extra conv1 channels that are relu(t/31 - c) steps.
    #   => extend conv1 with 32 step channels: ch nch .. nch+31
    #      conv1: W1[ch,1,1,1]=1, b1[ch] = BIG - c_j  (c_j = t_j/31)
    #      after relu this equals BIG when t/31 >= c_j else 0 -> a step.
    #      out: weight = b(t_j) on channel ch (centre tap) -> gives b(t_j) when
    #      the step is on.  But the step is "on" for all t >= t_j, so the sum
    #      of steps gives a cumulative representation: b(t) = sum_j d_j S_j(t)
    #      with S_j(t) = 1[t >= t_j].  Solve for d_j by differencing b(t):
    #      d_j = b(t_j) - b(t_{j-1}) (with b(t_{-1})=0) gives
    #      sum_{j<=t} d_j = b(t).  Exactly representable!
    #      Note: the step channel value is BIG (not 1) after relu, so the out
    #      weight must be d_j / BIG.
    #      Also the t channel value is t/31 + BIG after relu (since BIG + t/31>0),
    #      so ch1/ch3 hold t/31 + BIG.  We use them only as carriers.
    steps = np.arange(32, dtype=np.float32) / 31.0
    dcoef = np.zeros(32, dtype=np.float64)
    prev = 0.0
    for j in range(32):
        dcoef[j] = bias_t[j] - prev
        prev = bias_t[j]

    # rebuild conv1 with the extra step channels (need <= 32 channels total)
    # channels: 0:x, 1:t, 2:x, 3:t, 4..4+nk-1: local mean hinges,
    #           4+nk .. 4+nk+31: t steps  -> total 4+nk+32 = 4+7+32 = 43 > 32!
    #   => reduce: we do not need ch2, ch3 (duplicate x and t) -> drop them.
    #      channels: 0:x, 1:t, 2..2+nk-1: mean hinges, 2+nk..2+nk+31: steps
    #      total = 2 + nk + 32 = 41 > 32.  Still too many.
    #   => reduce the number of timesteps represented by steps: the bias b(t)
    #      is smooth in t; represent it with a *linear* function of t plus a
    #      modest number of hinge steps (e.g. 8 knots) -> 2 + nk + 8 = 17 <= 32.
    #      We fit b(t) directly as a piecewise-linear function of t with 8
    #      breakpoints using the same hinge mechanism on the t channel.
    nkt = 8
    tknots = np.linspace(0.0, 1.0, nkt + 1)[:-1].astype(np.float32)  # 0..~0.857
    # represent b(t) ~ sum_j e_j * relu(t/31 - tknot_j) + e_lin*(t/31)
    # fit e_j by least squares on the 32 values of bias_t
    Tb = np.arange(32, dtype=np.float64) / 31.0
    Ht = np.maximum(Tb[:, None] - tknots.reshape(1, nkt), 0.0)
    Dt = np.concatenate([Ht, Tb[:, None]], axis=1)
    e_t = np.linalg.lstsq(Dt, bias_t, rcond=None)[0]

    nch = 2 + nk + nkt + 1
    assert nch <= 32, nch
    W1 = np.zeros((32, 2, 3, 3), dtype=np.float32)
    b1 = np.zeros(32, dtype=np.float32)
    W1[0, 0, 1, 1] = 1.0
    b1[0] = BIG
    W1[1, 1, 1, 1] = 1.0
    b1[1] = BIG
    for j in range(nk):
        W1[2 + j, 0] = box
        b1[2 + j] = BIG - float(knots[j])
    for j in range(nkt):
        W1[2 + nk + j, 1, 1, 1] = 1.0
        b1[2 + nk + j] = BIG - float(tknots[j])
    # linear t carrier (t/31) with zero offset so that relu keeps it exact:
    W1[2 + nk + nkt, 1, 1, 1] = 1.0
    b1[2 + nk + nkt] = 0.0        # relu(t/31) = t/31 (t>=0)
    # conv2 / conv3 copies
    W2 = np.zeros((64, 32, 3, 3), dtype=np.float32)
    b2 = np.zeros(64, dtype=np.float32)
    for c in range(nch):
        W2[c, c, 1, 1] = 1.0
    W3 = np.zeros((64, 64, 3, 3), dtype=np.float32)
    b3 = np.zeros(64, dtype=np.float32)
    for c in range(nch):
        W3[c, c, 1, 1] = 1.0

    # out layer
    Wo = np.zeros((1, 64, 3, 3), dtype=np.float32)
    bo = np.zeros(1, dtype=np.float32)
    Wo[0, 0] = F9.astype(np.float32)          # spatial filter on x
    for j in range(nk):
        Wo[0, 2 + j, 1, 1] = float(hcoef[j])
    # t-dependent bias: hinge channels give relu(t/31 - tknot) (no BIG offset
    # because bias = -tknot and value t/31 >= 0 -> relu(t/31 - tknot) exactly)
    for j in range(nkt):
        Wo[0, 2 + nk + j, 1, 1] = float(e_t[j])
    Wo[0, 2 + nk + nkt, 1, 1] = float(e_t[nkt])
    bo[0] = 0.0

    # --------------------------------------------------- verify with a forward
    def forward(xin, tin, W1_, b1_, W2_, b2_, W3_, b3_, Wo_, bo_):
        Nn = xin.shape[0]
        tb = (np.asarray(tin, dtype=np.float32).reshape(Nn, 1, 1, 1) / 31.0)
        inp = np.concatenate([xin, np.broadcast_to(tb, (Nn, 1, H, Wd))], axis=1)
        h = np.maximum(conv2d(inp, W1_, b1_), 0.0)
        h = np.maximum(conv2d(h, W2_, b2_), 0.0)
        h = np.maximum(conv2d(h, W3_, b3_), 0.0)
        o = conv2d(h, Wo_, bo_)
        return o

    pred = forward(x, t, W1, b1, W2, b2, W3, b3, Wo, bo)
    mse_new = float(np.mean((pred - y) ** 2))
    mse_old = float(np.mean((np.asarray(f['prediction'], dtype=np.float32) - y) ** 2))

    # ------------------------------------------------------------- finalise
    out = {}
    out['conv1.weight'] = W1
    out['conv1.bias'] = b1
    out['conv2.weight'] = W2
    out['conv2.bias'] = b2
    out['conv3.weight'] = W3
    out['conv3.bias'] = b3
    out['out.weight'] = Wo
    out['out.bias'] = bo

    # keep the original key set / shapes, clip to the allowed range
    for k in list(out.keys()):
        out[k] = np.clip(np.asarray(out[k], dtype=np.float32), -1000.0, 1000.0)
        assert out[k].shape == np.asarray(w[k]).shape, (k, out[k].shape)
        assert np.all(np.isfinite(out[k]))

    diagnostics = {
        'mse_before': round(mse_old, 6),
        'mse_after': round(mse_new, 6),
        'mse_pred_zero': round(float(np.mean(y ** 2)), 6),
        'n_channels_used': int(nch),
        'F9_absmax': round(float(np.abs(F9).max()), 6),
        'bias_t_absmax': round(float(np.abs(bias_t).max()), 6),
        'hinge_coef_absmax': round(float(np.abs(hcoef).max()), 6),
    }
    return {'weights': out, 'diagnostics': diagnostics}
