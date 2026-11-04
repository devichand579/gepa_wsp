import numpy as np

def edit(w, f):
    # ---------------- data ----------------
    x = np.asarray(f["x"], dtype=np.float64)            # [N,1,16,16]
    t = np.asarray(f["t"]).astype(np.int64).ravel()       # [N]
    y = np.asarray(f["y"], dtype=np.float64)              # [N,1,16,16]
    clean = np.asarray(f["clean"], dtype=np.float64)
    ab = np.asarray(f["alphabar"], dtype=np.float64).ravel()
    N = x.shape[0]
    H = W_ = x.shape[2]
    P = H * W_
    X = x.reshape(N, P)
    Y = y.reshape(N, P)

    # ---------------- noise-model identification ----------------
    # x = sa*clean + sb*eps  ->  find (sa,sb) per timestep by least squares on (clean, y)
    sa = np.ones(32); sb = np.ones(32)
    for k in range(32):
        m = t == k
        if m.sum() < 2:
            continue
        A = np.stack([clean[m].reshape(m.sum(), P), Y[m].reshape(m.sum(), P)], axis=2)
        b = X[m]
        try:
            sol, *_ = np.linalg.lstsq(A.reshape(-1, 2), b.reshape(-1), rcond=None)
            sa[k], sb[k] = float(sol[0]), float(sol[1])
        except Exception:
            pass
    # blend with the schedule alphabar (robust): use alphabar as the primary prior
    ab_ok = np.isfinite(ab) & (ab > 0) & (ab < 1)
    if ab_ok.sum() >= 8:
        # monotone-ish fit: keep alphabar, but correct with identified ratios if they disagree wildly
        sa_sched = np.sqrt(np.clip(ab, 1e-8, 1 - 1e-8))
        sb_sched = np.sqrt(np.clip(1 - ab, 1e-8, 1 - 1e-8))
        # identify ratio r_k = sa/sb from data, compare to schedule
        r_data = np.where(sb > 1e-6, sa / np.maximum(sb, 1e-6), np.nan)
        r_sched = sa_sched / sb_sched
        # use schedule for the ratio but keep data-driven scale sanity check
        sa_use, sb_use = sa_sched.copy(), sb_sched.copy()
    else:
        sa_use, sb_use = sa.copy(), sb.copy()

    # ---------------- closed-form optimal linear predictor ----------------
    # model: eps_hat(p) = sum_q F(q) x_t(p+q) + g_t * 1  (F shared, g_t per timestep)
    # Solve ridge least squares with per-example features: [patches of x_t (25 taps), 1]
    # and per-example, per-timestep constant handled by fitting g_t afterwards.
    # Step 1: per-timestep optimal scalar predictor to get target gain/bias behaviour.
    # Step 2: joint fit of shared spatial filter F (25) + per-timestep gain g_t (32) + bias h_t (32)
    #         via alternating closed-form updates (linear in each block).

    # build patch features for a 5x5 neighborhood (radius 2) using shifted copies
    def shift_stack(A, rad):
        # A [N,P]; returns [N, P, (2rad+1)^2] of shifted copies (zero padding)
        Nn, Pn = A.shape
        out = np.zeros((Nn, Pn, (2 * rad + 1) ** 2), dtype=np.float64)
        idx = 0
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                out[:, :, idx] = shift_img(A.reshape(Nn, H, W_), dr, dc).reshape(Nn, Pn)
                idx += 1
        return out

    def shift_img(A, dr, dc):
        # A [N,H,W] ; shift by (dr,dc) with zero padding
        Nn = A.shape[0]
        B = np.zeros_like(A)
        src_r0, src_r1 = max(0, -dr), min(H, H - dr)
        src_c0, src_c1 = max(0, -dc), min(W_, W_ - dc)
        dst_r0, dst_r1 = max(0, dr), min(H, H + dr)
        dst_c0, dst_c1 = max(0, dc), min(W_, W_ + dc)
        if src_r0 < src_r1 and src_c0 < src_c1:
            B[:, dst_r0:dst_r1, dst_c0:dst_c1] = A[:, src_r0:src_r1, src_c0:src_c1]
        return B

    RAD = 2
    K = (2 * RAD + 1) ** 2
    # accumulate normal equations in chunks over examples
    # unknowns: F (K), g_t (32), h_t (32)
    nF = K
    # design per example i: row per pixel p
    #   features: patch(p) (K), plus one-hot of t (32) for g, plus one-hot of t (32) for h
    # Use block normal equations with structure.
    # Gram: [K, 32g, 32h]
    G = np.zeros((nF + 64, nF + 64))
    b = np.zeros(nF + 64)
    # also per-timestep counts for regularization
    chunk = 100
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        M = e - s
        Xc = X[s:e]
        Yc = Y[s:e]
        tc = t[s:e]
        # patches: [M, P, K]
        Pc = shift_stack(Xc, RAD)                       # M,P,K
        # design matrix rows: M*P
        D = np.zeros((M * P, nF + 64))
        D[:, :nF] = Pc.reshape(M * P, K)
        for j in range(M):
            blk = slice(j * P, (j + 1) * P)
            D[blk, nF + tc[j]] = 1.0
            D[blk, nF + 32 + tc[j]] = 1.0
        G += D.T @ D
        b += D.T @ Yc.reshape(-1)
    # ridge
    lam = 1e-3 * (np.trace(G) / G.shape[0])
    G += lam * np.eye(G.shape[0])
    try:
        sol = np.linalg.solve(G, b)
    except Exception:
        sol = np.linalg.lstsq(G, b, rcond=None)[0]
    F = sol[:nF].reshape(2 * RAD + 1, 2 * RAD + 1)
    g = sol[nF:nF + 32]
    h = sol[nF + 32:nF + 64]

    # ---------------- realize inside the fixed conv stack ----------------
    # conv1: 32 channels, each = ReLU( x_t(center) + M_c + T_c*tau ), tau = t/31
    # conv2: 64 channels, ch o = ReLU( (F3x3_o . x_t) + const_o(t) )  -> we only need a few
    # conv3: identity pass-through of selected conv2 channels
    # out  : center-tap mix of conv3 channels + constant channel for t-dependent bias
    #
    # Simplify: use conv1 to carry x_t(center) and a tau-offset; conv2 to apply the 3x3
    # filter (center 3x3 of the 5x5 F) and produce a constant channel carrying tau;
    # conv3 identity; out mixes.

    # --- 3x3 filter for conv2: take the central 3x3 block of F (and put the rest of F
    #     into conv1's own 3x3 kernel so the composition conv1->conv2 yields a 5x5 F).
    # conv1 kernel K1 (3x3) applied to x_t, then conv2 kernel K2 (3x3): effective 5x5 = F.
    # Choose K1 = center 3x3 of F, K2 = delta (center tap) -> effective = F_center3x3 only.
    # Instead: put full F into conv2 by using conv1 as identity (center tap) and conv2
    # as 3x3 -> effective 3x3. To realize a 5x5 F we need both layers non-trivial.
    # Use: conv1 kernel = F shifted so that its 3x3 covers F rows 0..2, cols 0..2 etc.
    # Simplest exact factorization of a 5x5 into two 3x3 is not possible in general;
    # therefore refit with a 3x3-only shared filter (K2=3x3) which is exactly realizable.
    # -> refit F as 3x3 (radius 1) using the same machinery.
    RAD2 = 1
    K2n = 9
    G2 = np.zeros((K2n + 64, K2n + 64))
    b2 = np.zeros(K2n + 64)
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        M = e - s
        Xc = X[s:e]; Yc = Y[s:e]; tc = t[s:e]
        Pc = shift_stack(Xc, RAD2)
        D = np.zeros((M * P, K2n + 64))
        D[:, :K2n] = Pc.reshape(M * P, K2n)
        for j in range(M):
            blk = slice(j * P, (j + 1) * P)
            D[blk, K2n + tc[j]] = 1.0
            D[blk, K2n + 32 + tc[j]] = 1.0
        G2 += D.T @ D
        b2 += D2Y(Yc) if False else (D.T @ Yc.reshape(-1))
    lam2 = 1e-3 * (np.trace(G2) / G2.shape[0])
    G2 += lam2 * np.eye(G2.shape[0])
    try:
        sol2 = np.linalg.solve(G2, b2)
    except Exception:
        sol2 = np.linalg.lstsq(G2, b2, rcond=None)[0]
    F3 = sol2[:K2n].reshape(3, 3)
    g3 = sol2[K2n:K2n + 32]
    h3 = sol2[K2n + 32:K2n + 64]

    # --- build weights ---
    W1 = np.zeros((32, 2, 3, 3)); B1 = np.zeros(32)
    W2 = np.zeros((64, 32, 3, 3)); B2 = np.zeros(64)
    W3 = np.zeros((64, 64, 3, 3)); B3 = np.zeros(64)
    W4 = np.zeros((1, 64, 3, 3)); B4 = np.zeros(1)

    # conv1 channels 0..31: identity on x_t center + big positive offset M + tau term
    # channel c: pre = x_t(p) + M + Tc*tau   -> ReLU linear (M large)
    M = 50.0
    for c in range(32):
        W1[c, 0, 1, 1] = 1.0
        W1[c, 1, 1, 1] = 0.0
        B1[c] = M
    # conv2: channel o in 0..8 gets filter F3 applied to conv1 channel 0 (center tap of conv1
    # output is x_t + M).  conv2 kernel: W2[o,0,:,:] = F3 ; other input channels 0.
    # conv2 pre-act(o) = sum_{k} F3[k] * (x_t(p+k) + M) + B2[o]
    #                = (F3 . x_t)(p) + M*sum(F3) + B2[o]
    # keep positive: B2[o] = 20 - M*sum(F3)  -> conv2_out = 20 + (F3.x_t)(p)
    # channel 9: constant tau channel: W2[9, :, :, :] = 0, B2[9] = 1.0 -> constant 1
    # channel 10: tau channel: use conv1 channel 1 which carries Tc*tau:
    #   conv1 channel 1: pre = x_t + M + Tc*tau  -> we want a channel that is pure tau:
    #   set conv1 channel 1 to have W1[1,0]=0, W1[1,1,1,1]=1, B1[1]=M  -> out = M + tau
    #   then conv2 channel 10: W2[10,1,1,1] = 1 -> pre = M + tau + B2[10]; set B2[10]=20-M -> 20+tau
    # conv3: identity pass-through: W3[o,o,1,1]=1 for o in 0..10, B3[o]=20 (keep positive)
    # out: mix conv3 channels 0..8 (the filtered x_t) with weights a_o = F3 flattened? No:
    #   out(p) = sum_o a_o * conv3_out_o(p) + b
    #   conv3_out_o(p) = 20 + (F3 . x_t)(p) for o in 0..8  (all identical!)
    #   -> we want out = (F3 . x_t)(p) + g_t + h_t
    #   so a_o = 1 for exactly one o (say o=0) and 0 for others, plus constant channel 9 (value 20)
    #   and tau channel 10 (value 20+tau).
    #   out(p) = 1*(20 + (F3.x_t)(p)) + a9*(20) + a10*(20+tau) + b
    #   We need the t-dependent part to be g_t (per-timestep gain on the *constant*, i.e. h_t)
    #   and the x_t part scaled by g_t as well!  Problem: gain must multiply x_t.
    #   => The per-timestep gain on the x_t term cannot be produced by a linear out layer.
    #   Workaround: use the tau channel to modulate the x_t term through conv2/conv3 nonlinear-free
    #   path is impossible; therefore we use a *per-timestep-free* filter but let the
    #   gain be folded into the target by rescaling: instead of g_t*F3, use F3 and adjust.
    #   Better: exploit that the fitted g_t is (nearly) proportional to a scalar we can
    #   realize: we approximate g_t*F3 by F3 scaled by the *average* gain and absorb the
    #   residual t-dependence in the constant term only.  Instead, we exploit the ReLU
    #   nonlinearity: build conv1 channel c with pre-act = x_t + M + Tc*tau and use
    #   conv2 to compute  sum_c W2[o,c] * conv1_out_c  -> still linear in x_t.

    # ---- REAL SOLUTION: implement t-dependent gain via *scaling in the conv1 t-channel
    #      is impossible linearly; but we can implement the gain by rescaling the target:
    #      fit per-timestep separate filters is impossible too (weights shared).
    #      => Use the fact that the *optimal* predictor can be written as
    #         eps_hat = (x_t - sqrt(a)*mu_hat)/sqrt(1-a)  ... which is gain 1/sqrt(1-a),
    #         i.e. a t-dependent gain.  We approximate it with a *piecewise* mechanism:
    #         two "branches" with different gains selected by ReLU thresholds that depend
    #         on tau via the t-channel offset.  We approximate this by fitting a
    #         *single* filter F3 and per-timestep gain g_t, then realizing the gain by
    #         scaling the conv1 identity channel by a constant and adding a second
    #         branch whose contribution is switched on/off by tau through ReLU.

    # ---- Practical approach: fit per-timestep gains g_t for a shared filter F3 and
    #      realize g_t by *duplicating* the x_t path into several channels with different
    #      fixed gains, and switching between them with tau-dependent ReLU gates.
    #      Gate trick: conv1 channel c has pre-act = Gc * (x_t) + Mc + Tc*tau.
    #      conv2 channel o = sum_c W2[o,c]*conv1_out_c + B2[o] with W2[o,c,1,1]=1 (center).
    #      Then conv2_out_o = Gc*x_t + const.  The *switching* still doesn't multiply by tau.

    #      => Use ReLU gating on the *output* side: conv3 channel o pre-act =
    #          sum_c W3[o,c,1,1]*conv2_out_c + B3[o]. If we want the output to be
    #          g(t)*x_t, we can use the identity ReLU(z) = (z+|z|)/2 and construct |.|
    #          via a channel that is ReLU(-z)...  |z| requires a subtraction of two ReLUs
    #          which needs negative weights downstream (allowed! out layer can have
    #          negative weights).  So: build channels p = ReLU(u) and q = ReLU(-u) where
    #          u = x_t - threshold(tau). Then p - q = u (linear) and p + q = |u|.
    #      Still no multiplication by tau.

    #      Multiplication by a per-example constant can be obtained as follows:
    #      u = x_t + c(tau)  (shift by tau-dependent constant)
    #      p = ReLU(u), q = ReLU(-u)  -> p - q = u, p + q = |u|
    #      Now consider the pair (p, q) and a downstream linear layer: we can form
    #      products only if we had multiplication.  Not available.

    #      CONCLUSION: exact t-dependent gain is not representable.  Use the best
    #      representable model: out = (F . x_t) * gbar + c(tau) with gbar a constant
    #      and c(tau) a per-example constant, and choose (F, gbar, c) to minimize MSE.
    #      We fit: for each timestep, the optimal (g_t, h_t) with shared filter F.
    #      Then we realize g_t by *rescaling the target*: define the network output as
    #      gbar*(F.x_t) + c(tau).  To keep the per-timestep fit, note that the fitted
    #      g_t varies; we approximate g_t by gbar and absorb the residual into c(tau)
    #      only if residual is a constant -- it is not.  So instead: choose F and gbar to
    #      minimize the true MSE directly, and c(tau) per timestep (32 free constants).
    #      This is a well-posed closed-form problem:
    #         minimize sum_i || gbar*(F.x_i) + c_{t_i} - y_i ||^2
    #      with F (9 taps) and c (32) and gbar (scalar).  Solve by alternating:
    #      (a) given gbar, F: c_t = mean over pixels/examples of (y - gbar*F.x)
    #      (b) given F, c: solve for (gbar, F) jointly per timestep? gbar shared.
    #          For fixed c, the model is linear in (gbar, F) jointly -> solve 10 unknowns
    #          per timestep with shared gbar: alternate gbar and F.
    #      Simpler: parametrize gain as a function of tau via the *t-channel* in a way
    #      that multiplies the constant part only, and additionally exploit that the
    #      optimal gain g_t is nearly linear in sqrt(1-a_t):  g_t ~ 1/sqrt(1-a_t).
    #      Then out = (F.x)/sqrt(1-a_t) + c_t.  The 1/sqrt(1-a_t) factor is a per-example
    #      scalar -> NOT representable multiplicatively.  So the best representable
    #      approximation is: out = gbar*(F.x) + c_t.

    # --- Fit (gbar, F, c_t) by closed-form alternating least squares ---
    # initialize F from the joint fit (F3) and gbar from mean of g3
    Fv = F3.reshape(-1).copy()
    gbar = float(np.mean(g3))
    # precompute patch features for 3x3
    Patch = shift_stack(X, RAD2)                       # [N,P,9]
    for it in range(60):
        # update c_t
        resid = Y - gbar * (Patch @ Fv).reshape(N, P)
        cvec = np.zeros(32)
        for k in range(32):
            m = t == k
            if m.sum():
                cvec[k] = resid[m].mean()
        # update (gbar, F) given c_t : model = gbar*(Patch@Fv) + c_{t}
        # design: for each example, rows pixels: [Patch(p) * gbar?] -> joint linear in
        # (gbar, F):  gbar*(Patch@Fv) = sum_j Fv_j * Patch_j * gbar -> bilinear.
        # Alternate: update F with gbar fixed (linear), then gbar with F fixed (linear).
        # (a) F update: for each timestep, regress y - c_t on Patch (9 taps), then average
        #     weighted by counts, with ridge.
        num = np.zeros(9); den = np.zeros((9, 9))
        for k in range(32):
            m = t == k
            if not m.any():
                continue
            Pk = Patch[m].reshape(-1, 9)
            Rk = (Y[m] - cvec[k]).reshape(-1)
            num += Pk.T @ Rk
            den += Pk.T @ Pk
        den += 1e-6 * np.trace(den) / 9 * np.eye(9)
        try:
            Fv = np.linalg.solve(den, num)
        except Exception:
            Fv = np.linalg.lstsq(den, num, rcond=None)[0]
        # (b) gbar update: with F fixed, model = gbar * Z + c_t, Z = Patch@Fv
        Z = (Patch @ Fv).reshape(N, P)
        numg = 0.0; deng = 0.0
        for k in range(32):
            m = t == k
            if not m.any():
                continue
            Zk = Z[m].reshape(-1); Rk = (Y[m] - cvec[k]).reshape(-1)
            numg += float((Zk * Rk).sum()); deng += float((Zk * Zk).sum())
        if deng > 1e-12:
            gbar = numg / deng
        # recompute c_t
        resid = Y - gbar * Z
        for k in range(32):
            m = t == k
            if m.any():
                cvec[k] = resid[m].mean()
    # final c_t
    Z = (Patch @ Fv).reshape(N, P)
    resid = Y - gbar * Z
    cvec = np.zeros(32)
    for k in range(32):
        m = t == k
        if m.any():
            cvec[k] = resid[m].mean()

    # --- realize: out(p) = gbar*(F . x_t)(p) + c_{t}
    # conv1 ch0: identity x_t center + offset M
    # conv2 ch0: kernel F (3x3) applied to conv1 ch0 -> (F.x_t)(p) + M*sum(F)
    # conv3 ch0: identity pass-through
    # out: weight gbar on conv3 ch0 center tap, plus constant channel carrying c_t.
    # constant channel: conv1 ch1 = M + tau (pure tau), conv2 ch1 = M+tau+off -> const+tau
    #   we need c_t which is an arbitrary function of tau -> use 32 channels? conv2 has 64
    #   channels: use 32 channels each carrying a one-hot of tau? tau is continuous in
    #   [0,1] but t is discrete 0..31 -> we can make conv1 channels that are "one-hot"
    #   in tau? conv1's t-channel input is tau = t/31 (a single number), so a channel
    #   can only be affine in tau.  => c_t must be affine in tau to be realizable!
    #   Fit c_t as affine in tau: c_t ≈ c0 + c1*tau.  (residual curvature is small)
    #   Alternatively realize c_t exactly by using the *bias* of the out layer? No,
    #   out.bias is per-example-independent.
    #   => Use affine-in-tau constant channel.  Fit c_t ~ affine(tau) by least squares
    #      (weighted by counts) and use it; keep the per-t c_t for diagnostics.
    tau_all = t / 31.0
    cnt = np.bincount(t, minlength=32).astype(np.float64)
    A = np.stack([np.ones(32), np.arange(32) / 31.0], axis=1)
    Wc = np.linalg.lstsq(A * cnt[:, None], cvec * cnt, rcond=None)[0]
    c_aff = A @ Wc

    # ---- write weights ----
    # conv1
    W1 = np.zeros((32, 2, 3, 3)); B1 = np.zeros(32)
    W1[0, 0, 1, 1] = 1.0; B1[0] = M
    W1[1, 1, 1, 1] = 1.0; B1[1] = M          # channel 1 -> M + tau
    # conv2
    W2 = np.zeros((64, 32, 3, 3)); B2 = np.zeros(64)
    W2[0, 0] = Fv.reshape(3, 3)              # filtered x_t (plus M*sum(F) constant)
    B2[0] = 20.0 - M * float(Fv.sum())
    W2[1, 1, 1, 1] = 1.0
    B2[1] = 20.0 - M                        # -> 20 + tau
    # conv3
    W3 = np.zeros((64, 64, 3, 3)); B3 = np.zeros(64)
    W3[0, 0, 1, 1] = 1.0; B3[0] = 20.0
    W3[1, 1, 1, 1] = 1.0; B3[1] = 20.0
    # out
    W4 = np.zeros((1, 64, 3, 3)); B4 = np.zeros(1)
    W4[0, 0, 1, 1] = gbar
    W4[0, 1, 1, 1] = Wc[1]                  # tau coefficient
    B4[0] = Wc[0] - gbar * 20.0 - Wc[1] * 20.0

    # ---------------- verify with an explicit forward pass ----------------
    def conv2d(xin, Wt, bt):
        # xin [N,C,H,W], Wt [O,C,3,3] cross-correlation, pad 1
        Nn, C, Hh, Ww = xin.shape
        O = Wt.shape[0]
        xp = np.pad(xin, ((0, 0), (0, 0), (1, 1), (1, 1)))
        out = np.zeros((Nn, O, Hh, Ww))
        for i in range(3):
            for j in range(3):
                out += np.einsum('nchw,oc->nohw', xp[:, :, i:i + Hh, j:j + Ww], Wt[:, :, i, j])
        out += bt.reshape(1, -1, 1, 1)
        return out

    def forward(xx, tt, W1_, B1_, W2_, B2_, W3_, B3_, W4_, B4_):
        Nn = xx.shape[0]
        tau = (tt.astype(np.float64) / 31.0).reshape(Nn, 1, 1, 1)
        inp = np.concatenate([xx, np.broadcast_to(tau, (Nn, 1, xx.shape[2], xx.shape[3]))], axis=1)
        h1 = np.maximum(conv2d(inp, W1_, B1_), 0.0)
        h2 = np.maximum(conv2d(h1, W2_, B2_), 0.0)
        h3 = np.maximum(conv2d(h2, W3_, B3_), 0.0)
        o = conv2d(h3, W4_, B4_)
        return o

    pred_new = forward(x, t, W1, B1, W2, B2, W3, B3, W4, B4)
    mse_new = float(np.mean((pred_new - y) ** 2))

    # also compute the achievable (non-representable) reference: per-t optimal scalar fit
    mse_ref = 0.0
    for k in range(32):
        m = t == k
        if not m.any():
            continue
        Xk = X[m]; Yk = Y[m]
        A = np.concatenate([Xk, np.ones((m.sum(), 1))], axis=1)
        sol, *_ = np.linalg.lstsq(A, Yk.reshape(-1), rcond=None)
        mse_ref += float(np.sum((A @ sol - Yk.reshape(-1)) ** 2))
    mse_ref /= float(N * P)

    # ---------------- write back ----------------
    out = {}
    for k, v in w.items():
        if k == "conv1.weight":
            out[k] = W1.astype(np.float32)
        elif k == "conv1.bias":
            out[k] = B1.astype(np.float32)
        elif k == "conv2.weight":
            out[k] = W2.astype(np.float32)
        elif k == "conv2.bias":
            out[k] = B2.astype(np.float32)
        elif k == "conv3.weight":
            out[k] = W3.astype(np.float32)
        elif k == "conv3.bias":
            out[k] = B3.astype(np.float32)
        elif k == "out.weight":
            out[k] = W4.astype(np.float32)
        elif k == "out.bias":
            out[k] = B4.astype(np.float32)
        else:
            out[k] = np.asarray(v, dtype=np.float32).copy()

    # safety: finite, bounded
    for k, v in out.items():
        v = np.nan_to_num(np.asarray(v, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        out[k] = np.clip(v, -1000.0, 1000.0).astype(np.float32)

    diagnostics = {
        "mse_new": round(mse_new, 5),
        "mse_baseline": 0.5147,
        "mse_per_t_optimal_ref": round(mse_ref, 5),
        "gbar": round(float(gbar), 5),
        "filter_taps": [round(float(v), 5) for v in Fv],
        "c_affine": [round(float(v), 5) for v in c_aff],
        "n_used": int(N),
    }
    return {"weights": out, "diagnostics": diagnostics}
