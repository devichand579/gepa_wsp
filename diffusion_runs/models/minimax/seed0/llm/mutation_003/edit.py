import numpy as np

def edit(w, f):
    # ---------------- helpers ----------------
    def conv2d(x, W, b):
        # x: (N,C,H,W), W: (O,C,3,3), pad 1, stride 1, cross-correlation
        N, C, H, Wd = x.shape
        O = W.shape[0]
        xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
        # im2col: (N,H,W,C*9)
        cols = np.empty((N, H, Wd, C * 9), dtype=np.float32)
        for kr in range(3):
            for kc in range(3):
                sl = xp[:, :, kr:kr + H, kc:kc + Wd]
                cols[:, :, :, (kr * 3 + kc) * C:(kr * 3 + kc + 1) * C] = np.transpose(sl, (0, 2, 3, 1))
        Wf = W.reshape(O, C * 9)
        out = np.einsum('nhwk,ok->nhwo', cols, Wf, optimize=True) + b[None, None, None, :]
        return np.transpose(out, (0, 3, 1, 2)).astype(np.float32)

    def relu(x):
        return np.maximum(x, 0.0)

    def forward(x, t, W):
        N = x.shape[0]
        tt = (t.astype(np.float32) / 31.0).reshape(N, 1, 1, 1)
        inp = np.concatenate([x, np.broadcast_to(tt, (N, 1, x.shape[2], x.shape[3]))], axis=1)
        h1 = relu(conv2d(inp, W['conv1.weight'], W['conv1.bias']))
        h2 = relu(conv2d(h1, W['conv2.weight'], W['conv2.bias']))
        h3 = relu(conv2d(h2, W['conv3.weight'], W['conv3.bias']))
        o = conv2d(h3, W['out.weight'], W['out.bias'])
        return h3, o

    def mse(pred, y):
        return float(np.mean((pred - y) ** 2))

    # ---------------- data ----------------
    x = np.asarray(f['x'], dtype=np.float32)
    t = np.asarray(f['t']).astype(np.int64)
    y = np.asarray(f['y'], dtype=np.float32)
    clean = np.asarray(f['clean'], dtype=np.float32)
    abar = np.asarray(f['alphabar'], dtype=np.float32)
    N = x.shape[0]

    # ---------------- diagnostics: optimal per-timestep linear gain ----------------
    # eps = (x_t - sqrt(a) x0)/sqrt(1-a)  ->  E[eps|x_t] ~ c(t) x_t (+ offset)
    s2 = float(np.mean(clean ** 2))            # E[x0^2]
    a_t = abar[t]
    one_m_a = np.maximum(1.0 - a_t, 1e-8)
    c_opt = np.sqrt(one_m_a) / (a_t * s2 + one_m_a)
    # closed-form per-timestep optimal affine predictor (fit on training data)
    gain = np.zeros(32, dtype=np.float64)
    off = np.zeros(32, dtype=np.float64)
    for k in range(32):
        m = (t == k)
        if m.sum() < 3:
            continue
        Xk = x[m].reshape(int(m.sum()), -1).astype(np.float64)
        Yk = y[m].reshape(int(m.sum()), -1).astype(np.float64)
        A = np.concatenate([Xk, np.ones((Xk.shape[0], 1))], axis=1)
        sol, *_ = np.linalg.lstsq(A, Yk, rcond=None)
        gain[k] = sol[0]
        off[k] = sol[1]
    gain = np.nan_to_num(gain, nan=0.0, posinf=0.0, neginf=0.0)
    off = np.nan_to_num(off, nan=0.0, posinf=0.0, neginf=0.0)
    # analytic fallback (smooth in t) for empty timesteps
    ga = np.sqrt(one_m_a) / (a_t * s2 + one_m_a)
    gain = np.where(np.abs(gain) > 0, gain, ga)
    off = np.where(np.abs(off) > 0, off, 0.0)

    # ---------------- build the gated identity network ----------------
    W = {k: np.array(v, dtype=np.float32) for k, v in w.items()}
    Nch = 32
    U = 3.0        # bound on |x_t| used for the clipping window
    Wc = 1.0       # window width (>= 2U+1)

    # ---- conv1: channel 0 = x_t (delta), channel 1 = t/31 (delta) ----
    c1w = np.zeros((Nch, 2, 3, 3), dtype=np.float32)
    c1b = np.zeros((Nch,), dtype=np.float32)
    # channel 0: pre = U + x_t  -> always positive (ReLU ~ identity)
    c1w[0, 0, 1, 1] = 1.0
    c1b[0] = U
    # channel 1: pre = tau (t/31)  -> used to build the gate
    c1w[1, 1, 1, 1] = 1.0
    c1b[1] = 0.0
    # channel 2: pre = -tau + 1  (i.e. 1 - tau)
    c1w[2, 1, 1, 1] = -1.0
    c1b[2] = 1.0
    # channel 3: pre = tau - 0.5  (step-ish helper)
    c1w[3, 1, 1, 1] = 1.0
    c1b[3] = -0.5
    # channel 4: pre = -tau + 0.5
    c1w[4, 1, 1, 1] = -1.0
    c1b[4] = 0.5
    # channels 5..: constant positive carriers (pre = 1)
    for j in range(5, Nch):
        c1b[j] = 1.0
    W['conv1.weight'] = c1w
    W['conv1.bias'] = c1b

    # ---- conv2: build gate channels g_j = clipped ramp of x_t with t-dependent shift ----
    # g_j = ReLU(u + A_j) - ReLU(u + A_j - Wc)  where A_j = shift depending on tau
    # We realize ReLU(u + A) as conv2 output (pre-activation) with bias A, then ReLU.
    # conv2 out channels:
    #   2*j   : pre = u + A_j        (from conv1 ch0 via delta)
    #   2*j+1 : pre = u + A_j - Wc
    # For the gain we need a step in tau. Use a ReLU(tau - c) style step:
    #   step_j(tau) = ReLU(tau - c) * scale  ->  in conv2 we can add conv1 ch3/ch4 (tau-0.5, 0.5-tau)
    # Simplify: use a single global gain ramp implemented with 2 tau-dependent shifts.
    # A_j(tau) = base_j + BIG * ReLU(tau - c_j)  -> implemented as:
    #    conv2 pre = u + base_j + BIG * (tau - c_j)   for tau>c_j, else u + base_j
    # BIG*ReLU(tau-c_j) is not linear; instead use: BIG*(tau - c_j) when tau>c_j.
    # We approximate the step with a large slope: BIG*(tau-c_j) is negative for tau<c_j and
    # positive for tau>c_j, but not zero. To emulate ReLU(tau-c_j) exactly we need a ReLU on
    # a tau-only quantity: conv1 ch3 = tau-0.5 (post-ReLU gives max(tau-0.5,0)) -> that IS a
    # ReLU of a tau-only quantity, i.e. an exact step ramp in tau.
    # So: conv1 ch3 = ReLU(tau - 0.5)  (exact, since tau>=0 and pre = tau-0.5)
    #     conv1 ch4 = ReLU(0.5 - tau)  (exact)
    # Then conv2 can form: A_j(tau) = base_j + S * (ch3 or ch4) with S large.
    #
    # Gate construction (per-pixel, u = x_t):
    #   We want out = g(t) * u with g(t) = piecewise-constant in tau, using:
    #     h1 = ReLU(u + A), h2 = ReLU(u + A - Wc)  -> clipped ramp of u
    #   and then multiply by gate?  Multiplication is not available.
    # Instead: use the *difference* of two clipped ramps with different shifts to obtain
    # a linear-in-u function whose slope is 1 in a window and 0 outside; combined with
    # a t-dependent window position this yields a t-dependent gain on u.
    # Concretely we build, for each of K gain levels, two channels:
    #     P_k = ReLU(u + A_k) - ReLU(u + A_k - Wc)   (ramp, slope 1 in window)
    # and the final out layer sums P_k with weights so that the total is g(t)*u.
    # To keep it simple and robust we instead use the *shifted ramp* trick:
    #     R(A) = ReLU(u + A) - ReLU(u + A - Wc)
    #   R(A)(u) = u + A      for u + A in [0, Wc]
    #            = 0          for u + A < 0
    #            = Wc         for u + A > Wc
    # With Wc > 2U the "u+A<0" and "u+A>Wc" cases can be avoided for a chosen A range.
    # We use A(tau) = -U + 2U*step(tau) so that:
    #   for tau below threshold: A = -U  -> u+A in [-2U,0] -> ramp = u - U (slope 1, but offset)
    #   for tau above threshold: A = +U  -> u+A in [0,2U] -> ramp = u + U
    # Difference of the two regimes is a constant offset, not a gain.  => use Wc trick:
    #   R with Wc = 2U+1 and A = -U-1 gives u+A in [-2U-1,-1] -> always 0  (gate OFF)
    #   R with A = +U+1 gives u+A in [1, 2U+1] -> always u+A (gate ON, slope 1)
    # So define per gain level k two channels:
    #     ON : pre = u + (U+1)          -> ramp = u + U + 1
    #     OFF: pre = u - (U+1)          -> ramp = 0
    # and the t-dependent switch between them is realised by making the *offset* itself
    # t-dependent through the exact tau-only ReLUs (ch3, ch4).
    # ------------------------------------------------------------------
    # Practical construction: use K=32 gain levels, one per timestep, and let the
    # out layer select the right one via a t-dependent scalar (built from ch3/ch4).
    # Because the out layer is linear, we implement the selection as:
    #   out = sum_k  w_k * P_k  with w_k(t) = alpha_k * (t/31) + beta_k  (affine in tau)
    # which is not a hard selection.  Instead we exploit the exact tau-only ReLUs to
    # build, inside conv3, per-level gate channels that are exactly 0/1 in tau:
    #   G_k(tau) = ReLU(tau - c_k) - ReLU(tau - c_{k+1})   (exact, from conv1 ch3-like channels)
    # To obtain those inside conv2 we need conv1 to expose ReLU(tau - c) for many c.
    # conv1 has 32 channels -> we can expose 30 distinct tau-only ReLU thresholds.
    # ------------------------------------------------------------------
    # So: conv1 channels 2..31 hold ReLU(tau - c_j) for thresholds c_j = j/31 (j=0..29),
    # each with weight 1 on the t-channel and bias -c_j (pre-activation = tau - c_j).
    # Then conv2 can build G_k = ch[j_k] - ch[j_{k+1}] exactly (both are ReLU(tau-c)).
    # And conv2 also builds the ON/OFF ramps of u.
    # conv3 combines: P_k * G_k  -> still a product!  Products are impossible.
    # ------------------------------------------------------------------
    # FINAL SIMPLE DESIGN (no products needed):
    #   out = sum_k alpha_k(tau) * u  where alpha_k(tau) is affine in tau is impossible
    #   => use the *exact* structure:  out = G(tau) * u  is a product of a tau-only
    #   function and u.  Implement products via the "clipped ramp" identity:
    #       G*u = ReLU(u + A) - ReLU(u + A - Wc)   with A chosen so that the ramp
    #       equals G*u for the *specific* G.
    #   Trick: if G in {0,1}, choose A = -U-1 (G=0 -> ramp 0) or A = U+1 (G=1 -> ramp u+U+1).
    #   A must be a tau-only affine function that switches between -U-1 and U+1.
    #   A(tau) = -U-1 + (2U+2) * ReLU(tau - c) works only if ReLU(tau-c) is 0/1, but it is
    #   continuous.  However, we can *saturate* it: use a second ReLU on a large multiple:
    #       S(tau) = ReLU(K*(tau - c) - K/2) / K   -> not exactly 0/1 either.
    #   Instead: build the exact step with two ReLUs of tau-only quantities:
    #       step(tau) = [ReLU(tau - c) - ReLU(tau - c - d)] / d  with d small -> approx step.
    #   Approximation is acceptable (MSE impact tiny) but let's avoid it.
    #
    # SIMPLEST EXACT SOLUTION: exploit that we can also *choose* the gain function freely.
    # We need out = g(t) u. Let us use the identity:
    #     g(t) u  =  ReLU(u + A(t)) - ReLU(u + A(t) - B(t))  is not it either.
    # Use instead:  g(t) u = u * g(t).  Write g(t) = sum_k g_k * 1[tau in I_k].
    # The product 1[tau in I_k] * u can be produced EXACTLY as:
    #     ReLU(u + A) - ReLU(u + A - Wc) with A = -U-1 + (2U+2)*step_k(tau),
    # where step_k(tau) must be exactly 0 or 1.
    # Exact 0/1 step from tau-only ReLUs:
    #     step(tau) = ReLU(tau - c) - ReLU(tau - c - eps)  is a ramp of width eps.
    #     To get exact 0/1 we need a ReLU whose argument is a *constant* when in range.
    #     We can get that: conv1 channel with weight 1 on t-channel and bias -c gives tau-c;
    #     conv2 can multiply by a large constant K: pre = K*(tau-c) + b.  Then ReLU of that
    #     is 0 for tau<c and K*(tau-c)+b for tau>c.  If we then subtract a linear ramp
    #     K*(tau-c)+b (available as a linear combination of tau-only channels) we get a
    #     function that is 0 for tau<c and b for tau>c -> EXACT step (value b), because the
    #     linear ramp is available exactly.
    #   So: step_c(tau) = ReLU(K*(tau-c) + b) - (K*(tau-c) + b) + ... hmm we need the linear
    #   part K*(tau-c)+b to be representable as a linear combination of tau-only channels
    #   (it is: K*tau + (b - K c) -> affine in tau, and conv1 ch1 gives tau exactly).
    #   Therefore:
    #       step_c(tau) = ReLU(K*(tau-c) + b) - K*tau - (b - K*c)   -> equals 0 if tau<c,
    #       and equals b - b = 0 ... let's recompute: for tau>c: ReLU = K*tau - K*c + b,
    #       minus (K*tau + b - K*c) = 0.  For tau<c: ReLU=0, minus (K*tau+b-K*c) = K*c - K*tau - b.
    #   Not a step.  Flip sign: define
    #       step_c(tau) = (K*tau + b - K*c) - ReLU(K*(tau-c)+b)  -> for tau>c: 0; for tau<c:
    #       K*tau + b - K*c  (negative ramp).  So -step_c(tau) = ReLU(...) - (K*tau+b-K*c)
    #       equals 0 for tau>c and (K*c - K*tau - b) for tau<c.
    #   Still a ramp.  The issue: ReLU of a ramp is piecewise linear; subtracting the ramp
    #   gives a piecewise linear function that is 0 on one side and a ramp on the other.
    #   To get a constant on one side we need the ramp to be constant there, i.e. use a
    #   *constant* input to the ReLU, not a ramp.  A constant that depends on tau only via
    #   a step is exactly what we lack.
    #
    # => Products are genuinely impossible in this architecture without approximation.
    #    BUT: we can approximate the step well using a steep ramp, and the resulting
    #    error in the gain is O(eps) -> negligible MSE impact.
    #
    # Practical, accurate construction:
    #   Let d = 1/32 (one timestep).  Define for each level k a "soft step"
    #       S_k(tau) = clip((tau - c_k)/d, 0, 1)  approximated by
    #       S_k(tau) = [ReLU(tau - c_k) - ReLU(tau - c_k - d)] / d
    #   which is exactly 0 for tau<=c_k, exactly 1 for tau>=c_k+d, linear in between.
    #   With c_k = k/31 and d = 1/31, S_k(tau) = 1 exactly when tau >= (k+1)/31,
    #   i.e. when t >= k+1, and 0 when t <= k.  Since tau = t/31 is an integer multiple
    #   of 1/31, S_k is EXACTLY 0/1 at all training timesteps!
    #   Great: S_k(tau) = ReLU(tau - k/31) - ReLU(tau - (k+1)/31) is exactly 1 if
    #   t >= k+1 and 0 if t <= k.
    #
    #   Now the gain: we want out = g(t) * u.  Use the clipped-ramp identity with a
    #   t-dependent offset built from S_k:
    #       A(tau) = -U-1 + (2U+2) * S(tau)   where S(tau) = step at the current t.
    #   Since S is a sum of per-level steps weighted by g_k, we can build
    #       A(tau) = sum_k a_k S_k(tau)   (affine combination of exact steps)
    #   and then
    #       ramp(u, A) = ReLU(u + A) - ReLU(u + A - Wc)
    #   gives, for the level k in effect, ramp = u + A_k (since u + A_k in [1, 2U+1] when
    #   A_k = U+1, and ramp = 0 when A_k = -U-1).
    #   => out = sum_k g_k * ramp_k where ramp_k uses A_k = -U-1 for "off" levels and
    #      A_k = U+1 for the "on" level.  But A depends on which level is on -> product
    #      again (A_k * S_k).
    #   Resolve: A(tau) = -U-1 + (2U+2)*S(tau) is a *single* tau-only function (not per
    #   level): S(tau) = 1 for tau >= c*, 0 otherwise, where c* is a single threshold.
    #   Then ramp(u, A(tau)) = 0 when S=0 and u+U+1 when S=1 -> out = u + U + 1 for
    #   the "on" region and 0 otherwise.  This gives a single threshold, not a per-t gain.
    #   For a per-t gain we need many thresholds, each with its own ramp -> products.
    #
    #   => Products ARE needed for exact per-t gain.  Approximate with soft steps:
    #   Use S_k(tau) soft (continuous ramp of width d) and multiply by ramp_k:
    #       out = sum_k g_k * S_k(tau) * u.
    #   Still a product of two tau-only functions and u.
    #
    #   Alternative that avoids products entirely: use the *linear* structure with
    #   per-timestep gain implemented by a *single* affine function of tau times u is
    #   impossible, so instead approximate g(t) by a piecewise-linear function of tau
    #   and note that a piecewise-linear function of tau can be implemented as a sum of
    #   ReLU(tau - c) terms (exact, tau-only). Multiplying by u is still a product.
    #
    #   FINAL: we accept an approximation using the *shifted ramp* trick with a
    #   tau-dependent shift A(tau) = alpha + beta*tau (affine).  Then
    #       out = ReLU(u + A(tau)) - ReLU(u + A(tau) - Wc)
    #   is piecewise linear in u with slope 1 in a window whose position depends on tau.
    #   This is NOT a pure gain.
    #
    #   ==> Different, clean, EXACT approach: change the *input scaling* trick.
    #   The gain can be implemented exactly if we let the network output
    #       out = sum_k g_k * u_k   with u_k = u (same signal) and the *coefficient*
    #   g_k applied to the SAME channel but selected by t.  Selection of a coefficient
    #   by t is: g(t) = sum_k g_k S_k(t) -> a tau-only function.  Then out = g(t)*u
    #   requires multiplying a tau-only function by u.
    #   In a *linear* network this is impossible; with ReLU it is possible via
    #   multiplication-by-constant tricks only.
    #   BUT we can multiply a tau-only function by u using the following exact identity
    #   (ReLU "hinge" trick):
    #       u * s  =  ReLU(u + M) - ReLU(u + M - s*M') ... no.
    #   Known exact identity: for s >= 0 constant, u*s = ReLU(u*s) - ReLU(-u*s) = ...
    #   but s*u requires s*u as an argument to ReLU -> product.
    #
    #   => Exact per-timestep gain is NOT implementable.  Therefore the achievable
    #   predictor must be of the form: out = sum_j c_j(t) * phi_j(u) where phi_j are
    #   fixed nonlinear functions of u and c_j(t) are affine (or piecewise-linear) in t.
    #   This is enough! Because g(t)*u can be approximated by a piecewise-linear-in-t
    #   combination of *linear* functions of u only if the coefficients were constants.
    #   However, with phi_j(u) = ReLU(u + A_j) - ReLU(u + A_j - Wc) (clipped ramps at
    #   different offsets), the family {phi_j} spans a rich set of piecewise-linear
    #   functions of u, and c_j(t) affine in t.  With enough j we can approximate
    #   g(t)*u well: for each t we need the combination to equal g(t)*u; since the
    #   phi_j are piecewise linear in u with kinks at u = -A_j, a linear combination
    #   with t-dependent coefficients can represent a scaled u only if the kinks
    #   cancel... For a *scaled* u (no kinks), all kink contributions must cancel, which
    #   forces coefficients of the kink-carrying basis to vanish except for a pure
    #   linear-in-u part.  So effectively out = c(t)*u + const(t) is representable
    #   iff we can get a linear-in-u term with t-dependent coefficient.  The linear-in-u
    #   term comes from a channel that is *always* in the linear region of the ReLU ramp,
    #   i.e. ReLU(u + A) with A >= U (always positive) -> gives u + A (affine, no kink).
    #   Its coefficient is set by the out layer (constant) -> cannot be t-dependent.
    #   UNLESS the "always positive" property is t-dependent: ReLU(u + A(t)) is always
    #   positive iff A(t) >= U for all t.  So no.
    #
    #   => We can only get t-dependent gain if the kink region shifts with t, which
    #   changes the function shape in u (not a pure gain).  This is a real limitation.
    #
    #   PRACTICAL COMPROMISE (good MSE, exact, simple):
    #   Use the clipped-ramp family with MANY offsets A_j and let the *out* layer use
    #   t-dependent coefficients c_j(t) that are affine in tau.  Then
    #       out(u,tau) = sum_j c_j(tau) * phi_j(u)
    #   where phi_j(u) = ReLU(u + A_j) - ReLU(u + A_j - Wc) (ramp of width Wc).
    #   Choosing A_j spread over a range and c_j(tau) = p_j + q_j*tau gives a function
    #   that is a piecewise-linear-in-t approximation of the ideal gain.  We can even
    #   solve for the coefficients directly by least squares on the training data!
    #   That is: build a feature bank phi_j(u) (K channels) and tau-only channels,
    #   then fit the final linear layer by closed-form LS to the targets.  The final
    #   layer sees conv3 outputs; we design conv1/conv2/conv3 to expose exactly the
    #   features we want, then LS-fit the out layer.
    #   => This is the plan: DESIGN the hidden layers to expose a rich, exactly
    #   computable feature bank, then solve the final layer in closed form.

    # ---- Feature bank design ----
    # conv1 channels (32):
    #   ch0 : u + U            (u = x_t, always positive -> linear channel)
    #   ch1 : tau              (t/31, exact)
    #   ch2..ch31 : ReLU(tau - c_j) for thresholds c_j = (j-2)/31, j=2..31 (30 channels)
    #   -> these are exact tau-only step ramps (0 or tau-c_j).
    # conv2 channels (64): build the ramp bank on u and the tau-only channels.
    #   For j in 0..K-1 (K=16 offsets):
    #     ch 2j   : pre = u + A_j          (from conv1 ch0 via delta, bias A_j)
    #     ch 2j+1 : pre = u + A_j - Wc
    #   plus 32 channels copying conv1 ch1..ch31 (tau-only) with weight 1 (delta), bias 0
    #   (32 + 32 = 64 exactly)
    # conv3 channels (64): identity-ish copies of conv2 (delta kernels) with bias 0,
    #   but we must ensure ReLU keeps them: conv2 outputs are post-ReLU >= 0, so
    #   conv3 pre-activations = those values (>=0) -> ReLU identity. Good.
    #   conv3: ch 0..31 = delta copies of conv2 ch 0..31 (the ramp bank)
    #           ch 32..63 = delta copies of conv2 ch 32..63 (tau-only channels)
    # out layer: LS fit on these 64 features -> target y.
    #   This is a linear model: out = sum_k a_k * ramp_k(u) + sum_m b_m * tauonly_m(tau)
    #   fitted per-pixel across examples by closed-form ridge LS.  It can represent
    #   g(t)*u to good accuracy because the ramp bank spans piecewise-linear functions
    #   of u and the tau-only channels let the coefficients vary with t in a
    #   *multiplicative* way?  No: still additive.  But the ramp bank with different
    #   offsets lets the model represent, for each t, a piecewise-linear function of u
    #   whose shape may vary with t through the additive tau-only terms only (shift-like).
    #   Hmm, additive tau-only terms cannot change the slope of u.
    #   => The gain g(t) cannot be produced.  We accept a *constant* gain (t-independent)
    #   plus t-dependent offsets, OR we add a second bank of ramps whose offsets are
    #   t-dependent (via the tau-only channels feeding conv2's bias? no, bias is constant).
    #   => To get t-dependent offsets in the ramp bank we need conv2's bias to depend on
    #   tau, impossible (bias is a constant vector).  BUT we can add tau-only contributions
    #   to conv2 pre-activations through the conv1 tau channels!  conv2 pre for ramp
    #   channel j: sum over conv1 channels of weights * conv1_out.  conv1 ch0 = u+U,
    #   conv1 ch1..31 = tau-only ramps.  So conv2 pre_j = w_{j,0}*(u+U) + sum_m w_{j,m}*tauonly_m.
    #   The tau-only part is an arbitrary function of tau (piecewise linear in tau with
    #   30 knots) -> we CAN make the ramp offset t-dependent!
    #   So: conv2 pre_j = (u + U) + A_j(tau) where A_j(tau) is piecewise-linear in tau.
    #   Then ramp_j(u,tau) = ReLU(u+U+A_j(tau)) - ReLU(u+U+A_j(tau)-Wc).
    #   With A_j(tau) chosen so that for each t the "active" ramps align to produce
    #   the desired gain g(t) * u.
    #   Concretely: choose K ramps whose offsets are A_j(tau) = -U-1 + (2U+2)*S_j(tau) where
    #   S_j(tau) = sum of tau-only step functions -> exact 0/1 per t.
    #   Then for a given t, ramp_j = 0 if S_j=0, and = u + U + 1 if S_j=1.
    #   Now out = sum_j c_j(tau) * ramp_j: with c_j(tau) affine in tau we still can't
    #   multiply by the indicator.  BUT we can put the indicator into the *out* layer
    #   coefficients? out layer coefficients are constants.  However, the out layer sees
    #   conv3 features which include BOTH the ramps (with the indicator baked in) AND
    #   the tau-only channels.  The out layer is linear in features, so
    #       out = sum_j a_j ramp_j(u,tau) + sum_m c_m tauonly_m(tau).
    #   ramp_j(u,tau) = S_j(tau) * (u + U + 1)  -> product of tau-only S_j with u.
    #   Still a product!  The product is inherent: any t-dependent gain needs it.
    #
    #   RESOLUTION: the product S_j(tau)*u can be formed if we can compute u * S_j(tau)
    #   where S_j is 0/1.  Trick: u*S_j = ReLU(u + M*S_j) - ReLU(u + M*S_j - Wc) with
    #   M large: when S_j=1 -> ramp = u+M (if u+M in [0,Wc]); when S_j=0 -> ramp=0.
    #   M*S_j is a tau-only quantity -> we can build it as a linear combination of
    #   tau-only channels (they are exactly 0 or their values; M*S_j is exactly 0/1*M).
    #   So: build in conv2 a channel with pre-activation = u + M*S_j(tau)  (u from conv1 ch0
    #   with weight 1, plus M*S_j from tau-only channels with weights M).
    #   Then conv2 output = ReLU(that) = u + M*S_j (if positive; M large enough).
    #   And a second channel with pre = u + M*S_j - Wc -> ReLU = u + M*S_j - Wc (if positive).
    #   Difference = Wc (constant!) ... not what we want.
    #   We want: ramp = ReLU(u + M*S_j) - ReLU(u + M*S_j - Wc) = u + M*S_j when
    #   0 < u + M*S_j < Wc.  With M large (say M = 10) and Wc = 2U+2 = 8:
    #     S_j=1: u+10 in [7,13]; need < Wc=8 -> fails for u> -2.
    #   Choose Wc = 2U + 2M = 20: then u+M in [7,13] ⊂ (0,20) -> ramp = u + M. Good.
    #     S_j=0: u in [-3,3], need u in (0,20) -> fails for u<0 -> ramp = 0 for u<0 but
    #     = u for u>0 (since u + 0 = u in (0,20)).  Hmm, that gives ramp = max(u,0) when off.
    #   Not a clean zero.  But we can subtract the "off" contribution: the off-ramp is
    #   max(u,0) which is a fixed function; we can subtract a copy of it (built with
    #   S=0 channel) times coefficient 1 -> out = ramp_on - ramp_off.
    #   Then out = S_j*(u+M) - max(u,0) ... hmm when S_j=1 we get u+M - max(u,0) which
    #   is not u.
    #   Let's be careful:
    #     R_j = ReLU(u + M*S_j) - ReLU(u + M*S_j - Wc)
    #     with Wc = 2U + 2M:
    #       S_j=1: u+M in [M-U, M+U] ⊂ (0, Wc) -> R_j = u + M.
    #       S_j=0: u in [-U,U]; R_0 = max(u,0) - max(u - Wc, 0) = max(u,0) (since u<Wc).
    #   So R_j = S_j*(u+M) + (1-S_j)*max(u,0).
    #   We want S_j*u.  Note S_j*(u+M) = R_j - (1-S_j)*max(u,0) - S_j*M.
    #   Hmm, (1-S_j)*max(u,0) is again a product.
    #   BUT: we can build max(u,0) exactly (channel with pre = u + U, ReLU -> u+U; and
    #   constant channel U; difference gives max(u,0)).  Then:
    #     S_j*u = R_j - max(u,0) + S_j*(max(u,0) - u) ... no.
    #   Let's just directly compute: we want the function f_j(u) = S_j * u.
    #   Take two channels:
    #     P = ReLU(u + M*S_j)          -> = u + M*S_j   (always positive since M*S_j>=0, u>=-U)
    #     Q = ReLU(u + M*S_j - Wc)     -> = u + M*S_j - Wc  if positive
    #   With Wc = 2U + 2M: when S_j=1, u+M-Wc = u - 2U - M < 0 always -> Q=0.
    #       when S_j=0: u - Wc < 0 always -> Q=0.
    #   So P - Q = u + M*S_j.  Then subtract M*S_j (tau-only, exact) -> u + M*S_j - M*S_j.
    #   Still leaves S_j*u mixed with u.  We need to remove the plain u part when S_j=1.
    #   => S_j*u = (u + M*S_j) - u  when S_j=1  ->  = u + M - u = M.  no.
    #   Let's solve directly: we want S*u.  Consider T = ReLU(u + M*S - M) - ReLU(u + M*S - M - Wc)
    #   with Wc = 2U+2:
    #     S=1: u+M-M = u; u in [-U,U]; u in (0,Wc)? only if u>0 -> T = u for u>0, 0 for u<0.
    #     S=0: u - M; negative -> T=0.
    #   So T = S * max(u,0).  Similarly with -u: T' = S*max(-u,0).
    #   Then S*u = T - T'.  EXACT!  Both T and T' are built from the same S channel.
    #   And max(u,0), max(-u,0) are exact.
    #   So the construction works exactly:
    #     S_j(tau) must be exactly 0 or 1 (a tau-only quantity) -> available as
    #     ReLU(tau - c) - ReLU(tau - c - 1/31) which is exactly 0 or 1/31 at integer t.
    #     Scale: use S = 31*(ReLU(tau-c) - ReLU(tau-c-1/31)) -> exactly 0 or 1.
    #     Then M*S is a tau-only quantity exactly 0 or M.
    #   Great. Now we need per-timestep gain g(t): build, for each t level k, a channel
    #   with S_k(tau) = 1 iff t == k.  S_k = 31*(ReLU(tau - k/31) - ReLU(tau - (k+1)/31)) gives
    #   1 iff t >= k+1... Let's define D_k(tau) = 31*(ReLU(tau - (k-1)/31) - ReLU(tau - k/31)):
    #   for t=k: ReLU((k-(k-1))/31)=1/31, ReLU(0)=0 -> D_k = 1. For t=k-1: 0 - 0 = 0. For t=k+1: 2/31 - 1/31 = 1/31*31 = 1. WRONG: for t=k+1, ReLU((k+1-k+1)/31)=2/31, ReLU((k+1-k)/31)=1/31 -> diff 1/31 -> D=1. Not an indicator.
    #   Correct indicator: I_k(tau) = 31*(ReLU(tau - (k-1)/31) - 2*ReLU(tau - k/31) + ReLU(tau - (k+1)/31))
    #   For t=k: 31*(1/31 - 0 + 0) = 1. For t=k-1: 31*(0-0+0)=0. For t=k+1: 31*(2/31 - 2/31 + 0) = 0. For t=k+2: 31*(3/31 - 2/31 + 1/31) = 31*(2/31)=2. Hmm not zero.
    #   Let's use: I_k(tau) = 31*(ReLU(tau-(k-1)/31) - 2*ReLU(tau-k/31) + ReLU(tau-(k+1)/31))? For t=k+2: 31*(3/31 - 2*(2/31) + 1/31) = 31*(0) = 0. Let me recompute: 3/31 - 4/31 + 1/31 = 0. Yes! I mis-added. Good.
    #   Check t=k+1: 2/31 - 2*(1/31) + 0 = 0. Good. t=k: 1/31 - 0 + 0 = 1/31 -> 1. t=k-1: 0-0+0=0. t=k+2: 0. t=k+3: 4/31 - 2*(3/31) + 2/31 = 0. Good.
    #   So I_k(tau) is exactly the indicator of t==k (for integer t in 0..31), using three
    #   tau-only ReLU ramps. But each I_k needs 3 tau-only channels; we only have 30.
    #   Alternative: build the indicators in conv2 from the 30 tau-only ramps available:
    #   conv1 gives R_j = ReLU(tau - j/31) for j=0..29 (30 channels), j=0 gives tau.
    #   Then I_k = 31*(R_{k-1} - 2 R_k + R_{k+1}) with R_{-1}=0, R_{30}=tau (since ReLU(tau-1)=tau-1... careful).
    #   Actually R_j = ReLU(tau - j/31) for j=0..30 needed; j=30: tau - 30/31.
    #   We can build R_j for j=0..30 using 31 channels (plus ch0 = u+U) -> 32 channels total.
    #   Then conv2 (64 channels) can form I_k for k=0..31 (32 indicators) using 3 taps each
    #   -> but conv2 output channels are limited to 64; we need 32 indicators + ramp channels.
    #   Hmm. Let's restructure:
    #   conv1 (32 ch): ch0 = u + U (linear u channel); ch1..ch31 = R_j = ReLU(tau - j/31), j=0..30.
    #   conv2 (64 ch): 32 channels = I_k (k=0..31) built as 31*(R_{k-1} - 2R_k + R_{k+1});
    #     remaining 32 channels = the ramp-building channels:
    #       for each k we need: P_k = ReLU(u + M*I_k) and Q_k = ReLU(u + M*I_k - Wc)
    #       -> 64 channels needed. Too many.
    #   Instead: note S*max(u,0) and S*max(-u,0) need, per k, two channels. 32 k's -> 64.
    #   Total conv2 channels needed: 32 (I_k) + 64 = 96 > 64.
    #   Reduce: we don't need per-timestep indicators if we accept a *piecewise-linear*
    #   gain in t.  The gain g(t) is smooth; we can approximate it with a piecewise-linear
    #   function of tau with, say, 8 knots.  But the product with u still needs the
    #   indicator structure (S*u) for each knot region... Each region needs its own
    #   S*u channel pair (2 channels) -> 8 regions -> 16 channels + 8 indicator channels
    #   (or build indicators from ramps: 8 indicators, each needing 3 R taps, but conv2 can
    #   combine multiple conv1 channels per output channel, so 8 indicator channels suffice).
    #   conv2: 8 indicator channels + 16 ramp channels = 24, plus we can also copy R_j channels
    #   (31) for the out layer to fit t-dependent offsets: 24+31 = 55 <= 64.
    #   conv3 (64 ch): identity copies of conv2 (delta) -> 55 used, rest zeros.
    #   out: LS fit on 64 features.
    #   But the out layer is LINEAR in features, so the final prediction is
    #       out = sum_k a_k * [S_k*max(u,0) - S_k*max(-u,0)] + sum_m c_m R_m(tau)
    #          = sum_k a_k S_k(tau) * u + sum_m c_m R_m(tau)
    #   The gain is now a *constant* a_k per region (not t-dependent within region) and
    #   the sum over regions with a single 'a' per region.  Since each region has its own
    #   S_k channel, the out layer can weight each region independently -> effectively
    #   out = sum_k a_k S_k(tau) u + ... = g_piecewise(tau) * u.  EXACT piecewise-constant
    #   gain with one value per region!
    #   With 8 regions we get a piecewise-constant approximation of g(t) with 8 levels.
    #   Even better: use 16 regions (16 indicators + 32 ramp channels = 48 <= 64).
    #   Let's use 16 regions (each covering 2 timesteps) -> gain error small.
    #   Actually we can do better: use 32 regions but build the S_k*u product more cheaply.
    #   Alternative cheap trick: S_k*u = S_k*u. Note S_k is a tau-only channel. If we build
    #   in conv2 a channel with pre-activation = u + M*S_k, then conv2 output = u + M*S_k
    #   (positive). Then conv3 can compute a *difference* with a channel that has u - M*S_k...
    #   Hmm, we want the product.
    #   Let's use the max trick but reduce channel count: For each region we need
    #     P_k = ReLU(u + M*I_k) and Q_k = ReLU(u + M*I_k - Wc)  -> 2 channels
    #   Actually we want S*max(u,0) and S*max(-u,0):
    #     S*max(u,0) = ReLU(u + M*S) - ReLU(u + M*S - Wc)   with Wc = 2U + 2M? Let's redo:
    #       S=1: u+M in [M-U, M+U]; choose Wc > M+U so that u+M-Wc < 0 -> Q=0 -> P = u+M.
    #       S=0: u in [-U,U]; P = max(u,0); Q = max(u-Wc,0) = 0 (since u<Wc) -> P-Q = max(u,0).
    #     So R_k = P_k - Q_k = S*(u+M) + (1-S)*max(u,0).
    #     We want S*u. Note S*u = R_k - (1-S)*max(u,0) - S*M.
    #     (1-S)*max(u,0) = max(u,0) - S*max(u,0) -> also a product. Circular.
    #   Use the two-sided trick:
    #     T_k = ReLU(u + M*S) - ReLU(u + M*S - Wc)   -> S=1: u+M ; S=0: max(u,0)
    #     T'_k = ReLU(-u + M*S) - ReLU(-u + M*S - Wc)  -> S=1: -u+M ; S=0: max(-u,0)
    #     T_k - T'_k = S*(2u) + (1-S)*(max(u,0)-max(-u,0)) = 2*S*u + (1-S)*u = u + S*u.
    #     Hmm: S=1: (u+M) - (-u+M) = 2u. S=0: max(u,0)-max(-u,0) = u. So T-T' = 2u when S=1, u when S=0.
    #     => S*u = (T_k - T'_k) - u.  And u is available exactly (from ch0 minus U constant).
    #     So S*u = T_k - T'_k - u.  EXACT!  Requires channels for T_k and T'_k: 4 channels
    #     per region (P,Q,P',Q').  Plus the indicator channel: 5 per region -> 16 regions
    #     = 80 > 64.  12 regions = 60 <= 64.  Hmm.
    #   Simplify: we don't need Q channels if we choose Wc such that Q is always 0:
    #     Q = ReLU(u + M*S - Wc): S=1 -> u+M-Wc <0 for all u if Wc > M+U. S=0 -> u-Wc<0 if Wc>U.
    #     Then T_k = P_k = ReLU(u + M*S) = S*(u+M) + (1-S)*max(u,0). 1 channel per region!
    #     Similarly T'_k = ReLU(-u + M*S) = S*(M-u) + (1-S)*max(-u,0). 1 channel per region.
    #     => 2 channels per region + 1 indicator channel per region = 3 per region.
    #     With 64 channels: 21 regions (63 channels) -> 21 regions over 32 timesteps.
    #     Even better: the indicator I_k can be built from R_j channels; each conv2 output
    #     channel can combine many conv1 channels, so 1 channel per region for the indicator.
    #     21 regions: use 16 regions (each 2 timesteps) = 48 channels, leaving 16 for R_j copies.
    #   Alternatively, note that we can get S*u with only ONE conv2 channel per region if we
    #   use the following: build in conv2 channel: pre = u + M*I_k (1 channel) -> out = u + M*I_k.
    #   and channel: pre = -u + M*I_k -> out = M*I_k - u (if positive) hmm -u+M could be negative
    #   for u > M.  With M large (say M=10) and u in [-3,3], -u+M in [7,13] > 0 always.
    #   So both channels are positive always: A_k = u + M*I_k, B_k = M*I_k - u.
    #   A_k - B_k = 2u (independent of I_k!). Not useful.
    #   A_k = u + M*I_k ; we want I_k*u.
    #   Consider also channel C_k = u (plain, from ch0) and D_k = M*I_k (indicator).
    #   A_k = C_k + D_k.  Products still needed.
    #   OK: 3 channels per region it is: I_k, A_k = u + M*I_k, B_k = -u + M*I_k.
    #   Then in conv3 we can form: A_k - B_k = 2u (useless), but
    #     I_k*u = ?  From A_k and B_k: A_k = u + M I_k, B_k = M I_k - u.
    #     We need u*I_k.  Note (A_k - B_k)/2 = u.  Hmm.
    #   Use the earlier exact relation with max: T_k = ReLU(u + M*I_k) = u + M*I_k (always positive)
    #   and T'_k = ReLU(-u + M*I_k) = M*I_k - u (always positive).
    #   T_k - T'_k = 2u.  T_k + T'_k = 2M*I_k.
    #   So I_k is available (as (T+T')/2M) and u is available, but the product I_k*u is NOT
    #   available from these two alone.
    #   => We need the max-based construction:
    #     T_k = ReLU(u + M*I_k)  -> = u + M I_k  (always pos)
    #     T'_k = ReLU(-u + M*I_k) -> = M I_k - u  (always pos)
    #     Now define U_k = ReLU(u + M*I_k - Wc) with Wc > M + U: always 0.  Useless.
    #     Instead define V_k = ReLU(u + M*I_k) - ReLU(u + M*I_k - Wc) ... same as T_k.
    #   Hmm, the max(u,0) decomposition: we need a channel that gives max(u,0) when I_k=0
    #   and something else when I_k=1.
    #   Let me reconsider: define
    #     P_k = ReLU(u + M*I_k)        = u + M I_k   (pos always)
    #     Q_k = ReLU(u + M*I_k - Wc)   = 0 always (if Wc > M+U)
    #     R_k = ReLU(-u + M*I_k)       = M I_k - u   (pos always)
    #     S_k = ReLU(-u + M*I_k - Wc)  = 0 always
    #   Now consider P_k - R_k = 2u. And P_k + R_k = 2M I_k.
    #   The product I_k*u: consider a channel that outputs ReLU(u + M*I_k) - ReLU(u + M*I_k - Wc)
    #   = P_k - Q_k = u + M I_k.  And a channel with ReLU(u) - ReLU(u - Wc) = max(u,0) (since u<Wc).
    #   So we have max(u,0) and (u + M I_k).  We want I_k*u.
    #   I_k*u = (u + M I_k) - u  when I_k=1 -> M; when I_k=0 -> 0.  no.
    #   Let's use: I_k*u = I_k*(u) . We have I_k (exactly) and u (exactly) but not the product.
    #   The ONLY way to get products is via ReLU nonlinearity with the product inside.
    #   So we MUST feed u + M*I_k into a ReLU in a way that the kink position depends on I_k.
    #   The kink of ReLU(z) is at z=0.  If z = u + M*I_k - Wc, the kink is at u = Wc - M*I_k,
    #   which depends on I_k!  So:
    #     Z_k = ReLU(u + M*I_k - Wc)  = max(0, u + M I_k - Wc)
    #   With Wc chosen so that for I_k=0: u - Wc < 0 for all u (Wc > U) -> Z_k = 0.
    #   With I_k=1: u + M - Wc; choose Wc = M + U + eps? then u + M - Wc = u - U - eps <= 0 for u<=U
    #   -> also 0.  Choose Wc = M - U (so that for I_k=1: u + M - Wc = u + U >= 0 for u>=-U,
    #   equality at u=-U) -> Z_k = u + U (for I_k=1), and for I_k=0: u - M + U < 0 if M > 2U -> 0.
    #   So Z_k = I_k * (u + U).  EXACT!  One channel per region (plus the indicator channel
    #   used to build it).
    #   Similarly Z'_k = ReLU(-u + M*I_k - Wc') with Wc' = M - U -> = I_k*(U - u).
    #   Then Z_k - Z'_k = I_k * 2u.
    #   So per region: indicator channel (I_k) + Z_k + Z'_k = 3 channels.
    #   With 64 conv2 channels: 21 regions max. Use 16 regions (48 channels) + 16 spare.
    #   Actually we could use 21 regions (63 channels) and 1 spare. Let's use 21 regions
    #   covering 32 timesteps: region r covers timesteps {r, r+21}? Better: assign regions
    #   to contiguous blocks: 32 timesteps / 21 regions -> uneven. Simpler: use 16 regions
    #   of 2 timesteps each (16*3 = 48 channels) + 16 spare channels for R_j copies.
    #   Even better: 16 regions of 2 timesteps: gain resolution 16 levels; the gain g(t)
    #   varies smoothly so 16 levels is plenty. But we can also fit the out layer with
    #   per-region coefficients, so effectively 16 distinct gains.
    #   Hmm, actually we can do better: 21 regions of ~1.5 timesteps. Let's just use 16
    #   regions of 2 timesteps each for clean indicator construction (region r covers
    #   t in {2r, 2r+1}).
    #   Wait, but the indicator for a *block* of timesteps: I_r(tau) = 1 iff t in {2r, 2r+1}.
    #   Using R_j = ReLU(tau - j/31): I_r = 31*(R_{2r-1} - 2R_{2r} + R_{2r+1})? Let's check for t=2r:
    #     R_{2r-1} = (t-(2r-1))/31 = 1/31, R_{2r} = 0, R_{2r+1} = 0 -> I = 1.
    #   For t=2r+1: R_{2r-1} = 2/31, R_{2r} = 1/31, R_{2r+1} = 0 -> 31*(2/31 - 2/31) = 0. Hmm, that's 0 but we want 1.
    #   Let's instead use exact per-timestep indicators I_k = 31*(R_{k-1} - 2R_k + R_{k+1}) (verified earlier),
    #   and for a block of two timesteps use I_{2r} + I_{2r+1}.
    #   For that we need R_j for j up to 32 (R_32 = ReLU(tau - 32/31) = 0 for t<=31; for t=31, tau=1, R_32 = 0). Actually R_j for j=0..31 suffices if we define R_{-1}=0 and R_32=0.
    #   Check I_31 = 31*(R_30 - 2R_31 + R_32) with R_32 = 0: for t=31: R_30 = 1/31, R_31 = 0 -> I=1. Good. For t=30: R_30=0, R_31=0 -> 0. Good.
    #   And I_0 = 31*(R_{-1} - 2R_0 + R_1) with R_{-1}=0: for t=0: R_0 = 0 (tau=0), R_1 = 0 -> 0. Oops! For t=0, tau=0, R_0 = ReLU(0-0) = 0, R_1 = 0 -> I_0 = 0. But we want I_0 = 1.
    #   Fix: I_k = 31*(R_{k-1} - 2R_k + R_{k+1}) where R_j = ReLU(tau - j/31). For t=k we need
    #   R_{k-1} = 1/31, R_k = 0, R_{k+1} = 0 -> I_k = 1. For k=0: R_{-1} = ReLU(tau+1/31) = tau + 1/31 = 1/31 at t=0 -> I_0 = 31*(1/31 - 0 + 0) = 1. Good, so R_{-1} = tau + 1/31 (i.e., R_{-1} = ReLU(tau + 1/31)).
    #   So we need R_j for j = -1..31 -> 33 channels. Too many for conv1 (32 channels, one
    #   used for u). Hmm.
    #   Alternative: build R_{-1} = tau + 1/31 inside conv2 from R_0 = tau (ch1) plus a
    #   constant offset: conv2 channel = R_0 + 1/31 (bias) -> but conv2's ReLU keeps it
    #   positive. Fine! conv2 can add constants via bias. But we need R_{-1} to appear in
    #   the *indicator* channel of conv2 — we can compute the indicator directly in conv2
    #   as a weighted sum of conv1 channels + bias:
    #     I_k = 31*(R_{k-1} - 2R_k + R_{k+1}) where R_{-1} = R_0 + 1/31 (as conv1 ch1 = tau).
    #   So conv2 indicator channel weights: from conv1 ch_{1+j} (R_j, j=0..30) with
    #   coefficients 31*(+1 for j=k-1, -2 for j=k, +1 for j=k+1), plus for k=0 the
    #   R_{-1} term: coefficient +31 on ch1 (tau) plus bias 1 (since 31*(1/31)=1).
    #   Wait: R_{-1} = tau + 1/31, so 31*R_{-1} = 31*tau + 1. And 31*R_0 = 31*tau. So
    #   I_0 = 31*R_{-1} - 2*31*R_0 + 31*R_1 = (31 tau + 1) - 2(31 tau) + 31 R_1
    #       = 1 - 31 tau + 31 R_1. At t=0: 1 - 0 + 0 = 1. At t=1: 1 - 31/31 + 0 = 0. Good.
    #   So conv2 indicator for k=0: weights: ch1 (tau) coefficient -31, ch2 (R_1) coefficient +31,
    #   bias +1. Note ch1 = R_0 = tau, ch2 = R_1, ..., ch31 = R_30.
    #   For k>=1: I_k = 31*R_{k-1} - 62*R_k + 31*R_{k+1} -> coefficients on ch_{1+k-1}, ch_{1+k},
    #   ch_{1+k+1} (indices within 1..31) -> for k=30: needs R_31 which doesn't exist ->
    #   use R_31 = 0 (ReLU(tau - 1) = 0 for t<=31) -> coefficient 0. For k=31: needs R_30,
    #   R_31(=0), R_32(=0) -> I_31 = 31*R_30 -> at t=31: 31*(1/31) = 1. Good.
    #   Great, all indicators computable in conv2 from conv1's 31 tau-channels.
    #   But conv2 has 64 channels; we need 32 indicators + 32 Z channels (Z_k and Z'_k for
    #   16 regions) = 64. Perfect fit! But Z_k depends on I_k which is computed in conv2 —
    #   we can't use conv2's own output in conv2. So Z_k must be computed in conv3!
    #   conv3 has 64 channels: 32 indicators (copied from conv2) + 32 Z channels. But Z needs
    #   I_k (conv2 output) and u (from conv1 ch0). conv3 takes conv2's output as input, so
    #   conv3 can't access conv1's u channel directly. Hmm.
    #   Fix: put u into conv2 as well: conv2 channels include copies of conv1 ch0 (u+U).
    #   Then conv3 can use conv2's copies of u and I_k to build Z.
    #   Layout:
    #     conv1 (32): ch0 = u + U ; ch1..ch31 = R_0..R_30 (tau-only).
    #     conv2 (64):
    #        ch0 : copy of u (from conv1 ch0, delta) -> value u+U
    #        ch1 : copy of u with weight -1 -> value -(u+U)?? ReLU would clip. Instead we
    #              need -u. Use conv1 ch0 weight -1 and bias +U: pre = -(u+U)+U = -u -> ReLU
    #              gives max(-u,0) not -u. Hmm, we want the raw -u.
    #              Alternative: keep u in conv2 as u+U (positive) and handle the -u case
    #              analytically: Z'_k = ReLU(-u + M*I_k - Wc'). We need -u as a pre-activation
    #              that may be negative -> ReLU clips it. But we want ReLU of a sum.
    #              Let's reformulate to avoid needing -u:
    #                Z_k  = ReLU(u + M*I_k - Wc)   -> I_k*(u+U) with Wc = M - U
    #                Z'_k = ReLU(u + M*I_k - Wc2) with a different offset? We want I_k*u.
    #                Actually: I_k*u = I_k*(u+U) - I_k*U = Z_k - I_k*U. And I_k*U is
    #                available (I_k is a channel, times constant U -> linear in features).
    #                So we only need Z_k per region (not Z'_k)! Because u = (u+U) - U and
    #                I_k*u = I_k*(u+U) - U*I_k.
    #                And I_k*(u+U) = Z_k (exact, as shown). So per region: 1 Z channel
    #                + the indicator channel (which we need anyway for the -U*I_k term).
    #                So per region: 2 channels in conv3 (indicator copy + Z). 16 regions ->
    #                32 channels in conv3, plus conv2 needs 16 indicator channels + 16 u-copies
    #                = 32 channels.
    #   Layout:
    #     conv1 (32): ch0 = u + U (delta on x-channel, bias U); ch1..ch31 = R_0..R_30.
    #     conv2 (64):
    #        ch0..15 : I_r for r=0..15 (block indicators: I_r = I_{2r} + I_{2r+1})
    #        ch16..31: copies of u+U (delta from conv1 ch0)
    #        ch32..63: copies of R_j (delta from conv1 ch1..ch31) -> 31 copies + 1 spare
    #     conv3 (64):
    #        ch0..15 : copies of I_r (delta from conv2 ch0..15)
    #        ch16..31: Z_r = ReLU(u + M*I_r - Wc) built from conv2 ch16+r (u+U) and
    #                   conv2 ch r (I_r): pre = (u+U) + M*I_r - (U + Wc)
    #                   with Wc = M - U -> pre = u + M*I_r - M + U - U + ... let me redo.
    #                   We want Z_r = ReLU( (u+U) + M*I_r - C ) = I_r*(u+U) when I_r=1 and
    #                   0 when I_r=0.
    #                   For I_r=1: (u+U) + M - C should be >= 0 for all u>=-U -> >= 0 -> C <= M.
    #                   For I_r=0: (u+U) - C < 0 for all u <= U -> C > 2U.
    #                   Contradiction unless M > 2U and C in (2U, M]. So need M > 2U.
    #                   Set M = 3U + 1 (with U=3: M=10), C = 2U + 0.5 = 6.5 (must be > 2U=6 and <= M=10). Good.
    #                   Then for I_r=1: (u+U) + M - C = u + U + 10 - 6.5 = u + 6.5 >= 3.5 > 0 -> Z = u+U+3.5?? Wait
    #                   Z_r = ReLU(pre) = (u+U) + M - C = u + 3 + 10 - 6.5 = u + 6.5. Hmm that's not u+U.
    #                   Let me recompute: we want Z_r = u + U (the value (u+U)) when I_r=1.
    #                   pre = (u+U) + M*I_r - C. With I_r=1: pre = u+U+M-C. We want Z_r = u+U,
    #   so we need M - C = 0 -> C = M. But then for I_r=0: pre = u+U-M < 0 for all u <= U iff M > 2U. Yes! M > 2U suffices.
    #                   So: C = M, with M > 2U (M = 10, U = 3). Then:
    #                     I_r=1: pre = u+U >= 0 -> Z_r = u+U (exactly, and >= 0 since u >= -U).
    #                     I_r=0: pre = u+U-M <= U+U-M = 2U-M < 0 -> Z_r = 0.
    #                   PERFECT and exact.
    #        ch32..63: copies of conv2 ch32..63 (the R_j channels) -> 32 copies.
    #     out (1): LS fit over the 64 conv3 features.
    #   Now the out layer sees: I_r (16), Z_r (16), R_j (32). It can compute
    #       out = sum_r a_r * (Z_r - U*I_r) + sum_j c_j R_j
    #          = sum_r a_r * I_r * u + sum_j c_j R_j(tau)
    #   -> a piecewise-constant gain a_r per 2-timestep block, plus additive t-dependent
    #   offsets via R_j.  The LS fit will choose a_r to best match the ideal gain g(t)
    #   and c_j for offsets.
    #   Wait: the offsets c_j R_j(tau) are additive in tau and can represent any
    #   piecewise-linear function of tau with knots at the 31 thresholds (plus constant).
    #   Good.
    #   Note: a_r is a single scalar per region shared across pixels -> the gain is
    #   spatially uniform. Good.
    #   Also note: the LS fit is over all pixels jointly (shared weights), so a_r will be
    #   chosen to minimize MSE across all examples in that region.
    #   This should achieve close to the optimal piecewise-constant gain.
    #   Additionally, the additive R_j terms let the model capture the optimal *offset*
    #   (E[eps|x_t] has a nonzero mean component when x0 has nonzero mean: E[eps] = -sqrt(a)E[x0]/sqrt(1-a)).
    #   Since MNIST images in [-1,1] have mean pixel ~ -0.5 or so, there IS an optimal
    #   offset. The R_j basis can represent a piecewise-linear function of tau -> good.
    #   Even better: we can also add "u" itself as a feature (unweighted by region) to allow
    #   a global linear term. Not needed since a_r covers it.

    # ---- implement the design ----
    U = 3.0
    M = 10.0
    # conv1
    c1w = np.zeros((32, 2, 3, 3), dtype=np.float32)
    c1b = np.zeros((32,), dtype=np.float32)
    c1w[0, 0, 1, 1] = 1.0
    c1b[0] = U
    # ch1..ch31 = R_j = ReLU(tau - j/31), j = 0..30
    for j in range(31):
        c1w[1 + j, 1, 1, 1] = 1.0
        c1b[1 + j] = -j / 31.0
    W['conv1.weight'] = c1w
    W['conv1.bias'] = c1b

    # conv2
    c2w = np.zeros((64, 32, 3, 3), dtype=np.float32)
    c2b = np.zeros((64,), dtype=np.float32)
    # ch0..15: block indicators I_r = I_{2r} + I_{2r+1}
    # I_k = 31*(R_{k-1} - 2R_k + R_{k+1}), R_{-1} = tau + 1/31, R_31 = R_32 = 0
    def ind_coeffs(k):
        # returns dict: conv1 channel index -> coeff, plus bias
        coef = {}
        bias = 0.0
        if k == 0:
            # 31*R_{-1} - 2*31*R_0 + 31*R_1 ; R_{-1} = tau + 1/31 = R_0 + 1/31
            coef[1] = 31.0 - 62.0          # ch1 = R_0
            bias += 1.0
            coef[2] = 31.0                  # ch2 = R_1
        else:
            if k - 1 <= 30:
                coef[1 + (k - 1)] = coef.get(1 + (k - 1), 0.0) + 31.0
            if k <= 30:
                coef[1 + k] = coef.get(1 + k, 0.0) - 62.0
            if k + 1 <= 30:
                coef[1 + (k + 1)] = coef.get(1 + (k + 1), 0.0) + 31.0
        return coef, bias

    for r in range(16):
        coef = {}
        bias = 0.0
        for k in (2 * r, 2 * r + 1):
            c2, b2 = ind_coeffs(k)
            for ci, cv in c2.items():
                coef[ci] = coef.get(ci, 0.0) + cv
            bias += b2
        for ci, cv in coef.items():
            c2w[r, ci - 1, 1, 1] = cv     # conv1 channel index ci -> conv1 axis index ci-1
        c2b[r] = bias
    # ch16..31: copies of u+U (conv1 ch0)
    for r in range(16):
        c2w[16 + r, 0, 1, 1] = 1.0
    # ch32..62: copies of R_j (conv1 ch1..ch31 -> conv1 idx 0..30)
    for j in range(31):
        c2w[32 + j, j, 1, 1] = 1.0
    # ch63: constant 1 (bias)
    c2b[63] = 1.0
    W['conv2.weight'] = c2w
    W['conv2.bias'] = c2b

    # conv3
    c3w = np.zeros((64, 64, 3, 3), dtype=np.float32)
    c3b = np.zeros((64,), dtype=np.float32)
    # ch0..15: copies of I_r (conv2 ch0..15)
    for r in range(16):
        c3w[r, r, 1, 1] = 1.0
    # ch16..31: Z_r = ReLU( (u+U) + M*I_r - M )  -> pre = conv2ch(16+r) + M*conv2ch(r) - M
    for r in range(16):
        c3w[16 + r, 16 + r, 1, 1] = 1.0
        c3w[16 + r, r, 1, 1] = M
        c3b[16 + r] = -M
    # ch32..63: copies of conv2 ch32..63
    for j in range(32):
        c3w[32 + j, 32 + j, 1, 1] = 1.0
    W['conv3.weight'] = c3w
    W['conv3.bias'] = c3b

    # ---- closed-form fit of the out layer on the designed features ----
    # compute conv3 features for all training examples (features = conv3 output)
    def conv3_features(xin, tin):
        Nn = xin.shape[0]
        tt = (tin.astype(np.float32) / 31.0).reshape(Nn, 1, 1, 1)
        inp = np.concatenate([xin, np.broadcast_to(tt, (Nn, 1, xin.shape[2], xin.shape[3]))], axis=1)
        h1 = relu(conv2d(inp, W['conv1.weight'], W['conv1.bias']))
        h2 = relu(conv2d(h1, W['conv2.weight'], W['conv2.bias']))
        h3 = relu(conv2d(h2, W['conv3.weight'], W['conv3.bias']))
        return h3

    # subsample pixels for speed: use all examples but a subset of pixels
    rng = np.random.default_rng(0)
    nsub = 40000
    idx_e = rng.integers(0, N, size=nsub)
    idx_p = rng.integers(0, 256, size=nsub)
    F = conv3_features(x[idx_e], t[idx_e])          # (nsub, 64, 16, 16)
    Fv = F[np.arange(nsub), :, idx_p // 16, idx_p % 16].astype(np.float64)   # (nsub,64)
    Yv = y[idx_e].reshape(N, -1)[idx_e, idx_p].astype(np.float64)
    # ridge LS
    lam = 1e-6
    A = Fv
    G = A.T @ A
    G[np.diag_indices_from(G)] += lam * (np.trace(G) / 64.0 + 1e-12)
    b = A.T @ Yv
    sol = np.linalg.solve(G, b)
    ow = sol.reshape(1, 64, 1, 1).astype(np.float32)
    ob = np.array([float(Yv.mean() - (A.mean(axis=0) @ sol))], dtype=np.float32)
    W['out.weight'] = ow
    W['out.bias'] = ob

    # ---- evaluate ----
    h3f, pred = forward(x, t, W)
    mse_new = mse(pred, y)
    mse_old = mse(np.asarray(f['prediction'], dtype=np.float32), y)

    # ---- safety: clip weights, ensure finite ----
    for k in W:
        W[k] = np.nan_to_num(np.asarray(W[k], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        W[k] = np.clip(W[k], -1000.0, 1000.0).astype(np.float32)

    diag = {
        'mse_before': round(mse_old, 5),
        'mse_after': round(mse_new, 5),
        'sigma_x0_sq': round(s2, 5),
        'gain_fit': [round(float(v), 4) for v in gain[:8]],
        'n_fit_samples': int(nsub),
        'out_w_absmax': round(float(np.abs(W['out.weight']).max()), 5),
    }
    return {'weights': W, 'diagnostics': diag}
