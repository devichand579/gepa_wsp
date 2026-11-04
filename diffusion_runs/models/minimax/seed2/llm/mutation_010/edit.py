import numpy as np

def edit(w, f):
    # ------------------------------------------------------------------
    # 0) helpers
    # ------------------------------------------------------------------
    def conv3x3(x, weight, bias):
        """x: (N,C,H,W) float32; weight: (O,C,3,3) -> (N,O,H,W)"""
        N, C, H, W = x.shape
        O = weight.shape[0]
        xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
        out = np.zeros((N, O, H, W), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                out += np.tensordot(xp[:, :, i:i + H, j:j + W],
                                    weight[:, :, i, j], axes=([1], [1])).transpose(1, 0, 2, 3)
        out += bias.reshape(1, O, 1, 1)
        return out

    def key_of(name):
        return name.split('.')[-1]

    W = {key_of(k): np.asarray(v, dtype=np.float64) for k, v in w.items()}

    x = np.asarray(f['x'], dtype=np.float64)          # (N,1,16,16)
    t = np.asarray(f['t']).astype(np.int64).ravel()   # (N,)
    y = np.asarray(f['y'], dtype=np.float64)          # (N,1,16,16)
    ab = np.asarray(f['alphabar'], dtype=np.float64).ravel()
    N = x.shape[0]
    T = ab.shape[0]

    # ------------------------------------------------------------------
    # 1) per-timestep closed-form linear fits  y ~ sum_off w_off * x_shifted + b
    # ------------------------------------------------------------------
    # build shifted copies of x (zero padding) for a 3x3 neighbourhood
    H, Wd = x.shape[2], x.shape[3]
    xp = np.pad(x[:, 0], ((0, 0), (1, 1), (1, 1)))     # (N,18,18)
    shifts = []
    for dy in range(3):
        for dx in range(3):
            shifts.append(xp[:, dy:dy + H, dx:dx + Wd])          # (N,H,W)
    A = np.stack(shifts, axis=1)                                 # (N,9,H,W)
    A = A.reshape(N, 9, H * Wd)
    yv = y[:, 0].reshape(N, H * Wd)

    # global (all-t) fit as fallback
    def fit(idx, ridge):
        if len(idx) == 0:
            return None
        Ai = A[idx].transpose(0, 2, 1).reshape(-1, 9)
        bi = yv[idx].reshape(-1)
        G = Ai.T @ Ai
        G[np.diag_indices(9)] += ridge
        b = Ai.T @ bi
        try:
            coef = np.linalg.solve(G, b)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(G, b, rcond=None)[0]
        res = bi - Ai @ coef
        return coef, float(np.mean(res * res)), len(idx)

    global_fit = fit(np.arange(N), 1e-3)

    coefs = {}
    for tt in range(T):
        idx = np.where(t == tt)[0]
        r = fit(idx, 1e-2)
        if r is None:
            coefs[tt] = (global_fit[0], global_fit[1])
        else:
            coefs[tt] = (r[0], r[1])

    # ------------------------------------------------------------------
    # 2) build the network that realises  out = a_t * x + b_t
    #
    #  conv1 (32ch):
    #     ch 0 : ReLU(+x)                     (signed positive part)
    #     ch 1 : ReLU(-x)                     (signed negative part)
    #     ch 2..31 : gate channels, one per timestep t0=0..29:
    #        pre = x + s_{t0}(t)  with s_{t0}(t)=0 at t=t0, <= -32 otherwise
    #        -> active only at t=t0, value = ReLU(x)
    #  conv2 (64ch): copies sign channels and gate channels (identity),
    #        plus 2 channels carrying the *negative* branch of the gates:
    #        conv2 ch_c = -(gate value) is not possible (ReLU) -> instead we
    #        use the identity  a*x = a*ReLU(x) - a*ReLU(-x); the ReLU(-x)
    #        part is produced by a SECOND gate family built on -x:
    #        conv1 ch for -x gates would exceed 32 channels, so instead
    #        we exploit that conv2 can build ReLU(-x + s(t)) from conv1 ch1
    #        (which equals ReLU(-x)) only when s(t)=0 ... not possible.
    #  ==> Use a different, exact construction:
    #        out = a_t * x  is realised as
    #             a_t * x = a_t * (ReLU(x) - ReLU(-x))
    #        We obtain a_t*ReLU(x) from a gated channel on +x, and
    #        a_t*ReLU(-x) from a gated channel on -x.  Both families need
    #        30 gates each -> 60 conv1 channels, too many.
    #        Instead: note the gate offset can be applied to the *t* channel
    #        of a conv whose x-part is the identity, and the SAME gate value
    #        can be reused for both signs by two conv2 outputs:
    #            conv2 ch = ReLU( +x + s(t) )  and  ReLU( -x + s(t) )
    #        Both are computed from conv1 outputs?  conv1 outputs are already
    #        post-ReLU, so we cannot re-apply a threshold.
    #        => Use conv1 to output the *raw* (pre-ReLU) is impossible.
    #        => Therefore we use the following exact alternative:
    #            a_t * x + b_t  =  a_t * x  and we realise it with a gated
    #            channel whose pre-activation is  x  (signed) by using the
    #            ReLU trick on the OUTPUT side:
    #                ReLU(x + s) - ReLU(x - s)  is not linear either.
    #        Practical exact solution: build the gate on the *t* channel and
    #        let the x-part be replicated through conv2/conv3 with the sign
    #        handled by out-layer weights:
    #            conv3 ch A: pre = x_pos_gate  -> post = ReLU(x) at t only
    #            conv3 ch B: pre = -x_pos_gate -> post = 0 (ReLU kills it)
    #        so the negative part of x is unavailable at conv3 level.
    #        => we therefore use the *global* (t-independent) sign split and
    #           apply the per-timestep coefficient through a SECOND gate
    #           family built on -x.  60 conv1 channels are needed, but
    #           conv1 has only 32.  We instead split the timesteps between
    #           the two families: 16 timesteps use +x gates, 16 use -x
    #           gates.  A timestep handled by the -x family contributes
    #             a_t*ReLU(-x)  and we need  a_t*(ReLU(x)-ReLU(-x));
    #           the missing ReLU(x) term for that timestep is supplied by
    #           a *non-gated* channel that is active only at that timestep
    #           via ... (not available)  -> not exact.
    #
    #  FINAL EXACT CONSTRUCTION (used below):
    #  We use per-timestep gates on a SINGLE channel that carries the
    #  *signed* value through the ReLU-free path: the out layer is the only
    #  ReLU-free stage, so we push the sign handling to it:
    #      conv1 ch0 = ReLU(x)      (all t)
    #      conv1 ch1 = ReLU(-x)     (all t)
    #      conv1 ch2..31 = gates on x:  pre = x + s_t(t)  -> at t=t0 gives
    #                       ReLU(x), elsewhere 0.
    #  Then conv2/conv3 pass these through unchanged, and the OUT layer
    #  (linear, no activation) forms:
    #      out = sum_{t0} a_{t0} * ( gate_{t0} - gateNeg_{t0} )
    #  where gateNeg_{t0} is produced by conv2/conv3 as
    #      conv2 ch = ReLU( -x + s_t(t) )   -- impossible from post-ReLU
    #  => We instead build the negative gates directly in conv1 by using
    #     the SECOND input slot of conv1 (the t channel) to carry -x?? no.
    #
    #  ==> RESOLUTION: use 30 gate channels on +x (t0 = 0..29) and obtain the
    #      negative branch by a *second* set of 30 gates implemented in
    #      conv2 using conv1's *pre-gate* structure: conv1 ch2..31 give us
    #      ReLU(x + s(t)); we need ReLU(-x + s(t)).  Note that
    #          ReLU(-x + s) = s - ReLU(x - s)   ... only if s>=0 and x<=s.
    #      Not exact.
    #
    #  ==> TRUE RESOLUTION (exact, used): exploit that we need only the
    #      *difference* ReLU(x)-ReLU(-x) = x, and that a gated channel can
    #      carry a *signed* quantity if the gate is implemented as an
    #      additive offset applied to a channel whose x-part is the identity
    #      and whose bias makes it always positive:
    #          pre = x + M + s(t),  M = 64  (always > 0 for all x,t)
    #          post = x + M + s(t)  (no ReLU clipping, since always positive)
    #      This is an exact *affine* copy of x with a t-dependent offset!
    #      Then conv2 can form  x + s(t)  (subtract M) and again
    #          pre = x + s(t)  -> ReLU -> ReLU(x) at t=t0, 0 elsewhere.
    #      And for the negative branch we use the same trick with -x:
    #          conv1 ch = -x + M + s(t)  -> post = -x + M + s(t) (positive)
    #      So each conv1 channel can carry a signed affine copy of x with a
    #      t-dependent offset.  We need 2 channels per timestep (signs),
    #      but we can build the negative copy in conv2 from the positive
    #      copy: conv2 ch = -(x + M + s(t)) + 2M + s(t)?? s(t) unknown in
    #      conv2 (no t channel there).  Instead: conv2 ch = M - (x+M+s) = -x-s
    #      -> post = ReLU(-x - s(t)) = ReLU(-x) at t=t0 (since s=0), else 0.
    #      YES! That works: conv2 sees conv1's positive copy, negates it and
    #      adds M: pre = -x - s(t); at t=t0 s=0 -> ReLU(-x); elsewhere
    #      s<=-32 -> pre = -x+32 > 0 -> post = 32 - x  (WRONG, not zero).
    #      Hmm: for other t we need the channel to be ZERO.  With pre =
    #      -x - s(t) and s(t) = -32 -> pre = -x + 32 > 0 -> not zero.
    #      So instead the gate must kill the channel: use pre = x + s(t)
    #      (s = -32 -> pre = x-32 < 0 -> ReLU -> 0).  For the negative
    #      branch we need pre = -x + s(t) with s = 0 at t0 and s <= -32
    #      elsewhere; but conv2 cannot compute s(t).
    #      => conv1 must build the negative copy itself: conv1 ch = -x + s(t)
    #      directly (weight on x-channel = -identity, on t-channel = s(t)).
    #      That needs 2 conv1 channels per timestep (60 total) > 32.
    #
    #  ==> FINAL: reduce the number of gated timesteps by noting the
    #      per-timestep coefficient a_t can be *interpolated* by a
    #      piecewise-linear function of t inside a single channel group:
    #      use the "affine copy" trick to carry  x + s(t)  (exact, no ReLU
    #      clipping) and then let the OUT layer apply a t-dependent
    #      coefficient via a *second* affine copy of t: the out layer is
    #      linear over channels, so
    #         out = sum_c w_c * conv3_c
    #      and conv3_c can be  x * (t-dependent)  only if we multiply.
    #      => we accept an approximation: use per-timestep gates but only
    #         for the 32 timesteps, with ONE channel per timestep carrying
    #         the *signed* x through the "always positive offset" trick,
    #         and handle the sign by making the gated channel's value
    #         = x + M + s(t)  (exact signed copy, offset known per t) and
    #         then producing the *coefficient* a_t by using the SAME
    #         channel's constant part: out = w * (x + M + s(t)) summed over
    #         gates gives sum_t0 w_{t0} x  (all timesteps active!) -- no.
    #
    #  ==> TRULY FINAL, SIMPLE, EXACT: exploit that the t channel is a
    #      constant map, so conv1 can compute  ReLU(x + s(t))  where
    #      s(t) is any piecewise-linear function of t with 30 hinges.
    #      Choose s(t) to be a *large* positive value M for all t except
    #      the target, where it is 0.  Then at the target t:
    #          post = ReLU(x) = x_pos
    #      and at other t: post = ReLU(x + M) = x + M  (always positive).
    #      That is the opposite of gating.
    #      Choose s(t) = 0 at target and -M elsewhere -> post = x_pos at
    #      target, 0 elsewhere.  This gives a_t * x_pos per timestep with
    #      ONE conv1 channel per timestep.  For the x_neg part we use the
    #      SAME trick but with the x-part negated: conv1 ch = -x + s(t).
    #      That needs 2 channels per timestep -> 64 > 32.
    #      SOLUTION: use conv2 to create the negative branch from the
    #      positive gate, exploiting that at the target t the gate value is
    #      x_pos and at other t it is 0:
    #          conv2 ch = M - gate   (M = 64)
    #          at target: M - x_pos  -> ReLU -> M - x_pos (since x_pos<=~4)
    #          at others: M - 0 = M -> ReLU -> M  (constant, not zero!)
    #      Then conv3 ch = M - that = x_pos at target, 0 elsewhere.  But
    #      conv3's ReLU keeps x_pos (positive) -> we get x_pos again, not
    #      x_neg.  To get x_neg we need the *negative* of the positive part,
    #      which is not recoverable from x_pos alone (we lost the sign).
    #      => information loss is unavoidable with one gate per timestep.
    #
    #  ==> THEREFORE: use 16 timesteps per sign family is wrong; instead
    #      use ONE gate per timestep on +x, and obtain the negative branch
    #      by a *separate* mechanism: a global (ungated) channel carrying
    #      ReLU(-x) (conv1 ch1) whose coefficient is applied per timestep
    #      by a *gated scalar*: out = sum_{t0} a_{t0} * gate_{t0}(x_pos)
    #                        + sum_{t0} a_{t0} * (something for x_neg)
    #      The x_neg coefficient must also be t-dependent -> same problem.
    #
    #  ==> BREAKTHROUGH: we can make the gate carry BOTH signs by using a
    #      *large negative* offset on the t channel so that the channel is
    #      active for ALL t (never clipped) and equals  x + s(t) exactly
    #      (affine, signed).  Then conv2 computes  ReLU( -(x+s(t)) + M )?
    #      Still no t-dependence.
    #      ==> Use TWO conv1 channels per timestep but only 16 timesteps
    #          need gating: note the coefficient a_t is smooth in t; we can
    #          approximate a_t by a piecewise-linear function of t with
    #          knots at every 2 timesteps (16 knots) -> 15 hinges + constant.
    #          Then out = (sum_j d_j h_j(t)) * x  is still a product...
    #      ==> BUT: with the "affine copy" trick we can realise the product
    #          exactly as follows.  Let conv1 channel c carry  x + s_c(t)
    #          (exact, signed, always positive due to +M offset).  Then
    #          conv2 channel c' can compute  w * (x + s_c(t))  for any fixed w
    #          (linear).  Now, the desired output is  a_t x + b_t.  Write
    #          a_t = sum_j d_j h_j(t) with h_j(t) = ReLU(t - j) (hinges).
    #          Then a_t x = sum_j d_j h_j(t) x  = sum_j d_j * [h_j(t) * x].
    #          We need h_j(t)*x.  Since h_j(t) is a per-example scalar and x
    #          is per-pixel, and we can produce the *product* via the
    #          following trick: conv1 channel j produces  x + M  (affine copy
    #          of x, always positive).  conv2 channel j produces
    #              h_j(t) + M   ... we need h_j(t) from the t channel, which
    #          conv2 does not have.  So conv1 must carry h_j(t) too.
    #          conv1 has 32 channels: we can allocate 15 for hinges
    #          h_j(t) (j=1..15, using t/31 input and bias) and 17 for
    #          affine copies of x? But the affine copy needs a t-dependent
    #          offset only for gating, not for the product construction.
    #          For the product h_j(t)*x we need, in conv2, a channel whose
    #          pre-activation is  h_j(t) * x  -- a product of two conv1
    #          channels.  Conv is linear -> impossible.
    #  ==> So exact t-dependent scaling of x is IMPOSSIBLE in this
    #      architecture (ReLU nets are piecewise linear; a_t(t)*x is not
    #      piecewise linear in (t,x)).  We must APPROXIMATE.
    #
    #  APPROXIMATION USED (exact and optimal for a *piecewise-linear-in-t*
    #  coefficient):  use the GATED construction with one gate per timestep
    #  on the POSITIVE part, and one gate per timestep on the NEGATIVE
    #  part, but implement the negative gates in conv2 by exploiting the
    #  following identity that IS exact:
    #      For a gate channel G (value = ReLU(x) at t0, 0 elsewhere) and the
    #      global channel N = ReLU(-x) (all t), we want at t0: -a*N.
    #      We can get -a*N at t0 only if we can gate N.  Gating N requires
    #      a conv1 channel with -x and a t-offset -> 1 more channel per t.
    #      => 2 channels per timestep in conv1, 32 channels available.
    #      => Only 16 timesteps can be gated exactly this way.
    #      BUT: we can use a 3x3 *spatial* trick: the gate offset s(t) can
    #      be applied to the t-channel of conv1 with a 3x3 kernel whose
    #      spatial taps are ZERO for x but the t-channel is constant, so
    #      only one channel per (timestep, sign) is needed regardless of
    #      spatial shifts.  Still 2 per timestep.
    #
    #  => RESOLUTION: use 16 timesteps gated exactly (both signs) and for
    #     the remaining 16 timesteps use a *shared* gate that is active
    #     over a contiguous range of t (a "block gate"): a channel with
    #     offset s(t) that is 0 for t in a block and -M outside is not
    #     piecewise linear in t with only 30 hinges... but we can build
    #     s(t) as a piecewise-linear function of t with 30 hinges, and
    #     s(t) = 0 exactly on a block of t values is impossible for a
    #     continuous PL function unless it is 0 everywhere on that block
    #     and the function is PL with knots at the block ends -> then it is
    #     0 on the block and can ramp outside.  YES: s(t) = 0 for t in
    #     [a,b] and negative outside, e.g. s(t) = -M*max(0, a-t)/31 ...
    #     PL with knots at a and b: s(t) = 0 on [a,b], s(t) = -(a-t)*K for
    #     t<a (K large enough), s(t) = -(t-b)*K for t>b.  Then the channel
    #     is active for all t in [a,b] -> gives ReLU(x) for a *range* of
    #     timesteps, with a *single* coefficient.  So we can cover the
    #     32 timesteps with a few blocks, each with its own coefficient.
    #     With 30 hinges in conv1 we can afford ~15 blocks x 2 signs = 30
    #     channels!  Each block covers a range of t and gets its own
    #     coefficient (a_t approximated as constant within the block, or
    #     better: use blocks of size 1-2 to keep the approximation tight).
    #
    #  => FINAL PLAN (implemented below):
    #     * conv1 ch0 = ReLU(x)          (global positive part)
    #     * conv1 ch1 = ReLU(-x)         (global negative part)
    #     * conv1 ch2..31 = 30 "block gates" on +x and -x:
    #         15 blocks for +x (channels 2..16) and 15 for -x (17..31).
    #         Each block b covers a contiguous range of timesteps; the
    #         gate offset s_b(t) is PL in t: 0 on the block, <= -M outside.
    #         The conv1 kernel for a gate: x-part = +-identity (center tap),
    #         t-part = 0, and the *offset* must come from the t channel:
    #         pre = +-x + (w_t * t/31 + b).  We need w_t*(t/31)+b = 0 on the
    #         block and <= -M outside -> PL in t with 2 knots; but the conv1
    #         t-part is a single linear function of t (w_t*t/31 + b) -> only
    #         ONE knot.  A single-knot function that is 0 on a block and
    #         -M outside needs 2 knots -> not possible with one channel.
    #         => use a *two-channel* gate: ch_a provides the "ramp up"
    #         (0 below a, negative above a) and ch_b the "ramp down";
    #         but their sum must be 0 on the block -> we need to ADD them,
    #         which conv2 can do (linear!) and then ReLU... but ReLU would
    #         clip.  Actually conv2 output = ReLU(sum) and we want the sum
    #         to be 0 on the block -> ReLU(0)=0 -> channel dead on block.
    #         We want the channel to be ALIVE on the block.  So instead:
    #         build the gate as:  conv2 ch = ReLU( A ) where
    #             A = (x + M + s(t))  ... still needs s(t) PL with 2 knots.
    #         => conv1 CANNOT produce a 2-knot offset with one channel, but
    #            conv2 CAN: conv2 ch = sum of conv1 channels (linear) then
    #            ReLU.  So: conv1 provides hinge channels h_1(t)..h_30(t)
    #            (30 channels, each a PL hinge in t: ReLU(t/31 - j/31)).
    #            conv2 computes  s(t) = sum_j c_j h_j(t) + c_0  (any PL
    #            function of t, exact) and combines it with the x-part:
    #            conv2 ch = ReLU( x + M + s(t) )  -> since x+M+s(t) > 0
    #            always (M large), the ReLU is inactive and the channel
    #            carries the exact affine copy  x + M + s(t).
    #            Then conv3 ch = ReLU( ... ) and we can subtract M etc.
    #            Now the GATE: we want a channel that is 0 for t outside a
    #            block.  Using the affine copy trick we can build, in conv3:
    #                pre = (x + M + s(t)) - (M + s(t))  = x   -> ReLU(x)
    #            but s(t) must be cancelled exactly, which requires conv3 to
    #            have access to s(t) as a channel: it does (conv2 can output
    #            s(t) itself as a channel, since s(t) is a PL function of t
    #            built from conv1 hinges).  So conv3 can compute
    #                pre = (affine copy) - (s-channel) - M = x
    #            -> ReLU(x): but that's not gated either (x for all t).
    #            To GATE we need the offset to be -M outside the block:
    #                s_gate(t) = 0 on block, <= -M outside  (2-knot PL)
    #                conv3 pre = x + M + s_gate(t)  -> always positive ->
    #                carries x + M + s_gate(t) (exact affine, no ReLU clip)
    #                conv3 post = that.  Then out (linear) can subtract the
    #                constant M + s_gate(t) using a separate s_gate channel
    #                carried to conv3 -> out = x + M + s_gate(t) - M - s_gate(t)
    #                    = x  ... but we want the channel to be ZERO outside
    #                    the block.  Subtracting s_gate(t) cancels it exactly,
    #                    leaving x everywhere.  To kill it outside the block
    #                    we must NOT cancel: instead we want the *value* to
    #                    be zero outside.  Since s_gate(t) <= -M outside, the
    #            affine copy is x + M + s_gate(t) <= x - something -> negative
    #            -> ReLU kills it!  So: conv3 pre = x + M + s_gate(t) with
    #            M chosen so that on-block it's positive (x+M > 0) and
    #            off-block it's negative (x + M - M' < 0).  Since s_gate is
    #            <= -M' off-block with M' > max|x| + M...  Let's set:
    #                on-block:  s_gate = 0            -> pre = x + M  > 0
    #                off-block: s_gate = -(M + 40)     -> pre = x - 40 < 0
    #            Then conv3 post = ReLU(pre) = x + M on-block, 0 off-block.
    #            Then out = sum over blocks: w_b * (gate_b - M) ... the -M
    #            constant is a per-example constant (M) which we can cancel
    #            via out.bias? No: out.bias is a global constant, but the
    #            "M" appears only on-block (gated) -> we need to subtract M
    #            only on-block.  Use a second gated channel that carries the
    #            constant 1 on-block: conv3 ch = ReLU( M + s_gate(t) )?  that's
    #            M on-block, 0 off-block (if M + s_gate = M - M - 40 <0 off).
    #            Then out = sum_b w_b*(gate_b - const_b) gives w_b * x on-block.
    #            This is exact!  And the negative branch: same with -x.
    #            So per block we need in conv3: 2 channels (+x gate, -x gate)
    #            plus 2 constant channels (or 1 shared if the same M).
    #            conv3 has 64 channels; conv2 has 64.
    #            conv1 has 32 channels: we need the hinge channels h_j(t)
    #            (up to 30) to build s_gate(t) with 2 knots -> we need
    #            hinges at the block boundaries.  With B blocks we need
    #            hinges at each block start and end -> about 2B hinges.
    #            Plus we need the x-parts: conv1 must carry x and -x to
    #            conv2 (2 channels) and the hinge channels.
    #            So conv1 channels: 2 (x, -x) + H hinges.  H = 30 available.
    #            With B blocks (each covering a contiguous t range) we need
    #            hinges at the 2B boundary points -> H = 2B <= 30 -> B <= 15.
    #            But we also need the s_gate(t) PL function to be built in
    #            conv2 as a linear combination of the hinge channels: conv2
    #            channel = sum_j c_j h_j(t) (+ bias) -> s_gate(t).  Fine.
    #            Then conv2 must ALSO carry the x-parts to conv3 together
    #            with s_gate(t) so that conv3 can form x + M + s_gate(t):
    #            conv3 pre = w_x * (x channel) + w_s * (s_gate channel) + M
    #            -> needs conv2 to output both x and s_gate in separate
    #            channels: conv2 ch_x = ReLU(x + M) (affine copy, always
    #            positive -> exact), conv2 ch_s = s_gate(t) + M2 (make it
    #            positive so ReLU doesn't clip: s_gate ranges in [-M-40, 0]
    #            -> add M+40 -> range [0, M+40]; but then conv3 must
    #            subtract (M+40) -> constant, fine, it's per-example constant
    #            but it's the SAME for all blocks if we use one global
    #            s_gate channel... but each block needs its own s_gate!
    #            => conv2 needs one s_gate channel per block: B channels.
    #            Plus conv2 needs x and -x channels: 2.
    #            Plus conv2 needs the hinge channels to build s_gate: they
    #            are conv1 outputs; conv2 can use them internally (linear
    #            combination) -> conv2 ch_s_b = sum_j c_bj h_j + bias.
    #            => conv2 channels: 2 (x copies) + B (s_gate) + spare.
    #            conv3 channels: per block: +x gate, -x gate, and constant
    #            gate (to cancel M) -> 3B, plus spare.  3B <= 64 -> B <= 21.
    #            Combined with B <= 15 from conv1 hinges -> B <= 15.
    #            Also we need per-block bias b_t: the additive term b_t must
    #            also be t-dependent (piecewise constant per block) -> we can
    #            add it via the constant gate channel: out = w_b * const_gate
    #            where const_gate = M on-block, 0 off -> contributes w_b*M.
    #            So per block: out contribution = a_b*(x on-block) + c_b*(const
    #            on-block) -> gives a_b*x + c_b*M on-block.  Perfect: we get
    #            per-block affine model with 2 dof (a_b, b_b) — exactly what
    #            we need (a_t, b_t) approximated per block.
    #            => B blocks with B <= 15 blocks covering 32 timesteps.
    #            To keep the approximation tight, use B = 15 blocks of size
    #            ~2 timesteps.  Actually we can do better: the number of
    #            distinct (a_t,b_t) values needed is 32, but with 15 blocks
    #            we approximate.  We could also use 2 blocks per sign family
    #            to double B? The hinge budget: 2B hinges needed (block
    #            boundaries).  Hinges available: 30 - (channels for x,-x) = 30.
    #            Hmm: conv1 has 32 channels: 2 for x/-x, 30 for hinges.
    #            Each block needs hinges at its two boundaries -> but adjacent
    #            blocks share boundaries -> B+1 distinct boundaries -> B+1
    #            hinges suffice (a PL function with knots at the boundaries
    #            can be 0 on each block and negative outside? Let's think:
    #            s_gate_b(t) must be 0 on block b and <= -M' outside.
    #            Using hinges h_j(t) = max(0, t-j) (j=1..30) and a constant:
    #            s_b(t) = -M' * [ max(0, a_b - t) + max(0, t - c_b) ] / 31
    #            where block b = [a_b, c_b].  max(0, a_b - t) = a_b - min(a_b,t)
    #            = a_b - t + max(0, t-a_b) -> needs hinge at a_b.
    #            max(0, t - c_b) -> hinge at c_b.
    #            So each block needs hinges at a_b and c_b -> up to 2B
    #            distinct knots, but many coincide (block boundaries shared
    #            between adjacent blocks) -> if blocks partition [0,31], the
    #            knots are the interior boundaries + the two ends.  For block
    #            b with boundaries (a_b, c_b): needs hinge at a_b and c_b.
    #            Adjacent blocks share a boundary -> total distinct knots
    #            = B+1.  But the hinge at the ends (a_1=0, c_B=31) would be
    #            hinge at 0 (constant, handled by bias) and hinge at 31
    #            (outside range of t/31 hinges? we have hinges j=1..30).
    #            So knots = interior boundaries (B-1 of them) + 30 (for the
    #            last block's upper end, but t<=31 so max(0,t-31)=0 always ->
    #            not needed) -> we need hinges at interior boundaries only:
    #            B-1 hinges.  Plus we need the "ramp" for the first block's
    #            lower end (max(0, 0 - t) = 0 for t>=0) -> not needed.
    #            So B-1 hinges suffice -> B <= 31!  But conv3 needs 3B
    #            channels <= 64 -> B <= 21.  And conv2 needs B+2 <= 64.
    #            => B = 21 blocks, 20 hinges, conv1: 2 + 20 = 22 channels
    #            used, 10 spare.  conv3: 3*21 = 63 channels.
    #            Even better: we can use 2 hinges per block if we want
    #            non-contiguous blocks; not needed.
    #            With B = 21 blocks over 32 timesteps, blocks of size 1-2.
    #            Let's use B = 16 blocks of 2 timesteps each -> hinge knots at
    #            2,4,...,30 -> 15 hinges; conv3 needs 48 channels; conv1 uses
    #            17 channels.  Coefficient error within a 2-step block is
    #            small.  Or B=21 with mixed sizes.  Let's implement generally
    #            with B blocks and per-block least-squares fit of (a,b) using
    #            the examples in that block (pooled), so the block fit is
    #            optimal for the pooled timesteps.
    #     * Additionally, we can include a 3x3 spatial filter? With the
    #       gating construction, the gated channel carries x (center only).
    #       To get spatial shifts we'd need 9 gated channels per block ->
    #       too many.  So we use the 1x1 (center pixel) model plus, to
    #       capture some spatial info, we can add a *global* (ungated)
    #       3x3-filtered term? Its coefficient would be t-independent ->
    #       less useful.  Let's evaluate: is the 1x1 per-timestep model
    #       close to the 3x3 per-timestep model? We'll compute both MSEs
    #       from the data and choose the better-supported design.  If the
    #       3x3 gain is large we can implement a hybrid: per-block
    #       center-pixel term (gated) + per-block *shifted* terms using the
    #       same gate mechanism but with shifted x copies (each shift needs
    #       its own gated channel: 9 shifts x 2 signs x B blocks -> too
    #       many).  Compromise: implement the center-pixel gated model
    #       (exact) and additionally a *global* ungated 3x3 term whose
    #       coefficient is constant across t (captures a bit of the
    #       denoising structure).  We can also allow per-block *one*
    #       extra spatial shift (e.g. the 4-neighbour average) if channels
    #       permit: conv3 has 64 channels; with B=16 blocks using 3
    #       channels each = 48, leaving 16 -> we could add 1 extra gated
    #       shift per block (16 channels) using the same gate (the gate
    #       channel s_gate_b is available in conv2, so conv3 can form
    #       shifted-x + M + s_gate_b for one extra shift per block).
    #       Let's do: per block, gated channels: +x(center), -x(center),
    #       +x(shift (1,1) i.e. down-right), -x(shift(1,1)), const.
    #       That's 5 per block -> 5*16 = 80 > 64.  Use B=12 blocks (of
    #       ~3 timesteps) -> 60 channels.  Hmm, trade-off.
    #       Let's first measure how much the 3x3 helps vs 1x1 per timestep.
    #       If small, use 1x1 with B=16 (48 channels) and keep it simple.
    #
    # ------------------------------------------------------------------
    # 3) Measure candidate models on the training data (closed form)
    # ------------------------------------------------------------------
    # per-timestep 1x1 fit:  y ~ a_t * x + b_t
    # per-timestep 3x3 fit: y ~ sum_off w_off * x_off + b_t
    # per-block 1x1 fit (pooled over the block's timesteps)
    # ------------------------------------------------------------------
    # (implemented below)

    # ---- 1x1 per-timestep fits
    a_t = np.zeros(T); b_t = np.zeros(T)
    for tt in range(T):
        idx = np.where(t == tt)[0]
        if len(idx) == 0:
            a_t[tt] = global_fit[0][0]; b_t[tt] = 0.0; continue
        xi = x[idx, 0].reshape(len(idx), -1)
        yi = yv[idx].reshape(len(idx), -1)
        Xm = np.concatenate([xi, np.ones((len(idx), 1))], axis=1)
        G = Xm.T @ Xm; G[np.diag_indices(2)] += 1e-4
        try:
            c = np.linalg.solve(G, Xm.T @ yi.reshape(-1))
        except np.linalg.LinAlgError:
            c = np.linalg.lstsq(G, Xm.T @ yi.reshape(-1), rcond=None)[0]
        a_t[tt] = c[0]; b_t[tt] = c[1]
    pred_1x1 = (a_t[t][:, None] * x[:, 0].reshape(N, -1) + b_t[t][:, None]).reshape(N, H, Wd)
    mse_1x1 = float(np.mean((pred_1x1 - yv.reshape(N, H, Wd)) ** 2))

    # ---- 3x3 per-timestep fits (for reference / possible hybrid)
    mse_3x3 = None
    coef3 = {}
    for tt in range(T):
        idx = np.where(t == tt)[0]
        if len(idx) == 0:
            coef3[tt] = global_fit[0]; continue
        r = fit(idx, 1e-2)
        coef3[tt] = r[0]
    # evaluate 3x3 model
    tot = 0.0
    for tt in range(T):
        idx = np.where(t == tt)[0]
        if len(idx) == 0: continue
        c = coef3[tt]
        pr = np.tensordot(A[idx], c, axes=([1], [0]))
        tot += float(np.sum((pr - yv[idx]) ** 2))
    mse_3x3 = tot / (N * H * Wd)

    # ---- choose block partition
    # blocks: contiguous ranges of timesteps; B chosen so that 3B+1 <= 64
    # and B-1 hinges <= 30 and B+2 <= 64
    B = 16
    edges = np.linspace(0, T, B + 1).astype(int)
    blocks = []
    for bi in range(B):
        lo, hi = int(edges[bi]), int(edges[bi + 1])   # timesteps lo..hi-1
        blocks.append((lo, hi))

    # per-block pooled 1x1 fit (a_b, b_b) using all examples in block
    a_b = np.zeros(B); b_b = np.zeros(B)
    for bi, (lo, hi) in enumerate(blocks):
        idx = np.where((t >= lo) & (t < hi))[0]
        if len(idx) == 0:
            a_b[bi] = a_t[lo]; b_b[bi] = b_t[lo]; continue
        xi = x[idx, 0].reshape(len(idx), -1)
        yi = yv[idx].reshape(len(idx), -1)
        Xm = np.concatenate([xi, np.ones((len(idx), 1))], axis=1)
        G = Xm.T @ Xm; G[np.diag_indices(2)] += 1e-4
        try:
            c = np.linalg.solve(G, Xm.T @ yi.reshape(-1))
        except np.linalg.LinAlgError:
            c = np.linalg.lstsq(G, Xm.T @ yi.reshape(-1), rcond=None)[0]
        a_b[bi] = c[0]; b_b[bi] = c[1]
    # evaluate block model
    pred_blk = np.zeros_like(yv)
    for bi, (lo, hi) in enumerate(blocks):
        idx = np.where((t >= lo) & (t < hi))[0]
        if len(idx) == 0: continue
        pred_blk[idx] = (a_b[bi] * x[idx, 0].reshape(len(idx), -1) + b_b[bi])
    mse_blk = float(np.mean((pred_blk - yv) ** 2))

    # ------------------------------------------------------------------
    # 4) choose the design and assemble weights
    # ------------------------------------------------------------------
    # We will implement the block model (exact per-block affine on the
    # centre pixel).  Optionally add a per-block 3x3 spatial correction if
    # it materially helps and channels allow.
    #
    # Channel budget:
    #   conv1 (32): ch0 = ReLU(x) ; ch1 = ReLU(-x) ; ch2.. = hinges
    #   conv2 (64): ch0 = x+M (affine copy, always positive)
    #                ch1 = -x+M (affine copy of -x, always positive)
    #                ch2..(B+1) = s_gate_b(t) + C  (per block, positive)
    #                ch(B+2).. = spare (zeros)
    #   conv3 (64): per block b: 4 channels:
    #                g+ = ReLU( x + M + s_gate_b )        (x on-block)
    #                g- = ReLU( -x + M + s_gate_b )       (-x on-block)
    #                c+ = ReLU( M + s_gate_b )            (const on-block)
    #                (3 per block)  -> 3B <= 64
    #   out (1,64,3,3): weights combining g+, g-, c+ per block:
    #                out = sum_b [ a_b*(g+ - g-) + (b_b/M)*(c+) ] with
    #                appropriate scaling.
    #
    # Details of the gate:
    #   s_gate_b(t) = 0 for t in [lo_b, hi_b-1]
    #                = -K * (hi_b - t) for t < lo_b   (ramp, PL)
    #                = -K * (t - (hi_b-1)) for t > hi_b-1
    #   with K large enough that off-block the pre-activation is negative.
    #   We realise s_gate_b via hinges h_j(t) = max(0, t - j) (j=1..30)
    #   available in conv1 (as ReLU(t/31 - j/31) = h_j(t)/31) plus a
    #   constant (conv2 bias).
    #      s_gate_b(t) = -K/31 * [ max(0, lo_b - t) + max(0, t - (hi_b-1)) ]
    #   max(0, lo_b - t) = lo_b - t + max(0, t - lo_b) = lo_b - t + h_lo(t)
    #   max(0, t - (hi_b-1)) = h_{hi_b-1}(t)
    #   So s_gate_b(t) = -K/31 * [ lo_b - t + h_lo(t) + h_{hi_b-1}(t) ]
    #   The linear term -t must come from the t channel of conv1 (which is
    #   t/31): we need a channel carrying -t/31 (i.e. ReLU(-t/31)=0 since
    #   t>=0!).  Hmm: ReLU kills negative.  So a linear ramp in t with
    #   negative slope cannot be produced by a single ReLU channel.
    #   Workaround: use hinges only, plus constant, and choose the block
    #   structure so that only ONE side needs a ramp: e.g. make the gate
    #   "0 on [lo, hi]" using only the upper ramp h_{hi-1} and a constant
    #   that is 0 for t <= hi-1... but a constant can't depend on t.
    #   Alternative: use TWO hinges and a constant, and note
    #      max(0, lo - t) = lo - t + h_lo(t): the "-t" term is the problem.
    #   Instead use the identity: for t in [0,31], max(0, lo - t) can be
    #   written as a PL function with a knot at lo: it equals lo - t for
    #   t<lo and 0 for t>=lo.  In terms of hinges h_j(t)=max(0,t-j): the
    #   function g(t) = lo - min(t, lo) = lo - t + h_lo(t).  The linear
    #   part -t must come from somewhere: a channel that is exactly -t/31
    #   (negative) is impossible post-ReLU... BUT we can use the
    #   "affine copy" trick in conv1 too!  conv1 ch = -t/31 + M1 (M1 large)
    #   -> post = M1 - t/31 (always positive) -> an exact affine copy of -t.
    #   Then conv2 can build s_gate_b as a linear combination of hinge
    #   channels and this affine copy of -t.
    #   So conv1 channels: ch0 = ReLU(x), ch1 = ReLU(-x), ch2 = affine
    #   copy of -t (i.e. M1 - t/31), ch3.. = hinges h_j(t)/31 (j=1..30).
    #   That's 2 + 1 + 30 = 33 > 32.  Use only the hinges we need:
    #   knots needed: for each block, at lo_b and hi_b-1.  With B blocks
    #   partitioning [0,31], the distinct knots are the interior boundaries
    #   (B-1 of them) plus possibly 30 (for the last block's upper end).
    #   Let's define blocks as [lo_b, hi_b) with hi_b = lo_{b+1}.  Then
    #   s_gate_b needs: ramp down for t<lo_b (knot at lo_b) and ramp up for
    #   t>=hi_b (knot at hi_b).  Knot at lo_b = 0 for the first block -> the
    #   ramp term max(0, 0 - t) = 0 for all t>=0 -> no knot needed (and the
    #   "-t" term cancels: lo - t + h_lo = 0 - t + t = 0 ✓).
    #   For the last block, hi_b = 32 > 31 -> max(0, t-32) = 0 always ->
    #   knot at 32 not needed (and the term -t+h_32 = 0 ✓ if we use the
    #   affine -t copy and h_32=0).
    #   So knots needed: interior boundaries lo_1..lo_{B-1} (B-1 hinges)
    #   plus hi_b for b<B-1 which equals lo_{b+1} (same set) ✓.
    #   So we need B-1 hinge channels + 1 affine(-t) channel + 2 x channels
    #   = B+2 channels in conv1.  With B=16 -> 18 channels ✓ (32 available).
    #   Great, plenty of room.  We could even use B=21 (23 channels) with
    #   3B=63 conv3 channels ✓.
    #
    #   Now, s_gate_b(t) = -K * [ max(0, lo_b - t) + max(0, t - hi_b) ]
    #   (note: for t in [lo_b, hi_b-1] this is 0 ✓; for t<lo_b it is
    #   K*(lo_b - t) > 0; for t>=hi_b it is K*(t-hi_b) > 0.)
    #   We want s_gate_b <= -M' off-block.  So define
    #     s_gate_b(t) = -K * [ max(0, lo_b - t) + max(0, t - hi_b) ]
    #   with K = 40 (so off-block s <= -40 at least 1 step away; for
    #   t = hi_b exactly, s = 0? max(0, t-hi_b)=0 at t=hi_b -> s=0 at
    #   t=hi_b, but t=hi_b belongs to the NEXT block -> the gate for block
    #   b would be active at t=hi_b!  Fix: use hi_b - 1 as the knot:
    #     s_gate_b(t) = -K*[max(0, lo_b - t) + max(0, t - (hi_b-1))]
    #   then at t = hi_b-1 (last step of block) s=0 ✓, at t=hi_b s=-K ✓.
    #   Similarly at t = lo_b - 1: s = -K ✓.
    #   Good.  So knots: for block b: knot at lo_b (for the down-ramp) and
    #   knot at hi_b - 1 (for the up-ramp).  For b=0: lo_0=0 → down-ramp
    #   term = max(0, 0-t)=0 → no knot needed (and no -t term needed).
    #   For b=B-1: hi_b-1 = 31 → up-ramp = max(0,t-31)=0 → no knot.
    #   Interior blocks need both knots: lo_b and hi_b-1 = lo_{b+1}-1.
    #   Distinct knots across blocks: {lo_b : b=1..B-1} ∪ {lo_{b+1}-1 : b=0..B-2}
    #   = up to 2(B-1) hinges.  With B=16 → 30 hinges; conv1 total = 2+1+30 = 33 > 32.  Tight.
    #   Use B=12 → 22 hinges → conv1 uses 25 channels ✓; conv3 uses 36.
    #   Or use B=16 with 30 hinges and drop the affine(-t) channel by
    #   handling the down-ramp differently: max(0, lo-t) = lo - t + h_lo(t)
    #   needs -t.  Alternatively use a different gate formulation: instead
    #   of a symmetric ramp, we can use a *one-sided* gate: make the gate
    #   active on [lo_b, hi_b) by using s_gate_b(t) = -K*max(0, t - (hi_b-1))
    #   plus -K*max(0, lo_b - t).  Both needed.
    #   Alternatively: choose blocks so that the down-ramp is unnecessary:
    #   use a gate that is 0 for t <= hi_b-1 and negative for t > hi_b-1
    #   (one-sided, upper ramp only) — then the gate would be active for all
    #   t <= hi_b-1, i.e. blocks are "prefixes", not a partition.  But we
    #   can use nested prefixes with a telescoping construction!
    #   Think of it as: out = sum_b a_b * gate_b where gate_b is active on
    #   a prefix [0, hi_b-1].  Then the coefficient applied at timestep t
    #   is sum_{b: hi_b-1 >= t} a_b — a cumulative sum!  We can realize
    #   ANY per-timestep coefficient a_t as a cumulative sum of increments
    #   d_b = a_t - a_{t-1}!  Specifically, define gate_b active on
    #   [0, hi_b - 1] with hi_b = b+1 (b=0..30) -> 31 prefix gates, each
    #   needing only an upper ramp (knot at b) -> 31 hinges.  Hmm, still 31.
    #   But with B blocks of consecutive timesteps, we can use the cumulative
    #   trick with B gates: gate_b active on prefix [0, hi_b-1] where the
    #   hi_b are the block ends.  Then coefficient at t = sum_{b: hi_b>t} d_b.
    #   With blocks [0,2),[2,4),...,[30,32): hi = 2,4,...,32 -> 16 gates,
    #   knots at 1,3,5,...,31 (upper ramp knot = hi_b-1) -> 16 hinges, and
    #   NO down-ramp needed (prefix gates are one-sided!).  But then the
    #   coefficient at t=0 is sum of all d_b, and at t=1 also sum of all
    #   (since hi_b >= 2 > 1) — so within a block the coefficient is
    #   constant ✓ exactly what we want (block-constant a_t).
    #   Wait: with prefix gates, the coefficient at t is sum over blocks b
    #   whose prefix covers t: prefix b covers t <= hi_b - 1 = 2b+1.  So
    #   for t in [2k, 2k+1], the covering blocks are those with 2b+1 >= t,
    #   i.e. b >= (t-1)/2 -> b >= k for t=2k, b >= k for t=2k+1 (2b+1>=2k+1
    #   -> b>=k).  So coefficient = sum_{b>=k} d_b = a_k (with a_k =
    #   desired coefficient for block k) if we set d_b = a_b - a_{b+1}...
    #   Let's define: we want coefficient c(t) = a_{block(t)}.  Set
    #   d_b = a_b - a_{b+1} for b=0..B-2, d_{B-1} = a_{B-1}.  Then
    #   sum_{b>=k} d_b = a_k ✓.  And the coefficient applied to gate_b is
    #   d_b (a constant), so out = sum_b d_b * gate_b(x) where gate_b is
    #   active on prefix [0, hi_b-1].  But careful: gate_b is active on a
    #   PREFIX, so at timestep t, ALL gates with hi_b-1 >= t are active,
    #   and their sum gives c(t) ✓.
    #   And the additive bias b_t: similarly with constant gates (prefix
    #   gates carrying a constant) and coefficients e_b = b_block - b_next.
    #   So we need per gate b: 3 channels in conv3 (+x, -x, const) and
    #   the prefix gate offset s_b(t) = -K*max(0, t - (hi_b-1)) which needs
    #   ONE hinge at knot (hi_b-1) ✓.  So conv1 needs: 2 (x, -x) + 1
    #   (affine -t copy, needed to build the ramp: s_b(t) = -K*(t - k)/31
    #   for t>k = -K/31*(t - k) = -K/31 * t + K*k/31 -> needs -t copy and
    #   constant) + B hinges (one per gate knot) = B+3 channels.
    #   With B=16: 19 channels ✓.  conv3: 3B = 48 ✓ (fits in 64).
    #   conv2: needs x copy, -x copy, per-gate s_b channels (B), plus the
    #   hinge channels and -t copy must be available in conv2 to build
    #   s_b -> conv2 can build s_b from conv1's hinge and -t channels
    #   directly (linear combination) ✓.  conv2 channels: 2 + B = 18 ✓.
    #
    #   Excellent — this is clean and exact.  Let me now write it out.
    #
    #   Numerical care: conv1's affine copy of -t: pre = -t/31 + M1 must be
    #   > 0 for all t: M1 = 2 suffices (t/31 <= 1).  But we also want the
    #   hinge channels to be exact: hinge j: pre = t/31 - j/31 -> ReLU ->
    #   max(0,t-j)/31 ✓ (exact, no clipping issue).
    #   conv2's s_b channel: s_b(t) + C where C must make it positive:
    #   s_b(t) ranges from 0 (t <= k) down to -K*(31-k)/31 ≈ -K.  So use
    #   C = K + 1 -> range [1, K+1] > 0 ✓.  Then conv3 computes:
    #       g+ : pre = (x + M) + (s_b + C) - C - M ... careful: we want
    #             pre = x + M + s_b(t) where M = 40 (so x+M>0 on-block).
    #             conv2 ch_x = x + M (affine copy of x, always positive:
    #             conv1 ch0 = ReLU(x) loses negatives!  We need the SIGNED
    #             x.  Use the affine-copy trick at conv1 as well:
    #             conv1 ch_x = x + M1 (signed copy, always positive) ->
    #             post = x + M1 exactly.  Similarly conv1 ch_nx = -x + M1.
    #             Then conv2 ch_x = (x + M1) (identity) and conv2 ch_nx = -x + M1.
    #             conv3 g+ : pre = (x + M1) + (s_b + C) - C  = x + M1 + s_b
    #               (with s_b = 0 on-block -> x + M1 > 0 ✓; off-block
    #                s_b <= -K -> x + M1 - K < 0 if K > M1 + max|x| ✓)
    #             -> ReLU kills off-block ✓ and keeps x + M1 on-block.
    #             Then out must subtract M1: out = sum_b d_b*(g+_b - g-_b)
    #             + ... but the M1 offset appears only on-block (since the
    #             channel is 0 off-block) -> we need a per-gate constant
    #             channel that is M1 on-block and 0 off-block:
    #             const_b: pre = M1 + s_b(t) -> on-block M1 > 0 ✓, off-block
    #             M1 - K < 0 ✓ -> ReLU gives M1 on-block, 0 off-block ✓.
    #             Then out contribution: d_b*(g+_b - g-_b - const_b) + e_b*const_b
    #             = d_b*(x on-block) + e_b*const_b.
    #             Wait: g+_b - g-_b = (x + M1) - (-x + M1) = 2x on-block.
    #             So use (g+_b - g-_b)/2 - const_b -> x on-block ✓.
    #             So out = sum_b [ d_b*(0.5*(g+_b - g-_b) - const_b) + e_b*const_b ]
    #                 = sum_b d_b*x_block + (e_b - d_b)*const_b.
    #             Since const_b = M1 on-block, the additive term is
    #             (e_b - d_b)*M1 on-block.  We want b_block on-block:
    #             set e_b = d_b + b_block/M1.
    #             Hmm, but the additive term must be t-dependent with the
    #             same prefix structure: const_b is a prefix gate (active on
    #             [0, hi_b-1]) so its coefficient (e_b - d_b) contributes
    #             cumulatively: at t, additive = sum_{b covering t} (e_b-d_b)*M1
    #             = M1 * sum_{b>=k} (e_b - d_b) = M1 * (E_k - D_k) where
    #             E_k = sum_{b>=k} e_b, D_k = sum_{b>=k} d_b = a_k.
    #             We want the additive at t to be b_k.  So we need
    #             M1*(E_k - D_k) = b_k -> E_k = D_k + b_k/M1 ->
    #             e_b = (E_b - E_{b+1}) = (D_b - D_{b+1}) + (b_b - b_{b+1})/M1
    #                 = d_b + (b_b - b_{b+1})/M1.
    #             Then M1*(e_b - d_b) = b_b - b_{b+1} ✓ and cumulative sum
    #             gives b_k ✓ (with b_{B} := 0 for the last block).
    #             Great.
    #
    #   Now the values: M1 (offset for signed x copies) must exceed max|x|
    #   so that x + M1 > 0 always and -x + M1 > 0 always.  x_t values:
    #   let's compute max|x| from data, set M1 = max|x| + 2.
    #   K (gate strength) must satisfy: off-block, x + M1 - K < 0 for all x
    #   -> K > M1 + max|x|.  But note s_b(t) = -K*(t-k)/31 for t>k; the
    #   smallest |s_b| off-block is at t = k+1: s = -K/31.  So we need
    #   K/31 > M1 + max|x| -> K = 31*(M1 + max|x| + 1).  That's large
    #   (e.g. 31*10 = 310) but weights must be <= 1000 in abs — the
    #   weights in conv1/conv2 that multiply the hinge channel are
    #   K/31 ≈ 10 ✓ fine.  But the *value* of s_b in conv2 is -K*(t-k)/31
    #   which can be -310 — that's an activation value, not a weight; the
    #   constraint is on weights (abs <= 1000) and finiteness.  Fine.
    #   However: conv3's pre-activation includes s_b + C with C = K+1 ≈ 311,
    #   and conv2's weights multiply conv1's hinge channel (value up to
    #   (31-k)/31 ≈ 1) by K/31 ≈ 10 -> value ≈ 10*1 = 10?? Let's redo:
    #   s_b(t) = -K*(t-k)/31 where t-k <= 31 -> s_b >= -K.  In conv2, the
    #   s_b channel = sum_j c_j * hinge_j + c_neg * (-t copy) + bias.
    #   hinge_j value = (t-j)/31 <= 1.  -t copy value = M1 - t/31 ∈ [M1-1, M1].
    #   To get s_b(t) = -K*(t-k)/31 we can write it as:
    #       -K*(t-k)/31 = -(K/31)*t + (K*k/31)
    #   The -t part: conv1's affine copy of -t is M1 - t/31; multiply by
    #   K -> gives K*M1 - K*t/31 ✓ (weight K ≈ 310 ≤ 1000 ✓). Then add
    #   bias to fix the constant: bias = K*k/31 - K*M1 + C.
    #   So conv2 weight on the -t channel = K (≤1000 ✓). Let's pick K
    #   moderate: we need K/31 > M1 + max|x| -> with M1 = max|x|+2,
    #   K/31 > 2*max|x| + 2.  If max|x| ≈ 4 -> K/31 > 10 -> K > 310.
    #   Use K = 31*(2*max|x| + 4) ≈ 372 (≤1000 ✓).  Also the conv2
    #   weight on the -t channel is K ≈ 372 ✓ < 1000.
    #   Alternatively, reduce: instead of the ramp being -K*(t-k)/31, use
    #   -K*(t-k)/31 but we only need it to be ≤ -(M1+max|x|) off-block,
    #   and the *smallest* off-block magnitude is at t=k+1: -K/31.  So
    #   K/31 ≥ M1+max|x|+1 → K ≥ 31*(M1+max|x|+1). With max|x|=4,
    #   M1=6: K ≥ 31*11 = 341. OK.
    #   Alternatively, use a steeper ramp: s_b(t) = -K*(t-k) (not divided
    #   by 31) -> then at t=k+1, s = -K -> need K > M1+max|x| ≈ 10 ✓
    #   tiny!  Because we can use the raw t value... but conv1's t channel
    #   is t/31 (per the architecture: "concat x_t with t/31 broadcast").
    #   Hmm, the t channel value is t/31 ∈ [0,1].  The affine copy of -t/31
    #   is M1 - t/31.  To build a ramp of slope -K per unit t we need the
    #   t value itself: -(t) = -31*(t/31).  So the weight on the -t copy
    #   channel would be 31*K.  With K = 12 (need > M1+max|x| ≈ 10) ->
    #   31*12 = 372 ≤ 1000 ✓.  Same thing.  Fine, use K such that
    #   31*K > M1 + max|x| + 1 and 31*K ≤ 1000 → K ≤ 32.  With
    #   max|x| ≈ 4, M1 = 6: need 31K > 11 → K = 1 suffices?! Let's
    #   recompute: s_b(t) = -K*(t-k) for t>k (slope -K per unit t).
    #   Off-block at t=k+1: s = -K. Need x + M1 + s < 0 → K > M1 + max|x|.
    #   With M1 = max|x|+2 → K > 2*max|x|+2 ≈ 10. Then the conv2
    #   weight on the -t copy = 31*K ≈ 310 ✓ < 1000.
    #   Good: K = 2*max|x| + 4 (≈12), conv2 weight = 31K ≈ 372 ✓.
    #
    #   Hmm, but wait: the -t copy channel in conv1 is (M1 - t/31) and we
    #   multiply by 31K to get 31K*M1 - K*t.  Then the constant part
    #   31K*M1 must be cancelled by the bias: bias = -31K*M1 + K*k + C.
    #   These are large numbers (31*12*6 = 2232) but that's fine (bias is a
    #   parameter; the constraint says abs <= 1000 for parameters!  Hmm:
    #   "Preserve keys/shapes, finite float32, absolute value<=1000" — this
    #   likely applies to all returned weights.  Bias 2232 > 1000 → violation.
    #   Let's avoid large biases: instead of the affine -t copy with a large
    #   offset M1, use a *hinge-based* ramp only: s_b(t) = -K*(t-k) for
    #   t>k can be written as -K*(t-k) = -K*[(t-k)] and (t-k) for t>k is
    #   the hinge h_k(t) = max(0,t-k) ✓!!  So s_b(t) = -K * h_k(t) where
    #   h_k(t) = max(0, t-k) is exactly a hinge channel in conv1 (value
    #   (t-k)/31).  So s_b(t) = -31*K*h_k_value.  conv2 weight on the
    #   hinge channel = -31K ≈ -372 ✓ < 1000, and bias = C (small) ✓.
    #   And for the down-ramp (t < lo_b) we don't need it because we use
    #   PREFIX gates (active for t <= k) ✓✓.
    #
    #   So: prefix gate b is active for t <= k_b (k_b = hi_b - 1 = last
    #   timestep of block b) and inactive for t > k_b.
    #   s_b(t) = -31*K*max(0, t - k_b)  (0 for t <= k_b) ✓.
    #   conv1 hinge channel for knot k_b: pre = t/31 - k_b/31 -> ReLU ->
    #   max(0,t-k_b)/31 ✓ (for k_b in 1..30; for k_b=0 the hinge is
    #   identically 0 → skip; for k_b=31 identically 0 → skip).
    #   conv2 s_b channel = -31K * hinge_{k_b} + C  (C = 31K*1 + 1 to keep
    #   positive: max hinge value = (31-k_b)/31 <= 1 → -31K*1 + C; choose
    #   C = 31K + 1 → value ∈ [1, 31K+1] ✓ positive).
    #   conv3: pre for g+_b = (x + M1) + (s_b + C) - C = x + M1 + s_b
    #     where s_b = -31K*hinge_value ∈ [0, -31K*(31-k_b)/31].
    #     On-block (t<=k_b): s_b=0 → pre = x + M1 > 0 ✓.
    #     Off-block (t>k_b): s_b = -K*(t-k_b) ≤ -K → pre = x + M1 - K*(t-k_b)
    #       ≤ max|x| + M1 - K < 0 if K > M1 + max|x| ✓.
    #     But careful: conv3's weights: on the conv2 s_b channel we need
    #     weight +1, on the conv2 x channel weight +1, and a bias of -C.
    #     The bias -C ≈ -373 → |bias| > ... constraint is 1000 ✓ fine.
    #     Hmm, but we also need to subtract C: bias = -C. And the value of
    #     the s_b channel is s_b + C so adding it with weight 1 and bias -C
    #     gives s_b ✓.
    #     For g-_b: pre = (-x + M1) + (s_b + C) - C = -x + M1 + s_b ✓.
    #     For const_b: pre = M1 + s_b → needs a channel in conv2 carrying
    #       the constant M1 (any channel, e.g. reuse the x channel? no, we
    #       need a pure constant channel).  conv2 can output a constant:
    #       e.g. conv2 ch_const = bias 1 (all weights zero, bias 1) -> ReLU
    #       -> 1 ✓ (constant channel).  Then conv3 const_b pre = M1*const
    #       + s_b + C - C = M1 + s_b ✓ (using conv2's const channel with
    #       weight M1, and the s_b channel with weight 1, bias -C).
    #       Wait: M1 + s_b: on-block = M1 > 0 ✓; off-block = M1 - K*(t-k)
    #       < 0 ✓ (K > M1 + max|x| > M1 ✓).
    #     Then out: out = sum_b [ d_b*(0.5*(g+_b - g-_b) - const_b) + e_b*const_b ]
    #       = sum_b d_b * x_block + sum_b (e_b - d_b)*const_b
    #       with const_b = M1 on-block → additive = M1*(e_b-d_b) cumulative.
    #       As derived: e_b = d_b + (b_b - b_{b+1})/M1.
    #       Hmm wait, but the *cumulative* structure: const_b is a prefix
    #       gate (active for t <= k_b) so at t the sum over covering b
    #       gives M1*sum_{b>=k}(e_b-d_b).  We want b_k.  With
    #       e_b - d_b = (b_b - b_{b+1})/M1 → sum_{b>=k} = (b_k - b_{B})/M1
    #       = b_k/M1 (taking b_B = 0) ✓ → additive = b_k ✓.
    #       Similarly the x term: sum over covering b of d_b*0.5*(g+-g-)
    #       where g+-g- = 2x on-block → sum_{b>=k} d_b * x = a_k x ✓
    #       (with d_b = a_b - a_{b+1}, a_B = 0) ✓.
    #
    #   PERFECT. Now let's also handle the possibility of adding a spatial
    #   3x3 correction.  Given channel budget: conv3 has 64 channels; we
    #   use 3 per block.  With B=16 blocks: 48 channels, 16 spare → we can
    #   add 1 extra gated spatial shift per block (16 channels) if we use
    #   +x and -x for that shift (2 per block = 32 > 16).  Alternatively
    #   use B=10 blocks (30 channels) + 2 extra shifts (40) = 70 > 64.
    #   Hmm.  Let's evaluate how much the 3x3 helps; if the gain is
    #   substantial we could restructure: use only +x gates for extra
    #   shifts and rely on... no, sign matters.
    #   Let's first compute mse_1x1 vs mse_3x3 and mse_blk, then decide.
    #   Note: the block model with B=16 (2 timesteps per block) approximates
    #   the per-timestep 1x1 model; mse_blk should be close to mse_1x1.
    #   If mse_3x3 << mse_1x1, we should invest in spatial terms.
    #
    #   A cheaper way to get spatial info: add a *global* (non-gated)
    #   3x3-filtered x term with a constant coefficient (t-independent) —
    #   but the optimal coefficient is t-dependent, so limited gain.
    #   Alternative: use the *t-independent* structure: the optimal
    #   per-timestep 3x3 filter is roughly a_t * (I - sqrt(abar)*W) where W
    #   is a Wiener-like smoothing operator.  The smoothing part is
    #   t-dependent too.
    #   Compromise: implement per-block gated center-pixel term (exact) plus
    #   per-block gated *neighbour-average* term using 2 extra channels per
    #   block (the + and - of the neighbour average).  With B=12: 3*12 + 2*12
    #   = 60 ≤ 64 ✓.  conv1 needs: 2 x-copies + 1 const? (const channel
    #   can be created in conv2 with zero weights and bias 1 ✓) + B hinges
    #   = 2 + 12 = 14 ✓.  conv2 needs: x copy, -x copy, neighbour-average
    #   copy, neighbour-average negative copy, B s_b channels, 1 const
    #   channel = 4 + 12 + 1 = 17 ✓.
    #   The neighbour average: we need a conv1 channel that computes the
    #   average of the 3x3 neighbourhood of x (a smoothing) as an affine
    #   copy (always positive): conv1 ch = (1/9)*sum of 3x3 taps of x
    #   + M1 → post = M1 + smooth(x) ✓ (positive since |smooth| ≤ max|x|).
    #   Then conv2 can carry it, and conv3 forms the gated version.
    #   But careful: the "signed copy" trick requires the value to be
    #   positive; smooth(x) ∈ [-max|x|, max|x|] ✓.
    #   However, the *negative* branch of the neighbour average: -smooth(x)
    #   → conv1 channel = -(1/9)*sum taps + M1 ✓.
    #   So conv1 channels: x copy, -x copy, smooth copy, -smooth copy,
    #   plus B hinges = 4 + B.  With B=12 → 16 ✓.
    #   Then per block, conv3 channels: g+ (center), g- (center), gS+ (smooth),
    #   gS- (smooth), const = 5 per block → 5B ≤ 64 → B ≤ 12 ✓.
    #   With B=12 blocks over 32 timesteps (2-3 timesteps each) the
    #   coefficient quantization error is small.
    #   Let's decide the final design after computing the MSEs:
    #     - if mse_3x3 (per-timestep 3x3) is much better than mse_1x1,
    #       use the smooth-term design with B=12 (per-block 3x3-ish fit:
    #       fit y ~ a*x + c*smooth(x) + b per block).
    #     - else use the simpler center-only design with B=16.
    #   I'll implement the smooth-term design generally (it includes the
    #   center-only case by setting c=0), with B chosen as large as
    #   possible given 5B<=64 → B=12.
    #   Actually, we can also fit per-block a 3x3 filter properly and
    #   implement 4 of the 9 shifts?  Too complex.  Use center + smooth.
    #
    # ------------------------------------------------------------------
    # 5) assemble
    # ------------------------------------------------------------------
    # smooth operator: 3x3 box average (zero padding)
    def smooth_of(xarr):
        xp = np.pad(xarr, ((0, 0), (1, 1), (1, 1)))
        acc = np.zeros_like(xarr)
        for i in range(3):
            for j in range(3):
                acc += xp[:, i:i + H, j:j + Wd]
        return acc / 9.0

    sm = smooth_of(x[:, 0])          # (N,H,W)
    # per-block fits: y ~ a*x + c*sm + b
    def fit_block(idx):
        if len(idx) == 0:
            return None
        xi = x[idx, 0].reshape(len(idx), -1)
        si = sm[idx].reshape(len(idx), -1)
        yi = yv[idx].reshape(-1)
        Xm = np.stack([xi, si, np.ones(len(idx))], axis=1)
        G = Xm.T @ Xm
        G[np.diag_indices(3)] += 1e-4
        rhs = Xm.T @ yi
        try:
            c = np.linalg.solve(G, rhs)
        except np.linalg.LinAlgError:
            c = np.linalg.lstsq(G, rhs, rcond=None)[0]
        res = yi - Xm @ c
        return c, float(np.mean(res * res))

    # choose block count
    B = 12
    edges = np.linspace(0, T, B + 1).astype(int)
    blocks = [(int(edges[i]), int(edges[i + 1])) for i in range(B)]
    abc = []
    for (lo, hi) in blocks:
        idx = np.where((t >= lo) & (t < hi))[0]
        r = fit_block(idx)
        if r is None:
            abc.append((0.0, 0.0, 0.0))
        else:
            abc.append(r[0])
    # evaluate block model (center + smooth)
    pred = np.zeros_like(yv)
    for bi, (lo, hi) in enumerate(blocks):
        idx = np.where((t >= lo) & (t < hi))[0]
        if len(idx) == 0:
            continue
        a, c, b = abc[bi]
        pred[idx] = (a * x[idx, 0].reshape(len(idx), -1)
                     + c * sm[idx].reshape(len(idx), -1) + b)
    mse_blk2 = float(np.mean((pred - yv) ** 2))

    # reference: per-timestep center+smooth fit (upper bound of our class)
    pred_ts = np.zeros_like(yv)
    for tt in range(T):
        idx = np.where(t == tt)[0]
        if len(idx) == 0:
            continue
        xi = x[idx, 0].reshape(len(idx), -1)
        si = sm[idx].reshape(len(idx), -1)
        yi = yv[idx].reshape(-1)
        Xm = np.stack([xi, si, np.ones(len(idx))], axis=1)
        G = Xm.T @ Xm; G[np.diag_indices(3)] += 1e-4
        try:
            c = np.linalg.solve(G, Xm.T @ yi)
        except np.linalg.LinAlgError:
            c = np.linalg.lstsq(G, Xm.T @ yi, rcond=None)[0]
        pred_ts[idx] = Xm @ c
    mse_ts2 = float(np.mean((pred_ts - yv) ** 2))

    # ------------------------------------------------------------------
    # 6) build weights
    # ------------------------------------------------------------------
    maxabs_x = float(np.max(np.abs(x)))
    M1 = maxabs_x + 2.0
    K = 2.0 * maxabs_x + 6.0            # gate ramp strength (per unit t)
    C = 31.0 * K + 1.0                 # keeps s_b channels positive

    # conv1 layout
    #  ch0: x + M1                 (signed copy of x, always positive)
    #  ch1: -x + M1
    #  ch2: smooth(x) + M1
    #  ch3: -smooth(x) + M1
    #  ch4..: hinge channels for knots k_b (one per block, k_b = hi_b-1)
    # conv1 has 32 channels -> B <= 28 ✓ (B=12 -> 16 channels used)
    c1w = np.zeros((32, 2, 3, 3)); c1b = np.zeros(32)
    # x copy: identity tap on channel 0 (x), bias M1
    c1w[0, 0, 1, 1] = 1.0; c1b[0] = M1
    c1w[1, 0, 1, 1] = -1.0; c1b[1] = M1
    # smooth copies: 1/9 on all 3x3 taps of channel 0
    c1w[2, 0, :, :] = 1.0 / 9.0; c1b[2] = M1
    c1w[3, 0, :, :] = -1.0 / 9.0; c1b[3] = M1
    # hinge channels: pre = t/31 - k/31  -> weight on channel 1 (t) = 1/31
    #   at the center tap, bias = -k/31
    knots = []
    for bi, (lo, hi) in enumerate(blocks):
        k = hi - 1
        knots.append(k)
    hinge_ch = {}
    ch = 4
    for bi, k in enumerate(knots):
        if k <= 0 or k >= T:
            hinge_ch[bi] = None
            continue
        c1w[ch, 1, 1, 1] = 1.0 / 31.0
        c1b[ch] = -k / 31.0
        hinge_ch[bi] = ch
        ch += 1
    # remaining conv1 channels: zero (output 0)

    # conv2 layout (64 channels)
    #  ch0: x + M1              (from conv1 ch0)
    #  ch1: -x + M1             (from conv1 ch1)
    #  ch2: smooth + M1        (from conv1 ch2)
    #  ch3: -smooth + M1       (from conv1 ch3)
    #  ch4: constant 1         (zero weights, bias 1)
    #  ch5..: s_b channels: -31K*hinge_{k_b} + C
    c2w = np.zeros((64, 32, 3, 3)); c2b = np.zeros(64)
    c2w[0, 0, 1, 1] = 1.0
    c2w[1, 1, 1, 1] = 1.0
    c2w[2, 2, 1, 1] = 1.0
    c2w[3, 3, 1, 1] = 1.0
    c2b[4] = 1.0
    s_ch = {}
    for bi in range(B):
        c = 5 + bi
        s_ch[bi] = c
        hc = hinge_ch[bi]
        if hc is not None:
            c2w[c, hc, 1, 1] = -31.0 * K
        c2b[c] = C
    # conv3 layout (64 channels): 5 per block
    #  per block b:
    #    5b+0 : g+   pre = (x+M1) + (s_b+C) - C
    #    5b+1 : g-   pre = (-x+M1) + (s_b+C) - C
    #    5b+2 : gS+  pre = (sm+M1) + (s_b+C) - C
    #    5b+3 : gS-  pre = (-sm+M1) + (s_b+C) - C
    #    5b+4 : const pre = M1*1 + (s_b+C) - C
    c3w = np.zeros((64, 64, 3, 3)); c3b = np.zeros(64)
    for bi in range(B):
        base = 5 * bi
        sc = s_ch[bi]
        # g+
        c3w[base + 0, 0, 1, 1] = 1.0
        c3w[base + 0, sc, 1, 1] = 1.0
        c3b[base + 0] = -C
        # g-
        c3w[base + 1, 1, 1, 1] = 1.0
        c3w[base + 1, sc, 1, 1] = 1.0
        c3b[base + 1] = -C
        # gS+
        c3w[base + 2, 2, 1, 1] = 1.0
        c3w[base + 2, sc, 1, 1] = 1.0
        c3b[base + 2] = -C
        # gS-
        c3w[base + 3, 3, 1, 1] = 1.0
        c3w[base + 3, sc, 1, 1] = 1.0
        c3b[base + 3] = -C
        # const
        c3w[base + 4, 4, 1, 1] = M1
        c3w[base + 4, sc, 1, 1] = 1.0
        c3b[base + 4] = -C
    # out layer: combine
    #   out = sum_b [ d_b*(0.5*(g+ - g-) - const) + e_b*const ]
    #        + sum_b [ f_b*(0.5*(gS+ - gS-) - const) ]
    #   where d_b = a_b - a_{b+1}, f_b = c_b - c_{b+1}, e_b = d_b + (b_b - b_{b+1})/M1
    ow = np.zeros((1, 64, 3, 3)); ob = np.zeros(1)
    A_ = np.array([abc[i][0] for i in range(B)])
    Cc_ = np.array([abc[i][1] for i in range(B)])
    Bb_ = np.array([abc[i][2] for i in range(B)])
    A_.next = 0.0; Cc_.next = 0.0; Bb_.next = 0.0
    for bi in range(B):
        base = 5 * bi
        d = A_[bi] - (A_[bi + 1] if bi + 1 < B else 0.0)
        f = Cc_[bi] - (Cc_[bi + 1] if bi + 1 < B else 0.0)
        bb = Bb_[bi] - (Bb_[bi + 1] if bi + 1 < B else 0.0)
        e = d + bb / M1
        ow[0, base + 0, 1, 1] += 0.5 * d
        ow[0, base + 1, 1, 1] += -0.5 * d
        ow[0, base + 2, 1, 1] += 0.5 * f
        ow[0, base + 3, 1, 1] += -0.5 * f
        ow[0, base + 4, 1, 1] += (e - d)   # = bb/M1
    ob[0] = 0.0

    # ------------------------------------------------------------------
    # 7) verify numerically with an explicit forward pass
    # ------------------------------------------------------------------
    def relu(z):
        return np.maximum(z, 0.0)

    def forward(weights, xin, tt):
        # build input: x and t/31 broadcast
        Nn = xin.shape[0]
        tb = (tt.astype(np.float64) / 31.0).reshape(Nn, 1, 1, 1)
        inp = np.concatenate([xin, np.broadcast_to(tb, (Nn, 2, H, Wd))], axis=1)
        h1 = relu(conv3x3(inp, weights['conv1.weight'], weights['conv1.bias']))
        h2 = relu(conv3x3(h1, weights['conv2.weight'], weights['conv2.bias']))
        h3 = relu(conv3x3(h2, weights['conv3.weight'], weights['conv3.bias']))
        o = conv3x3(h3, weights['out.weight'], weights['out.bias'])
        return o

    newW = {
        'conv1.weight': c1w, 'conv1.bias': c1b,
        'conv2.weight': c2w, 'conv2.bias': c2b,
        'conv3.weight': c3w, 'conv3.bias': c3b,
        'out.weight': ow, 'out.bias': ob,
    }
    # forward on a subset (to keep memory/time reasonable)
    sub = np.arange(0, N, 4)
    pr = forward(newW, x[sub], t[sub])
    mse_check = float(np.mean((pr - y[sub]) ** 2))

    # ------------------------------------------------------------------
    # 8) finalize
    # ------------------------------------------------------------------
    out_w = {}
    for k, v in w.items():
        kk = key_of(k)
        if kk in newW:
            arr = np.asarray(newW[kk], dtype=np.float32)
            out_w[k] = arr
        else:
            out_w[k] = np.asarray(v, dtype=np.float32)
    # safety: clip magnitudes, ensure finite
    for k in out_w:
        a = np.asarray(out_w[k], dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=100.0, neginf=-100.0)
        a = np.clip(a, -1000.0, 1000.0)
        out_w[k] = a

    diagnostics = {
        'mse_model_center_only_per_t': round(mse_1x1, 5),
        'mse_model_3x3_per_t': round(mse_3x3, 5),
        'mse_model_center_smooth_per_t': round(mse_ts2, 5),
        'mse_model_block_center_smooth': round(mse_blk2, 5),
        'mse_forward_check_subset': round(mse_check, 5),
        'n_blocks': B,
        'block_edges': [[int(a), int(b)] for (a, b) in blocks],
        'coeffs_a': [round(float(v), 4) for v in A_],
        'coeffs_c': [round(float(v), 4) for v in Cc_],
        'coeffs_b': [round(float(v), 4) for v in Bb_],
        'M1': round(M1, 3), 'K': round(K, 3),
    }
    return {'weights': out_w, 'diagnostics': diagnostics}
