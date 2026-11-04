import numpy as np


def edit(w, f):
    # ------------------------------------------------------------------
    # Closed-form design of a fixed-architecture epsilon predictor:
    #   eps_hat[i,j] = c(t) * x_t[i,j],  c(t) = sqrt(1-abar)/(abar*s0 + 1-abar)
    # implemented exactly with the given conv/ReLU stack.
    # ------------------------------------------------------------------
    x = np.asarray(f["x"], dtype=np.float64)            # [N,1,16,16]
    t = np.asarray(f["t"], dtype=np.int64)              # [N]
    clean = np.asarray(f["clean"], dtype=np.float64)    # [N,1,16,16]
    abar = np.asarray(f["alphabar"], dtype=np.float64)  # [32]

    N = x.shape[0]
    H = W_ = x.shape[2]
    T = abar.shape[0]

    # ---- per-pixel variance of clean images (prior variance sigma0^2) ----
    s0 = float(clean.reshape(N, -1).var(axis=0, ddof=1).mean())

    # ---- optimal per-timestep linear gain ------------------------------
    a = np.clip(abar, 1e-8, 1.0 - 1e-8)
    c_k = np.sqrt(1.0 - a) / (a * s0 + (1.0 - a))
    c_k = np.clip(c_k, -50.0, 50.0)

    # ---- time embedding: s = t/31, one-hot hat functions in s -----------
    # g_k(s) = 1 iff t == k, 0 otherwise, computed exactly by a
    # piecewise-linear function of s (hat functions on the grid s_k=k/31).
    s = (t.astype(np.float64) / 31.0).reshape(N, 1)
    sk = (np.arange(T, dtype=np.float64) / 31.0).reshape(1, T)
    # hat_k(s) = max(0, 1 - |s - s_k| * 31)  -> exactly 1 at s=s_k, 0 elsewhere
    hat = np.maximum(0.0, 1.0 - np.abs(s - sk) * 31.0)
    G = hat  # [N,T], column k is the one-hot gate for timestep k

    # ---- helper: cross-correlation conv (PyTorch layout) ---------------
    def conv2d(inp, weight, bias, pad=1):
        # inp [N,C,H,W], weight [O,C,kh,kw], bias [O]
        Nn, C, Hh, Ww = inp.shape
        O, _, kh, kw = weight.shape
        if pad:
            inp = np.pad(inp, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        Hp, Wp = inp.shape[2], inp.shape[3]
        cols = np.empty((Nn, C, kh, kw, Hp - kh + 1, Wp - kw + 1), dtype=np.float64)
        for i in range(kh):
            for j in range(kw):
                cols[:, :, i, j, :, :] = inp[:, :, i:i + Hp - kh + 1, j:j + Wp - kw + 1]
        out = np.einsum('nchwkl,ockl->nohw', cols, weight)
        out += bias.reshape(1, O, 1, 1)
        return out

    def id_kernel(Cin, Cout):
        k = np.zeros((Cout, Cin, 3, 3), dtype=np.float64)
        for o in range(Cout):
            k[o, o % Cin, 1, 1] = 1.0
        return k

    # ==================================================================
    # conv1 : 2 -> 32
    #   channels 0..15 : +x_t        (center tap on input channel 0)
    #   channels 16..31: -x_t        (center tap on input channel 0)
    # ==================================================================
    c1w = np.zeros((32, 2, 3, 3), dtype=np.float64)
    for o in range(16):
        c1w[o, 0, 1, 1] = 1.0
        c1w[o + 16, 0, 1, 1] = -1.0
    c1b = np.zeros(32, dtype=np.float64)

    # ==================================================================
    # conv2 : 32 -> 64   (ReLU pair trick)
    #   out ch 0..31  : ReLU( x + beta_k * s )
    #   out ch 32..63 : ReLU(-x - beta_k * s )
    #   s is produced inside conv2 from the conv1 x-channels?  No:
    #   we need s as a spatial constant.  Instead we inject s in conv1 is
    #   impossible (all 32 channels used).  => use a dedicated pair of
    #   channels in conv2 fed by the *bias* trick: a conv bias is constant
    #   per channel, so we can encode s via conv1 bias?  No, bias is
    #   constant over examples.
    #
    #   Solution: conv1 channels 0..15 hold +x, 16..31 hold -x, and we
    #   obtain s inside conv2 by using a 3x3 *all-ones* kernel on the
    #   constant-in-space input channel 1 (t/31).  Zero padding perturbs
    #   border pixels, so instead we use a center tap on channel 1:
    #   conv1 must expose channel 1 -> we replace 4 x-channels by s-ch.
    # ==================================================================
    # Revised conv1 layout (32 channels):
    #   0..13 : +x_t
    #   14..27: -x_t
    #   28    : +s   (center tap on input channel 1)
    #   29    : -s
    #   30,31 : +x_t (extra copies, unused downstream)
    c1w = np.zeros((32, 2, 3, 3), dtype=np.float64)
    for o in range(14):
        c1w[o, 0, 1, 1] = 1.0
    for o in range(14, 28):
        c1w[o, 0, 1, 1] = -1.0
    c1w[28, 1, 1, 1] = 1.0
    c1w[29, 1, 1, 1] = -1.0
    c1w[30, 0, 1, 1] = 1.0
    c1w[31, 0, 1, 1] = -1.0
    c1b = np.zeros(32, dtype=np.float64)

    # conv2 weights: build z_k = x + beta_k * s  (pre-activation)
    #   input channels: 0..13 = ReLU(x), 14..27 = ReLU(-x),
    #                   28 = ReLU(s), 29 = ReLU(-s)
    #   out ch k (0..31):  sum_{i<14} w*ReLU(x) + sum_{i>=14} w*ReLU(-x)
    #                      + w28*ReLU(s) + w29*ReLU(-s)
    #   to obtain x (not x^+) we need both signs: use two output channels
    #   per k:  A_k = ReLU(x + beta_k s), B_k = ReLU(-x - beta_k s)
    #   A_k - B_k = x + beta_k s  (exact, since ReLU(z)-ReLU(-z)=z)
    #   => conv2 has 64 channels: 32 A's (0..31) and 32 B's (32..63).
    #   beta_k must be distinct per k so that conv3 can select.
    betas = np.linspace(-1.0, 1.0, 32)
    c2w = np.zeros((64, 32, 3, 3), dtype=np.float64)
    c2b = np.zeros(64, dtype=np.float64)
    for k in range(32):
        # A_k = +x + beta_k * s
        for i in range(14):
            c2w[k, i, 1, 1] = 1.0
        for i in range(14, 28):
            c2w[k, i, 1, 1] = -1.0
        c2w[k, 28, 1, 1] = betas[k]
        c2w[k, 29, 1, 1] = -betas[k]
        # B_k = -x - beta_k * s
        o = 32 + k
        for i in range(14):
            c2w[o, i, 1, 1] = -1.0
        for i in range(14, 28):
            c2w[o, i, 1, 1] = 1.0
        c2w[o, 28, 1, 1] = -betas[k]
        c2w[o, 29, 1, 1] = betas[k]

    # ==================================================================
    # conv3 : 64 -> 64  (ReLU pair trick, gating with one-hot in s)
    #   conv2 outputs: ch k = ReLU(A_k) where A_k = x + beta_k s
    #                  ch 32+k = ReLU(B_k) where B_k = -(x + beta_k s)
    #   ReLU(A_k) - ReLU(B_k) = A_k  (exact)
    #   We want, per k, a channel = g_k(s) * A_k  (gating in s).
    #   g_k(s) is a hat function of s: piecewise linear with breakpoints.
    #   Implement g_k(s) via ReLU of affine functions of s:
    #     hat_k(s) = ReLU(1 - 31|s - s_k|) = ReLU(1 + 31 s_k - 31 s)
    #                                   + ReLU(1 - 31 s_k - 31 s) - 1
    #   (for s in [0,1], s_k = k/31: 1 - 31|s-s_k| >= 0 exactly at s=s_k)
    #   So g_k(s) = ReLU(a_k + 31 s) + ReLU(b_k - 31 s) - 1 with
    #     a_k = 1 + 31 s_k, b_k = 1 - 31 s_k.
    #   Then: gate*A_k  needs product of two piecewise-linear functions of
    #   different variables (s and x) -> not directly representable.
    #   Instead: use the *shifted ReLU* trick to build the product
    #   g_k(s) * A_k via:  ReLU(A_k + M(g_k-1)) - ReLU(A_k - M g_k)
    #   which equals  A_k^+ *g_k - A_k^+*(1-g_k)  -> not the product.
    #
    #   Cleaner: build the gate directly as an affine function of s in the
    #   *pre-activation* of conv3, and combine with x multiplicatively
    #   through the classic "ReLU(x + c) - ReLU(x - c)" identity:
    #       for a scalar c:  ReLU(x + c) - ReLU(x - c) = clamp(2c, ...)...
    #   We instead exploit that we need only ONE scalar per example:
    #   c(t) = sum_k c_k g_k(s).  Then eps_hat = c(t) * x.
    #   Multiplication of a per-example scalar by a per-pixel value is
    #   achievable because the final layer is LINEAR: if conv3 outputs
    #   channels  F_k = g_k(s) * x  (per-pixel gated x), then
    #   out = sum_k c_k F_k = c(t) x.
    #   So we need conv3 to output g_k(s)*x for 32 different k -> 32
    #   channels (plus their sign partners) -> 64 channels exactly.
    #
    #   Product g_k(s)*x construction (exact, using hat functions and
    #   ReLU pairs):
    #     Let h_k(s) = hat_k(s) (piecewise linear, exact one-hot).
    #     For each k we need ReLU-based channels whose difference
    #     equals h_k(s)*x.
    #     Use:  ReLU(x + M(h_k - 1)) - ReLU(x - M h_k) = x^+ (2 h_k - 1)
    #     and   ReLU(-x + M(h_k - 1)) - ReLU(-x - M h_k) = x^- (2 h_k - 1)
    #     => x (2 h_k - 1) = sum of the two differences.
    #     Since we need per-k isolation, use 4 channels per k -> too many.
    #     Instead: use the *bias* trick to isolate: conv3 has 64 outputs;
    #     we use 2 channels per k (32 k's) and put the "+x/2" term in the
    #     out layer via a shared x channel (conv3 channel 63 = x).
    #     With M large (say 1e3) and x bounded, ReLU(x + M(h-1)):
    #        h=1 -> ReLU(x) = x^+ ;  h=0 -> ReLU(x - M) = 0
    #     ReLU(x - M h):  h=1 -> ReLU(x-M)=0 ; h=0 -> ReLU(x)=x^+
    #     => D_k = ReLU(x + M(h_k-1)) - ReLU(x - M h_k) = x^+ (2h_k - 1)
    #     Similarly E_k for -x.
    #     x(2h_k-1) = D_k + E_k  -> needs 2 channels per k = 64 channels
    #     for 32 k's, leaving no room for the shared x channel.
    #     => reduce to 16 groups?  We need one-hot per timestep though.
    #
    #   RESOLUTION: we do not need one-hot per timestep at all.  We need
    #   c(t) = sum_k c_k h_k(s) where h_k are hat functions; we can use
    #   ANY piecewise-linear basis in s.  Use 32 basis functions but
    #   implement the gate as a *linear* function of s inside conv3
    #   pre-activations, exploiting that conv3's pre-activation is
    #   linear in conv2's outputs, which are ReLU(A_k) and ReLU(B_k).
    #   Since A_k = x + beta_k s, we have (for each k) a channel pair
    #   giving A_k exactly.  Now build conv3 pre-activations:
    #       P_k = A_k * (alpha_k + gamma_k * s)   -- still a product.
    #   => unavoidable product of x and s.
    #
    #   FINAL RESOLUTION: implement the product x * p(s) exactly using
    #   the identity (valid for all real x, all real p):
    #       x * p  =  0.5 * [ (x + p)^2 - (x - p)^2 ] / ... no.
    #   Instead use the *bilinear* identity valid because p is a scalar
    #   and x is a scalar:
    #       x*p = 0.5*[ (x+p)*? ] ...
    #   Use instead the exact identity:
    #       x * p = 0.25 * [ (x+p)^2 - (x-p)^2 ]
    #   and note that squares of affine-in-(x,s) quantities are again
    #   products.  Circular.
    #
    #   PRACTICAL RESOLUTION: the out layer is a 3x3 CONVOLUTION.  Use the
    #   spatial dimension to realise the product: place the per-example
    #   scalar p in EVERY spatial position of a conv3 channel, and the
    #   per-pixel x in another channel; the product still needs a ReLU.
    #
    #   ==> Use the well-known ReLU product trick with SHIFTED copies:
    #       For p >= 0 and x any real:
    #         ReLU(x + p) - ReLU(x - p) = min(2p, ...)  (not x*p)
    #   ==> Use instead: since p is a *constant per example* and the
    #   network is convolutional, we can realise x*p by making the
    #   conv3 pre-activation contain x + p*eps_channel where the second
    #   term is a per-example constant times a *per-pixel* channel that
    #   equals 1 (a constant channel).  Then conv3 output = ReLU(x + p)
    #   and ReLU(-x - p) -> difference = x.  Still no p.
    #
    #   ==> LAST RESOLUTION (works, exact): make the *out* layer perform
    #   the selection using per-example gates that are already present in
    #   the FEATURES, exploiting that out is linear and that the gate
    #   multiplies x *inside* conv3 via the shift trick applied to a
    #   channel that already contains x:
    #       channel = ReLU(x + M(h_k - 1)) - ReLU(x + M h_k - 2M)  ...
    #   Let's verify a clean identity.  Define for h in {0,1}:
    #       F_h(x) = ReLU(x + M(h-1)) - ReLU(x + M h - 2M)   [M large]
    #       h=1: ReLU(x) - ReLU(x - M) = x^+ - 0 = x^+
    #       h=0: ReLU(x - M) - ReLU(x - 2M) = 0 - 0 = 0
    #       => F_h = h * x^+.
    #       Similarly for x^-:  G_h = h * x^-.
    #       Then h*x = F_h - G_h  (needs 2 channels per k).
    #       With 32 k's -> 64 channels exactly (no room for shared x).
    #       BUT: we can drop the separate sign handling by noting that
    #       the out layer can also use a *shared* x channel if we free
    #       one slot: use 31 timesteps?  No.
    #       Instead: note that x^+ - x^- = x, and we need h*x.
    #       Use 2 channels per k but with the second channel carrying
    #       -x^- contribution: define channel P_k = h_k * x^+ - h_k * x^-?
    #       Not linear.
    #
    #   ==> SIMPLEST EXACT SOLUTION: exploit that ReLU is applied to the
    #       conv3 pre-activation, and that we can choose the pre-activation
    #       to be  x + M*(h_k - 1)  and  -x + M*(h_k - 1).  Their ReLU
    #       outputs are  h_k*x^+  and  h_k*x^-  (as shown above with the
    #       second term ReLU(.-M) = 0).  So ONE conv3 channel gives
    #       h_k * x^+, and we need h_k * x^- too.
    #       => 2 channels per timestep, 32 timesteps = 64 channels.
    #       The shared "+x/2" correction is unnecessary because
    #       h_k * x = h_k*x^+ - h_k*x^-  (exact, from the two channels).
    #       So: conv3 channel 2k   = h_k * x^+
    #          conv3 channel 2k+1 = h_k * x^-
    #       and out weight for channel 2k = +c_k/2 *?  We need
    #          out = sum_k c_k * h_k * x = sum_k c_k*(ch2k - ch2k+1)
    #       => out.weight[0, 2k] = c_k, out.weight[0, 2k+1] = -c_k.
    #       And h_k(s) must be produced inside conv3's pre-activation as
    #       an affine function of s (since conv3 pre-activation is linear
    #       in conv2 outputs, and conv2 outputs include ReLU(A_k) with
    #       A_k = x + beta_k s, plus ReLU(B_k) = -(x+beta_k s)).
    #       We need h_k(s) alone (no x) inside conv3: build it from the
    #       conv1 s-channels?  conv2 destroyed them (conv2 outputs are
    #       only A_k/B_k).  So we must carry s through conv2 as well.
    #       => conv2 outputs: 32 A_k + 32 B_k = 64 (full).
    #       FIX: use only 16 distinct A/B pairs in conv2 and let conv3
    #       handle 32 timesteps by using *different* affine combos of the
    #       16 available (x + beta_k s) channels.  Since conv3 pre-
    #       activation is linear in conv2 outputs, we can form any
    #       combination  sum_j lambda_j (x + beta_j s) = (sum lambda_j) x
    #       + (sum lambda_j beta_j) s.  To get h_k(s) with zero x-part we
    #       need sum lambda_j = 0 and sum lambda_j beta_j = 1 (times scale).
    #       With 16 betas, we can build 32 such linear functionals giving
    #       32 hat functions in s (hat functions are piecewise linear with
    #       2 breakpoints; a linear combination of hat functions is
    #       piecewise linear with many breakpoints - fine).
    #       But conv3's pre-activation must be ReLU(x + M(h_k -1)) which
    #       contains x AND h_k -> product again!
    #       => the shift trick requires the pre-activation to be
    #          x + M*h_k - M, i.e. x-part plus s-part in ONE channel.
    #       So conv3 pre-activation = x + M*h_k(s) - M.  This is affine
    #       in (x, s) with s-piece = M*h_k(s) which is piecewise linear,
    #       NOT affine -> cannot be a single linear combination of
    #       conv2 outputs (which give affine-in-s pieces only).
    #       BUT: h_k(s) is a hat function = max(0, 1-31|s-s_k|) which is
    #       piecewise linear with a break at s_k.  We can realise the
    #       *positive part* structure inside conv2 by using ReLU there!
    #       i.e. conv2 produces channels = ReLU(affine in s) and we can
    #       build hat functions as sums of ReLU(affine) terms.
    #
    #   ==> PLAN: conv2 (64 channels) produces:
    #         32 channels: ReLU( x + beta_k s )        (A_k)
    #         32 channels: ReLU( -(x + beta_k s) )     (B_k)
    #       conv3 (64 channels) produces, for k=0..31:
    #         ch 2k   : ReLU( x + M*(h_k(s) - 1) )  -> h_k * x^+
    #         ch 2k+1 : ReLU( -x + M*(h_k(s) - 1) ) -> h_k * x^-
    #       where h_k(s) must be *affine* in the conv3 pre-activation.
    #       Since conv3 pre-activation is linear in conv2 outputs, and
    #       conv2 outputs are ReLU of affine-in-(x,s) functions, we can
    #       realise h_k(s) as a linear combination of ReLU(affine) terms
    #       ONLY if h_k is piecewise linear with breakpoints matching.
    #       Hat function h_k(s) = max(0, 1 - 31|s-s_k|):
    #           = max(0, 1 + 31 s_k - 31 s) + max(0, 1 - 31 s_k - 31 s) - 1
    #       Each max(0, affine in s) is a ReLU of an affine function of s.
    #       And s alone is available in conv2?  conv2 outputs are
    #       ReLU(x + beta_k s) - not pure s.  BUT we can get pure s
    #       from conv2 if some channels have zero x-part: e.g. A_k with
    #       x-coefficient 0 -> ReLU(beta_k s).  We can dedicate a few
    #       channels of conv2 to that (we have 64 channels; use 32 for
    #       A/B pairs of 16 distinct betas, plus 32 spare).
    #
    #   ==> FINAL CONCRETE PLAN (exact, all layers used):
    #     conv1 (32): ch 0..13 = +x, 14..27 = -x, 28 = +s, 29 = -s,
    #                 30,31 = +x (spare)
    #     conv2 (64): build 32 channels of ReLU(x + beta_k s) [A_k]
    #                 and 32 channels of ReLU(-(x + beta_k s)) [B_k],
    #                 where betas are 32 distinct values.
    #     conv3 (64): for each k, produce
    #                 ch 2k   = ReLU( x + M*(h_k(s)-1) )
    #                 ch 2k+1 = ReLU( -x + M*(h_k(s)-1) )
    #               with h_k(s) = hat function, realised as an affine
    #               combination of the conv2 outputs:
    #                 h_k(s) = sum_j lam_{k,j} * A_j(s-only part)
    #               We need s-only channels in conv2.  Use the spare
    #               channels: conv2 channels 32..47 hold ReLU(s - s_j)
    #               style pieces (s-only, x-free) and 48..63 hold
    #               ReLU(-(x+beta_k s)) pairs.
    #   This is getting complex; implement it concretely below with a
    #   clean, verified construction.
    # ==================================================================

    # ------------------------------------------------------------------
    # CONCRETE CONSTRUCTION
    # ------------------------------------------------------------------
    # conv1: 32 channels
    #   0..13  : +x
    #   14..27 : -x
    #   28     : +s
    #   29     : -s
    #   30,31  : +x (spare)
    c1w = np.zeros((32, 2, 3, 3), dtype=np.float64)
    for o in range(14):
        c1w[o, 0, 1, 1] = 1.0
    for o in range(14, 28):
        c1w[o, 0, 1, 1] = -1.0
    c1w[28, 1, 1, 1] = 1.0
    c1w[29, 1, 1, 1] = -1.0
    c1w[30, 0, 1, 1] = 1.0
    c1w[31, 0, 1, 1] = 1.0
    c1b = np.zeros(32, dtype=np.float64)

    # conv2: 64 channels
    #   0..31  : A_k = ReLU( x + beta_k * s )      (k = 0..31)
    #   32..63 : B_k = ReLU( -(x + beta_k * s) )
    betas = np.linspace(-1.0, 1.0, 32)
    c2w = np.zeros((64, 32, 3, 3), dtype=np.float64)
    c2b = np.zeros(64, dtype=np.float64)
    for k in range(32):
        for i in range(14):
            c2w[k, i, 1, 1] = 1.0
        for i in range(14, 28):
            c2w[k, i, 1, 1] = -1.0
        c2w[k, 28, 1, 1] = betas[k]
        c2w[k, 29, 1, 1] = -betas[k]
        o = 32 + k
        for i in range(14):
            c2w[o, i, 1, 1] = -1.0
        for i in range(14, 28):
            c2w[o, i, 1, 1] = 1.0
        c2w[o, 28, 1, 1] = -betas[k]
        c2w[o, 29, 1, 1] = betas[k]

    # conv3: 64 channels
    #   ch 2k   : ReLU( x + M*(h_k(s) - 1) )   -> h_k(s) * x^+
    #   ch 2k+1 : ReLU( -x + M*(h_k(s) - 1) )  -> h_k(s) * x^-
    #   h_k(s) must be an affine function of the conv2 outputs.
    #   Available in conv2: A_j = ReLU(x + beta_j s), B_j = ReLU(-(x+beta_j s)).
    #   A linear combination sum_j lam_j A_j + mu_j B_j is affine in (x,s)
    #   only in the region where the ReLUs are active; not globally.
    #   => We instead obtain h_k(s) from *pure s* channels.  To get pure
    #   s channels out of conv2 we need conv2 channels with zero
    #   x-coefficient: use channels 32..47 to hold ReLU(s - s_j) pieces.
    #   That means dropping the B_k channels -> conv2 layout changes:
    #     0..15  : A_k = ReLU(x + beta_k s)      (16 distinct betas)
    #     16..31 : B_k = ReLU(-(x + beta_k s))     (16 distinct betas)
    #     32..47 : P_j = ReLU(s - s_j)             (s-only pieces)
    #     48..63 : Q_j = ReLU(-s + s_j)            (s-only pieces)
    #   Then h_k(s) (hat function) can be written as
    #     h_k(s) = ReLU(1 + 31 s_k - 31 s) + ReLU(1 - 31 s_k - 31 s) - 1
    #   Using P/Q pieces: ReLU(a - 31 s) = ReLU(a - 31 s_j + 31(s_j - s))...
    #   simpler: use P_j = ReLU(s - s_j) with s_j = j/31 for j=0..15 and
    #   Q_j = ReLU(s_j - s).  Then any hat function can be assembled:
    #     hat_k(s) = max(0, 1 - 31|s - s_k|)
    #              = max(0, (1 + 31 s_k) - 31 s) + max(0, (1 - 31 s_k) - 31 s) - 1
    #   The first term = ReLU(c - 31 s) with c = 1+31 s_k.  Express as
    #   ReLU(c - 31 s) = ReLU((c - 31 s_j) + 31(s_j - s)) -- not linear.
    #   Instead: note ReLU(c - 31 s) = c - 31 s + ReLU(31 s - c)  (identity
    #   ReLU(u) = u - ReLU(-u)).  So we need ReLU(31 s - c) pieces:
    #   ReLU(31 s - c) = 31*ReLU(s - c/31) = 31 * P_{c/31 index}.
    #   So with P_j = ReLU(s - j/31) for j = 0..31 we can get any
    #   ReLU(s - u) for u = j/31.  c/31 = (1+31 s_k)/31 = s_k + 1/31.
    #   => need P_j for j = 0..31 (32 channels) and Q_j = ReLU(j/31 - s)
    #      for j = 0..31 (32 channels) -> 64 channels, no room for A/B.
    #   => Use a coarser but sufficient basis: hat functions on a grid
    #      of 16 nodes give a piecewise-linear interpolant of c(t) in s
    #      with 16 dof; but we need the *product* with x, i.e. we need
    #      32 separate gates to select 32 different gains c_k exactly.
    #      Alternative: interpolate c(t) piecewise-linearly in s with
    #      32 nodes -> need 32 hat functions -> 32 gates.
    #   => We need 32 gates; each gate needs 2 channels (x+ and x-) in
    #      conv3 -> 64 channels in conv3 -> conv3 pre-activations must
    #      contain x and h_k(s) in the SAME channel -> conv2 must supply
    #      both x and s pieces in a form that allows a *single* affine
    #      combination to give x + M*h_k(s) - M.
    #      Since h_k(s) is piecewise linear (not affine), we need the
    #      ReLU nonlinearity to create the hat shape.  That means the
    #      hat must be formed in conv2 (ReLU layer) and then conv3 can
    #      combine it with x linearly: conv3 pre-activation =
    #      alpha * x_part + M * hat_part - M.
    #      So conv2 must supply (i) a channel equal to x (or x^+,x^-)
    #      and (ii) channels equal to hat_k(s).
    #      x^+ and x^- need 2 channels; hats need 32 -> 34 > 64? fits!
    #      But hats need 2 pieces each (ReLU of affine in s) unless we
    #      can build hat from a single ReLU piece: hat_k(s) = ReLU(1-31|s-s_k|)
    #      is a tent -> needs 2 pieces.  However we can use a *different*
    #      one-hot basis: instead of hats, use the identity
    #         h_k(s) = sum_{j} d_{kj} * ReLU(s - j/31) + e_{kj} * ReLU(j/31 - s)
    #      with 32+32 pieces.  Too many.
    #      Use fewer timesteps: the gain c(t) is smooth in t; we can
    #      approximate it with a piecewise-linear function on a grid of
    #      M nodes (M <= 16) and use M gates.  With M=16 gates we need
    #      16*2 = 32 conv3 channels for the gated x^+/x^-, plus we can
    #      use the remaining 32 conv3 channels for anything.
    #      For the gates we need hat functions on a 16-node grid over
    #      s in [0,1]: hat_j(s) = max(0, 1 - |s - s_j|*15) for s_j = j/15.
    #      Each hat needs 2 ReLU pieces in s -> 32 s-only channels in
    #      conv2 + 2 x channels = 34 <= 64.
    #      Then conv3: for each of 16 gates j: ch 2j = ReLU(x + M(h_j-1)),
    #      ch 2j+1 = ReLU(-x + M(h_j-1)); 32 channels used, 32 spare.
    #      out: weight on ch 2j = c_j, on ch 2j+1 = -c_j where c_j is the
    #      gain at node j (piecewise-linear interpolation of c(t)).
    #      => eps_hat = sum_j c_j h_j(s) x = c_interp(s) * x.
    #      Since c(t) is smooth and we interpolate on 16 nodes, the
    #      approximation error is small.
    #      BETTER: use 16 nodes but choose the node positions to match
    #      the 32 timesteps in pairs (t=2j, 2j+1 share a node) -> then
    #      c_j = c(2j) and c(2j+1) approximated by c(2j).  Hmm, better
    #      to interpolate.
    #      Actually we can do better: use 16 gates but each gate can be
    #      a "two-node" indicator covering 2 timesteps?  Keep it simple:
    #      16 nodes at s_j = j/15, c_j = c(t) interpolated at s_j.
    #      Even better: use 24 nodes? conv3 needs 2 channels per gate
    #      => max 32 gates. conv2 needs 2 s-pieces per gate (tent) => 64
    #      channels, leaving no room for x channels.  So 16 gates is the
    #      max with this scheme (32 s-channels + 2 x-channels = 34 <= 64,
    #      and 32 conv3 channels for gates <= 64).  We could use up to
    #      31 gates if conv2 has 62 s-channels + 2 x-channels = 64.
    #      Let's use 24 gates: 48 s-channels + 2 x-channels = 50 <= 64,
    #      conv3 uses 48 channels for gates, 16 spare.  Good compromise.
    #      Actually let's just use 31 gates with 62 s-channels + 2
    #      x-channels = 64 exactly, and conv3 uses 62 channels.
    #      Wait, x^+ and x^- require the ReLU to be applied to x itself:
    #      conv2 channel = ReLU(x) gives x^+, and ReLU(-x) gives x^-.
    #      Then conv3 pre-activation for ch 2j:  x^+ + M*(h_j - 1)?? No:
    #      we need ReLU(x + M(h_j-1)) which is not x^+ + ... .
    #      Careful! The shift trick requires the pre-activation to be
    #      x + M(h_j - 1) where x is the SIGNED value, not x^+.
    #      If conv2 only supplies x^+ and x^-, we cannot form signed x
    #      linearly.  BUT: we can use the identity
    #         ReLU(x + M(h-1)) = ?
    #      Alternative: use the *difference* trick with x^+ and x^-:
    #         h * x = h*x^+ - h*x^-
    #      and h*x^+ can be obtained as: ReLU(x^+ + M(h-1))? For h=1:
    #      ReLU(x^+) = x^+; for h=0: ReLU(x^+ - M) = 0 (if x^+ < M).
    #      => channel = ReLU(x^+ + M(h-1)) = h * x^+  (since x^+ >= 0).
    #      Yes! Because x^+ >= 0 always, ReLU(x^+ + M(h-1)) = x^+ when h=1
    #      and 0 when h=0 (assuming x^+ < M).
    #      Similarly h*x^- = ReLU(x^- + M(h-1)).
    #      So conv3 pre-activation = x^+ + M(h_j - 1) and x^- + M(h_j - 1),
    #      where x^+, x^- come from conv2 channels and h_j(s) from s-only
    #      channels.  All linear in conv2 outputs.
    #      Then out: eps_hat = sum_j c_j (ch_{2j} - ch_{2j+1}) = c(s) x.
    #      PERFECT and exact.
    # ------------------------------------------------------------------

    # ---- gate grid ----
    NG = 24
    s_nodes = np.linspace(0.0, 1.0, NG)
    # piecewise-linear interpolation of c(t) at nodes
    c_nodes = np.interp(s_nodes, np.arange(T) / 31.0, c_k)

    # conv2 layout (64 channels):
    #   0      : x^+ = ReLU(x)          (x-coefficient +1 on ch0 of conv1)
    #   1      : x^- = ReLU(-x)         (x-coefficient -1 on ch14)
    #   2..(2+2*NG-1) : s-only ReLU pieces
    #       for gate j:  need ReLU(s - u) and ReLU(u - s) for the tent
    #       hat_j(s) = max(0, 1 - |s - s_j| * (NG-1))
    #       = max(0, (1 + (NG-1) s_j) - (NG-1) s) + max(0, (1 - (NG-1) s_j) - (NG-1) s) - 1
    #       Using identity max(0, a - b s) = a - b s + max(0, b s - a):
    #         term1 = (1 + (NG-1)s_j) - (NG-1)s + max(0, (NG-1)s - (1+(NG-1)s_j))
    #         term2 = (1 - (NG-1)s_j) - (NG-1)s + max(0, (NG-1)s - (1-(NG-1)s_j))
    #       so hat_j(s) = 1 - 2(NG-1)s + ReLU((NG-1)s - (1+(NG-1)s_j))
    #                              + ReLU((NG-1)s - (1-(NG-1)s_j))
    #       Each ReLU term = (NG-1)*ReLU(s - u) with u = (1±(NG-1)s_j)/(NG-1)
    #       These u values are not on the node grid in general; but we can
    #       use a finer grid of s-pieces.  Simpler: use the node grid
    #       directly: hat_j(s) = sum_{i} phi_{ji} * P_i(s) where
    #       P_i(s) = ReLU(s - i/(NG-1)) and we also need the "left"
    #       pieces Q_i(s) = ReLU(i/(NG-1) - s).
    #       hat_j(s) = max(0, 1 - |s-s_j|*(NG-1)) can be written as
    #         sum_{i<=j} (something) ...
    #       Cleanest: build hat_j from P_i and Q_i with the identity
    #         hat_j(s) = sum_{i=0}^{NG-1} w_{ji} * [ ReLU(s - i/(NG-1)) ]
    #                   + sum_{i=0}^{NG-1} v_{ji} * [ ReLU(i/(NG-1) - s) ]
    #       Since {ReLU(s - u_i)} and {ReLU(u_i - s)} for a grid of u_i
    #       span all continuous piecewise-linear functions with knots at
    #       the grid points (plus the linear part), any tent function with
    #       knot at s_j is representable.  Specifically
    #         hat_j(s) = sum_{i=0}^{j} (u_i - u_{i-1}) ... easier:
    #       use the standard B-spline / hat basis identity:
    #         hat_j(s) = sum_{i=0}^{NG-1} [ ReLU(s-u_i) - 2 ReLU(s-u_{i-1}) + ReLU(s-u_{i-2}) ] ...
    #       Let's just use: hat_j(s) = sum_i a_{ji} P_i(s) + sum_i b_{ji} Q_i(s)
    #       with P_i(s)=ReLU(s-u_i), Q_i(s)=ReLU(u_i-s), u_i = i/(NG-1).
    #       The functions {P_i, Q_i} form a basis of PL functions with
    #       knots at u_i (dimension 2*NG, but with one redundancy).
    #       Solve for coefficients by least squares on a fine grid of s
    #       values (closed form, tiny).
    #   Let's do that: build design matrix on a fine grid (e.g. 2001 points)
    #   of basis functions [P_0..P_{NG-1}, Q_0..Q_{NG-1}, 1] and target
    #   hat_j(s); solve least squares (pinv) -> coefficients.  Then conv2
    #   channel for P_i: pre-activation = s - u_i  (x-free), channel for
    #   Q_i: pre-activation = u_i - s.
    #   Number of conv2 channels needed: 2 (x^+,x^-) + 2*NG (P,Q) = 50
    #   for NG=24.  Fits in 64.
    #   conv3 channels: 2 per gate = 48 <= 64.  Fits.

    grid = np.linspace(0.0, 1.0, 2001)
    u = np.linspace(0.0, 1.0, NG)
    basis = np.concatenate([np.maximum(0.0, grid[:, None] - u[None, :]),
                             np.maximum(0.0, u[None, :] - grid[:, None]),
                             np.ones((grid.size, 1))], axis=1)
    coef = np.zeros((NG, basis.shape[1]))
    for j in range(NG):
        target = np.maximum(0.0, 1.0 - np.abs(grid - u[j]) * (NG - 1))
        sol, *_ = np.linalg.lstsq(basis, target, rcond=None)
        coef[j] = sol

    # conv2 weights
    c2w = np.zeros((64, 32, 3, 3), dtype=np.float64)
    c2b = np.zeros(64, dtype=np.float64)
    # channel 0: x^+  -> pre-activation = x  (from conv1 ch0 = +x)
    c2w[0, 0, 1, 1] = 1.0
    # channel 1: x^-  -> pre-activation = -x (from conv1 ch14 = -x)
    c2w[1, 14, 1, 1] = 1.0
    # channels 2 .. 2+NG-1 : P_i(s) = ReLU(s - u_i)  from conv1 ch28 (+s)
    for i in range(NG):
        c2w[2 + i, 28, 1, 1] = 1.0
        c2b[2 + i] = -u[i]
    # channels 2+NG .. 2+2NG-1 : Q_i(s) = ReLU(u_i - s) from conv1 ch29 (-s)
    for i in range(NG):
        c2w[2 + NG + i, 29, 1, 1] = 1.0
        c2b[2 + NG + i] = u[i]

    # conv3 weights: for gate j:
    #   ch 2j   : ReLU( x^+ + M*(h_j - 1) )
    #   ch 2j+1 : ReLU( x^- + M*(h_j - 1) )
    #   h_j = sum_i coef[j,i] * basis_i  (basis uses conv2 outputs)
    #   M large enough that x^+ + M(h-1) < 0 when h=0 -> need M > max x^+
    xmax = float(np.abs(x).max())
    M = 4.0 * xmax + 10.0
    c3w = np.zeros((64, 64, 3, 3), dtype=np.float64)
    c3b = np.zeros(64, dtype=np.float64)
    for j in range(NG):
        # x-part
        c3w[2 * j, 0, 1, 1] = 1.0
        c3w[2 * j + 1, 1, 1, 1] = 1.0
        # s-part: M * (h_j - 1)
        for i in range(NG):
            a = M * coef[j, i]
            c3w[2 * j, 2 + i, 1, 1] += a
            c3w[2 * j + 1, 2 + i, 1, 1] += a
        for i in range(NG):
            a = M * coef[j, NG + i]
            c3w[2 * j, 2 + NG + i, 1, 1] += a
            c3w[2 * j + 1, 2 + NG + i, 1, 1] += a
        c3b[2 * j] = -M
        c3b[2 * j + 1] = -M

    # out weights: eps_hat = sum_j c_j (ch2j - ch2j+1)
    ow = np.zeros((1, 64, 3, 3), dtype=np.float64)
    ob = np.zeros(1, dtype=np.float64)
    for j in range(NG):
        ow[0, 2 * j, 1, 1] = c_nodes[j]
        ow[0, 2 * j + 1, 1, 1] = -c_nodes[j]

    # ------------------------------------------------------------------
    # VERIFY the constructed network numerically on a subset
    # ------------------------------------------------------------------
    def forward(xin, tin, Wd):
        s = (tin.astype(np.float64) / 31.0).reshape(-1, 1, 1, 1)
        inp = np.concatenate([xin, np.broadcast_to(s, (xin.shape[0], 1, xin.shape[2], xin.shape[3]))], axis=1)
        h1 = np.maximum(conv2d(inp, Wd["conv1.weight"], Wd["conv1.bias"]), 0.0)
        h2 = np.maximum(conv2d(h1, Wd["conv2.weight"], Wd["conv2.bias"]), 0.0)
        h3 = np.maximum(conv2d(h2, Wd["conv3.weight"], Wd["conv3.bias"]), 0.0)
        return conv2d(h3, Wd["out.weight"], Wd["out.bias"])

    Wd = {
        "conv1.weight": c1w, "conv1.bias": c1b,
        "conv2.weight": c2w, "conv2.bias": c2b,
        "conv3.weight": c3w, "conv3.bias": c3b,
        "out.weight": ow, "out.bias": ob,
    }
    idx = np.arange(0, N, max(1, N // 64))
    pred = forward(x[idx], t[idx], Wd)
    tgt = np.asarray(f["y"], dtype=np.float64)[idx]
    mse_check = float(((pred - tgt) ** 2).mean())
    # ideal linear predictor reference on the same subset
    ref = x[idx] * c_k[t[idx]].reshape(-1, 1, 1, 1)
    mse_ref = float(((ref - tgt) ** 2).mean())

    # ------------------------------------------------------------------
    # write back (float32, clipped)
    # ------------------------------------------------------------------
    out = {}
    for k in ("conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias",
              "conv3.weight", "conv3.bias", "out.weight", "out.bias"):
        v = np.asarray(Wd[k], dtype=np.float32)
        v = np.clip(v, -1000.0, 1000.0)
        v = np.where(np.isfinite(v), v, 0.0)
        out[k] = v

    diagnostics = {
        "sigma0_sq": round(s0, 5),
        "mse_constructed_subset": round(mse_check, 5),
        "mse_ideal_linear_subset": round(mse_ref, 5),
        "n_verify": int(idx.size),
        "n_gates": int(NG),
        "gain_min": round(float(c_k.min()), 4),
        "gain_max": round(float(c_k.max()), 4),
    }
    return {"weights": out, "diagnostics": diagnostics}
