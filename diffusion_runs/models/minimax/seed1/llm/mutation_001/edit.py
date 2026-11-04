import numpy as np

def edit(w, f):
    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def g(a):
        return np.asarray(a, dtype=np.float64)

    x      = g(f["x"])            # [N,1,16,16]
    t      = np.asarray(f["t"]).astype(np.int64).ravel()
    y      = g(f["y"])            # [N,1,16,16]
    clean  = g(f["clean"])        # [N,1,16,16]
    abar   = g(f["alphabar"])      # [32]

    N = x.shape[0]
    H = W = 16

    # ------------------------------------------------------------------
    # statistics of the clean images (per-pixel mean / variance)
    # ------------------------------------------------------------------
    c0 = clean[:, 0]                       # [N,16,16]
    mu = c0.mean(axis=0)                   # [16,16]  E[x0] per pixel
    var = c0.var(axis=0)                   # [16,16]  Var[x0] per pixel
    sig2 = float(var.mean())              # average per-pixel variance
    sig2 = max(sig2, 1e-4)

    # ------------------------------------------------------------------
    # Bayes-optimal (diagonal-covariance) linear epsilon estimator:
    #   E[eps | x_t] = c(t) * (x_t - a(t)*mu)
    #   c(t) = b / (a^2*sig2 + b^2),  a = sqrt(abar), b = sqrt(1-abar)
    # ------------------------------------------------------------------
    a2 = abar ** 2
    b2 = 1.0 - a2
    den = a2 * sig2 + b2
    c_t = np.sqrt(b2) / den                # gain on x_t
    d_t = np.sqrt(b2) * a2 / den           # gain on mu (offset, negative)

    # ------------------------------------------------------------------
    # Build the network so that it computes  out = c(t)*x_t + d(t)*mu
    #
    #  conv1 : channel 0 -> ReLU(x_t)      (identity 3x3 on x channel)
    #           channel 1 -> ReLU(-x_t)     (negated identity)
    #           channels 2.. -> constant maps carrying t/31 (for the gain)
    #  conv2 : identity on channels 0,1 and on the t-carrier channels
    #  conv3 : identity on channels 0,1 and on the t-carrier channels
    #  out   : +c on ch0, -c on ch1 (=> c*x_t), plus t-dependent gains
    #
    # The per-pixel offset d(t)*mu is realised by keeping a *spatially
    # structured* constant map: conv1 channels 2.. carry  mu (constant over
    # the image, obtained from the t channel is impossible) -- instead we
    # use the fact that the offset term is small relative to the gain and
    # realise it through the t-carrier channels scaled by the *mean* of mu.
    # ------------------------------------------------------------------
    # We realise the mu-offset properly: the offset map must be spatially
    # varying.  We obtain it by adding, at the conv1 level, a set of
    # channels whose pattern equals mu, generated from the *t* channel is
    # impossible, so instead we exploit the following exact identity:
    #
    #   the network output can be written as
    #       c(t)*x_t  +  d(t)*mu
    #   where the second term is realised by conv1 channels that carry the
    #   *spatially varying* pattern mu via a constant-per-pixel bias that
    #   is impossible; therefore we approximate the offset by the best
    #   achievable *linear* function of x_t (see below).
    #
    # Practical approach: the offset d(t)*mu is a fixed image.  We instead
    # fold it into the fit by *not* subtracting mu at all and letting the
    # network produce  c(t)*x_t  plus a constant per-pixel bias obtained
    # from the conv1 bias of the identity channels (a scalar).  The
    # resulting bias error is  d(t)*std(mu) which we minimise by choosing
    # the scalar bias optimally (least squares over the training set).
    #
    # --- exact closed-form least squares for the final readout ---------
    # We build the network first (deterministically), then compute the
    # exact linear readout that minimises MSE on the training data using
    # the *actual* forward pass of the constructed network.  This is a
    # closed-form ridge regression on the last layer only.

    # ---- construct conv1 ------------------------------------------------
    c1w = np.zeros((32, 2, 3, 3), dtype=np.float64)
    c1b = np.zeros(32, dtype=np.float64)
    c1w[0, 0, 1, 1] = 1.0          # +x_t
    c1w[1, 0, 1, 1] = -1.0         # -x_t
    # channels 2..: t-carriers, ReLU(gamma_k * (t/31) + beta_k)
    #   gamma_k chosen so that the ramp is informative over the whole range
    #   beta_k = 0 -> ReLU(gamma*(t/31)) ; we use pairs (+g, -g) so that
    #   ReLU(g*u) - ReLU(-g*u) = g*u  (exact linear in t/31)
    n_t = 8                        # 4 pairs -> 8 t-carrier channels
    gammas = [1.0, 2.0, 4.0, 8.0]
    for k, gm in enumerate(gammas):
        c1w[2 + 2 * k, 1, 1, 1] = gm      # +g * (t/31)
        c1w[3 + 2 * k, 1, 1, 1] = -gm     # -g * (t/31)
    # remaining channels 10..31: unused (set to zero -> ReLU(0)=0)

    # ---- conv2 : identity for channels 0..9 ------------------------------
    c2w = np.zeros((64, 32, 3, 3), dtype=np.float64)
    c2b = np.zeros(64, dtype=np.float64)
    for c in range(10):
        c2w[c, c, 1, 1] = 1.0
    # channels 10..63 unused

    # ---- conv3 : identity for channels 0..9 ------------------------------
    c3w = np.zeros((64, 64, 3, 3), dtype=np.float64)
    c3b = np.zeros(64, dtype=np.float64)
    for c in range(10):
        c3w[c, c, 1, 1] = 1.0
    # channels 10..63 unused

    # ---- forward pass helper (numpy, cross-correlation, pad 1) ----------
    def conv2d(inp, wt, bias):
        # inp [N,C,H,W], wt [O,C,3,3]
        Nn, C, Hh, Ww = inp.shape
        O = wt.shape[0]
        pad = np.zeros((Nn, C, Hh + 2, Ww + 2), dtype=np.float64)
        pad[:, :, 1:-1, 1:-1] = inp
        out = np.zeros((Nn, O, Hh, Ww), dtype=np.float64)
        for kr in range(3):
            for kc in range(3):
                out += np.einsum('nchw,oc->nohw', pad[:, :, kr:kr + Hh, kc:kc + Ww], wt[:, :, kr, kc])
        out += bias[None, :, None, None]
        return out

    def forward(xin, tvec, c1w_, c1b_, c2w_, c2b_, c3w_, c3b_, ow, ob):
        Nn = xin.shape[0]
        u = tvec.astype(np.float64) / 31.0
        inp = np.concatenate([xin, np.broadcast_to(u[:, None, None, None], (Nn, 1, H, W))], axis=1)
        h1 = np.maximum(conv2d(inp, c1w_, c1b_), 0.0)
        h2 = np.maximum(conv2d(h1, c2w_, c2b_), 0.0)
        h3 = np.maximum(conv2d(h2, c3w_, c3b_), 0.0)
        return conv2d(h3, ow, ob)

    # ------------------------------------------------------------------
    # Feature design matrix for the final linear readout.
    # The final features h3 have channels 0..9 informative:
    #   ch0 = ReLU(x_t), ch1 = ReLU(-x_t)   -> difference = x_t
    #   ch(2+2k) = ReLU(g_k*u), ch(3+2k) = ReLU(-g_k*u) -> difference = g_k*u
    # The readout is a 3x3 conv over 64 channels, so the effective
    # prediction at pixel p is  sum_{c,d} W[c,d] h3[c, p+d] + b.
    # We therefore solve a *pixel-local* least squares problem:
    #   target(p) = y[p];  features(p) = [h3[:, p] (64 values), 1]
    # but the 3x3 support means each output pixel also sees neighbours.
    # To keep it exact and cheap we restrict the readout to the centre tap
    # (W[:, :, 1, 1] only) -- then prediction(p) depends only on h3[:, p].
    # ------------------------------------------------------------------
    # Compute h3 for the whole training set (memory: N*64*16*16*8 bytes)
    # -> 2000*64*256*8 = 262 MB, acceptable.  Use float32 to halve it.
    def compute_h3(xin, tvec):
        Nn = xin.shape[0]
        u = tvec.astype(np.float64) / 31.0
        inp = np.concatenate([xin, np.broadcast_to(u[:, None, None, None], (Nn, 1, H, W))], axis=1)
        h1 = np.maximum(conv2d(inp, c1w, c1b), 0.0)
        h2 = np.maximum(conv2d(h1, c2w, c2b), 0.0)
        h3 = np.maximum(conv2d(h2, c3w, c3b), 0.0)
        return h3

    # chunked computation to limit memory
    h3 = np.zeros((N, 64, H, W), dtype=np.float32)
    chunk = 250
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        h3[s:e] = compute_h3(x[s:e], t[s:e]).astype(np.float32)

    # design: for each (n,p) row: [h3[n,:,p] (64), 1]
    # only channels 0..9 are informative; but keep all 64 for generality
    # Build Gram matrix incrementally (64+1 dims) -> cheap.
    D = 65
    XtX = np.zeros((D, D))
    XtY = np.zeros((D, 1))
    # rows: N*256 ; features: 64 channels at that pixel + bias
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        hh = h3[s:e]                              # [n,64,16,16]
        n_ = hh.shape[0]
        F = hh.reshape(n_, 64, H * W).transpose(0, 2, 1).reshape(-1, 64).astype(np.float64)
        F = np.concatenate([F, np.ones((F.shape[0], 1))], axis=1)
        T = y[s:e, 0].reshape(-1, 1).astype(np.float64)
        XtX += F.T @ F
        XtY += F.T @ T

    # ridge (tiny) for stability
    lam = 1e-6 * np.trace(XtX) / D
    XtX_r = XtX + lam * np.eye(D)
    beta = np.linalg.solve(XtX_r, XtY)            # [65,1]

    # ------------------------------------------------------------------
    # build the readout: out.weight[0,c,1,1] = beta[c], out.bias = beta[64]
    # ------------------------------------------------------------------
    ow = np.zeros((1, 64, 3, 3), dtype=np.float64)
    ow[0, :, 1, 1] = beta[:64, 0]
    ob = np.array([beta[64, 0]], dtype=np.float64)

    # ------------------------------------------------------------------
    # Evaluate the resulting network on the training set (exact forward)
    # ------------------------------------------------------------------
    def mse_of(xin, tvec, tgt):
        tot = 0.0
        cnt = 0
        for s in range(0, xin.shape[0], 250):
            e = min(xin.shape[0], s + 250)
            p = forward(xin[s:e], tvec[s:e], c1w, c1b, c2w, c2b, c3w, c3b, ow, ob)
            d = p - tgt[s:e]
            tot += float(np.sum(d * d))
            cnt += d.size
        return tot / cnt

    mse_new = mse_of(x, t, y)
    mse_old = float(np.mean((g(f["prediction"]) - y) ** 2))

    # ------------------------------------------------------------------
    # Assemble output weights (float32, clipped)
    # ------------------------------------------------------------------
    out = {}
    out["conv1.weight"] = np.clip(c1w, -1000, 1000).astype(np.float32)
    out["conv1.bias"]   = np.clip(c1b, -1000, 1000).astype(np.float32)
    out["conv2.weight"] = np.clip(c2w, -1000, 1000).astype(np.float32)
    out["conv2.bias"]   = np.clip(c2b, -1000, 1000).astype(np.float32)
    out["conv3.weight"] = np.clip(c3w, -1000, 1000).astype(np.float32)
    out["conv3.bias"]   = np.clip(c3b, -1000, 1000).astype(np.float32)
    out["out.weight"]   = np.clip(ow, -1000, 1000).astype(np.float32)
    out["out.bias"]     = np.clip(ob, -1000, 1000).astype(np.float32)

    # keep any other keys that might exist in w
    for k, v in w.items():
        if k not in out:
            out[k] = np.asarray(v, dtype=np.float32)

    # per-timestep diagnostics
    per_t = {}
    for tt in range(32):
        m = (t == tt)
        if m.sum() == 0:
            continue
        per_t[str(tt)] = round(mse_of(x[m], t[m], y[m]), 5)

    diag = {
        "mse_before": round(mse_old, 6),
        "mse_after": round(mse_new, 6),
        "mse_per_t": per_t,
        "sig2_clean": round(sig2, 6),
        "mu_mean": round(float(mu.mean()), 6),
        "readout_norm": round(float(np.linalg.norm(beta[:64, 0])), 6),
    }
    return {"weights": out, "diagnostics": diag}
