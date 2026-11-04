import numpy as np

def edit(w, f):
    # ---------------- helpers ----------------
    def conv3x3(x, W, b):
        # x: (N,C,H,W), W: (O,C,3,3) cross-correlation, pad 1
        N, C, H, Wd = x.shape
        O = W.shape[0]
        xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
        out = np.zeros((N, O, H, Wd), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                out += np.tensordot(xp[:, :, i:i + H, j:j + Wd], W[:, :, i, j], axes=([1], [1]))
        return out + b.reshape(1, -1, 1, 1)

    def forward(x, t, Wd):
        # x: (N,1,16,16) float64, t: (N,) int
        N = x.shape[0]
        tb = (t.astype(np.float64) / 31.0).reshape(N, 1, 1, 1)
        inp = np.concatenate([x, np.broadcast_to(tb, (N, 1, 16, 16))], axis=1)
        h1 = np.maximum(conv3x3(inp, Wd['conv1.weight'], Wd['conv1.bias']), 0.0)
        h2 = np.maximum(conv3x3(h1, Wd['conv2.weight'], Wd['conv2.bias']), 0.0)
        h3 = np.maximum(conv3x3(h2, Wd['conv3.weight'], Wd['conv3.bias']), 0.0)
        return conv3x3(h3, Wd['out.weight'], Wd['out.bias'])

    def mse(pred, y):
        return float(np.mean((pred - y) ** 2))

    # ---------------- data ----------------
    x = np.asarray(f['x'], dtype=np.float64).reshape(-1, 1, 16, 16)
    y = np.asarray(f['y'], dtype=np.float64).reshape(-1, 1, 16, 16)
    t = np.asarray(f['t']).reshape(-1).astype(np.int64)
    clean = np.asarray(f['clean'], dtype=np.float64).reshape(-1, 1, 16, 16)
    abar = np.asarray(f['alphabar'], dtype=np.float64).reshape(-1)
    N = x.shape[0]

    base = {k: np.asarray(v, dtype=np.float64) for k, v in w.items()}
    base_mse = mse(forward(x, t, base), y)

    # ---------------- clean-image statistics ----------------
    C = clean.reshape(N, 256)
    mu = C.mean(axis=0)                       # (256,)
    D = C - mu
    # covariance with shrinkage for stability
    S = (D.T @ D) / N
    tr = np.trace(S) / 256.0
    Sreg = S + 1e-3 * tr * np.eye(256)

    # ---------------- analytic optimal linear predictor ----------------
    # For each timestep: E[eps|x_t] = (I + lam*Sigma)^-1 (x_t - c*mu) / s
    #   c = sqrt(abar), s = sqrt(1-abar), lam = s^2 / c^2
    # Build per-timestep 256x256 operator K_t (applied to flattened x_t).
    # Then realize the (global) map approximately inside the CNN by
    # projecting onto a local (7x7) linear operator:  A_t = P^T K_t P
    # where P embeds a 7x7 local patch of x_t into 256 dims.

    # local patch embedding: for output pixel (r,c), patch rows r-3..r+3, cols c-3..c+3
    H = 16
    R = 3  # radius -> 7x7 receptive field
    idx = []
    for r in range(H):
        for c in range(H):
            v = np.zeros(256)
            for dr in range(-R, R + 1):
                for dc in range(-R, R + 1):
                    rr = min(max(r + dr, 0), H - 1)
                    cc = min(max(c + dc, 0), H - 1)
                    v[rr * 16 + cc] = 1.0
            idx.append(v)
    P = np.array(idx)                    # (256, 256) rows: patch indicator
    # A_t = P^T K_t P  -> (256,256): local operator in patch coordinates
    # pred_flat = P A_t P^T (x_flat - c*mu) / s  ... careful with scaling
    # Actually: K_t applied to (x - c mu): full = K_t (x - c mu)/s ; local approx = P K_t P^T (x - c mu)/s
    # we want local operator in patch space: A_t = P K_t P^T  (256x256)

    # t-dependent gain: implement via per-timestep channel gains in the network.
    # Network design (linear regime):
    #   conv1: 32 channels: ch0..15 = local 3x3 filters of x (patch of radius1)
    #           ch16..31 = constant channels encoding t (for gain) -> handled separately
    # To get pred = g(t) * (local linear map of x) we need multiplication.
    # Instead: we build the network so that the *effective* map is
    #   pred = sum_j w_j(t) * (F_j * x)  where F_j are 3x3 filters,
    # implemented with a "ReLU-linear-regime" trick:
    #   channel u_j = ReLU(F_j * x + b_j)  (b_j large -> u_j = F_j*x + b_j)
    #   then a t-gated channel: v_j(t) = ReLU(a_j * t/31 + b_j)  (pure t)
    # Multiplication is unavailable; instead we exploit that the desired
    # predictor is (approximately) separable:  pred = alpha(t) * L(x) + beta(t)
    # We implement alpha(t) by *rescaling the effective filter weights* is impossible,
    # so we use the following exact trick:
    #   ReLU(s * u) = s * ReLU(u) for s>0  -> implement alpha(t) by making the
    #   *first* layer produce s(t)-scaled copies of the x-features.  s(t) is a
    #   function of t only, and conv1 sees t as an additive channel, so we can
    #   build s(t) via ReLU(t) hinges and then... still needs product.
    # => Instead: use a *piecewise-constant-in-t* set of channels and let the
    #    network select via ReLU thresholds.  We approximate alpha(t) by a
    #    staircase of 32 levels using 32 dedicated "gate" channels whose
    #    activation is 1 exactly for t in a given bin (via ReLU on t and 1-t
    #    hinges).  The x-features are duplicated per gate -> too many channels.
    # PRACTICAL RESOLUTION: use the analytic map directly as a *fixed* linear
    # map (t-independent) plus a t-dependent affine term, and additionally
    # exploit that the network's out layer can be fit in closed form.
    # ------------------------------------------------------------------
    # We therefore do the following concrete, verifiable construction:
    #   1) compute the optimal linear predictor target  g_t(x) for every training
    #      example using the Gaussian model (global 256x256 operator).
    #   2) fit, in closed form (ridge), a *local* 7x7 linear operator
    #      L (256x256 in patch coords) and a per-timestep scalar gain alpha(t)
    #      and offset beta(t) that best reproduce g_t(x) on the training data.
    #   3) realize L via conv1/conv2/conv3 (linear regime) and realize
    #      alpha(t), beta(t) via the t-channel + ReLU hinges in conv1 and the
    #      out bias path.
    #   4) verify with a forward pass; keep the best of {analytic, ridge-readout}.

    # ---- step 1: analytic targets ----
    # per-timestep operator (applied to flattened x)
    Kops = {}
    for tt in range(32):
        c = float(np.sqrt(max(abar[tt], 1e-12)))
        s = float(np.sqrt(max(1.0 - abar[tt], 1e-12)))
        lam = (s * s) / (c * c)
        A = np.eye(256) + lam * Sreg
        Kops[tt] = np.linalg.solve(A, np.eye(256))   # (I+lam S)^-1

    # ---- step 2: local (7x7) linear operator fit to the analytic targets ----
    # Build design: for each example and pixel, features = 7x7 patch of x (49 dims)
    # target = analytic prediction at that pixel.
    # Use a subsample of pixels for speed.
    Xf = x.reshape(N, 256)
    # local operator in patch coords: solve min || P K_t P^T x - target ||
    # Instead directly fit: for each t, target_flat = (K_t (x - c mu))/s
    # and we want pred_flat = A_loc (x - c mu) with A_loc = P K_t P^T (256x256).
    # Fit a single A_loc across all t by least squares on the *scaled* problem:
    #   minimize sum_t || (K_t (x-c mu))/s - P A_loc P^T (x - c mu) ||^2
    # Solve for A_loc (256x256) using normal equations with the patch design.
    # Patch design matrix: for each (example, pixel) row -> 49 features.
    # Build once (subsample pixels).
    rng = np.random.default_rng(0)
    pix = rng.choice(256, size=256, replace=False)   # 256 pixels subsample
    rows = []
    for p in pix:
        r, c = divmod(p, 16)
        v = np.zeros(256)
        for dr in range(-R, R + 1):
            for dc in range(-R, R + 1):
                rr = min(max(r + dr, 0), 15)
                cc = min(max(c + dc, 0), 15)
                v[rr * 16 + cc] = 1.0
        rows.append(v)
    rows = np.array(rows)                      # (P, 256)
    # design: features for (example, pixel) = rows[p] . x_flat  (49 dims)
    # target: analytic
    # Solve for B (256 x 49) with pred = (x_flat - c mu) projected: use raw x
    # We fit: target_pix = sum_q B[q, :] * (patch of x) + bias
    # Build normal equations over all examples & sampled pixels.
    nP = len(pix)
    # patch feature extraction via shifted indexing
    def patches(Xf, pix):
        # Xf: (N,256) -> (N, nP, 49)
        Nn = Xf.shape[0]
        out = np.empty((Nn, nP, 49))
        for j, p in enumerate(pix):
            r, c = divmod(p, 16)
            feats = []
            for dr in range(-R, R + 1):
                for dc in range(-R, R + 1):
                    rr = min(max(r + dr, 0), 15)
                    cc = min(max(c + dc, 0), 15)
                    feats.append(Xf[:, rr * 16 + cc])
            out[:, j, :] = np.stack(feats, axis=1)
        return out

    # subsample examples for the fit
    ex_idx = rng.choice(N, size=min(N, 1200), replace=False)
    Px = patches(Xf[ex_idx], pix)             # (Ne, nP, 49)
    # analytic targets
    Tg = np.empty((len(ex_idx), nP))
    for k, ii in enumerate(ex_idx):
        tt = t[ii]
        c = float(np.sqrt(max(abar[tt], 1e-12)))
        s = float(np.sqrt(max(1.0 - abar[tt], 1e-12)))
        K = Kops[tt]
        full = (K @ (Xf[ii] - c * mu)) / s
        Tg[k] = full[pix]
    # normal equations: X^T X (50x50), X^T y (50,)
    Xd = Px.reshape(-1, 49)
    Xd = np.concatenate([Xd, np.ones((Xd.shape[0], 1))], axis=1)
    Yd = Tg.reshape(-1)
    G = Xd.T @ Xd
    b = Xd.T @ Yd
    lam_reg = 1e-2 * np.trace(G) / G.shape[0]
    G[np.diag_indices_from(G)] += lam_reg
    sol = np.linalg.solve(G, b)
    B = sol[:49]                              # (49,)
    b0 = sol[49]

    # predicted local operator applied to full image (for verification)
    def local_pred(Xf, tvec):
        Nn = Xf.shape[0]
        out = np.empty((Nn, 256))
        # build patch features for all pixels (256) -> heavy but ok for 2000? 2000*256*49 = 25M floats = 200MB
        # do it in chunks
        for st in range(0, Nn, 200):
            en = min(Nn, st + 200)
            Xc = Xf[st:en]
            Nc = en - st
            # patch features for all 256 pixels
            feats = np.empty((Nc, 256, 49))
            for p in range(256):
                r, c = divmod(p, 16)
                fl = []
                for dr in range(-R, R + 1):
                    for dc in range(-R, R + 1):
                        rr = min(max(r + dr, 0), 15)
                        cc = min(max(c + dc, 0), 15)
                        fl.append(Xc[:, rr * 16 + cc])
                feats[:, p, :] = np.stack(fl, axis=1)
            out[st:en] = feats.reshape(-1, 49) @ B + b0
        return out.reshape(Nn, 1, 16, 16)

    # ---------------- realize the local linear map in the CNN ----------------
    # Strategy: put the 7x7 filter into the conv stack in "linear regime".
    # conv1 (2->32,3x3): channel 0 = x channel (identity-ish), channels 1..: t channel
    # We need a 7x7 effective filter.  Use:
    #   conv1: 32 ch. ch0 = x (3x3 identity), ch1 = t/31 constant (3x3 zeros + bias? no,
    #         bias is constant not t).  To inject t we use input channel 2 (t/31)
    #         with a 3x3 kernel that is zero except center = 1 -> gives t/31.
    #   conv2: 64 ch: ch0 = conv1 ch0 (identity 3x3) -> gives 3x3 support of x
    #            ch1..: t channel copies
    #   conv3: 64 ch: ch0 = conv2 ch0 (identity) -> 5x5 support
    #   out: 1 ch: 3x3 filter -> 7x7 support
    # With ReLU in linear regime (large positive biases), the stack is linear.
    # But a single 7x7 filter is not enough: our fitted B has 49 taps for a
    # *single* output pixel; the same 7x7 filter applies at every pixel ->
    # that is exactly a 7x7 convolution!  So B reshaped to (7,7) is the filter.
    # However our fit above was per-pixel with the same filter (translation
    # invariant) - yes: B is the 7x7 filter, applied at each pixel.
    filt = B.reshape(2 * R + 1, 2 * R + 1)     # (7,7)

    # We need the network to apply this 7x7 filter to x and output it.
    # Build conv stack:
    #   conv1.weight[0,0] = identity 3x3 ; conv1.weight[0,1] = 0
    #   conv1.weight[1,1] = center 1 (t channel) ; conv1.weight[1,0]=0
    #   conv1.bias = [BIG, 0]
    #   conv2.weight[0,0] = identity 3x3 (on ch0) ; conv2.weight[1,1]=identity (t)
    #   conv2.bias = [BIG, 0]
    #   conv3.weight[0,0] = identity 3x3 ; conv3.weight[1,1] = identity
    #   conv3.bias = [BIG, 0]
    #   out.weight[0,0] = 7x7 filter embedded in 3x3?  No: out is 3x3 only.
    # The 7x7 support comes from conv1(3)+conv2(3)+conv3(3)+out(3) = 11? Actually
    # composition: conv1 3x3 -> conv2 3x3 -> conv3 3x3 -> out 3x3 gives 11x11 RF.
    # We want a 7x7 filter: use conv1 (3) + conv2 (3) + out (3) = 7x7? That's 3+2+2 = 7.
    # Simplest: put the 7x7 filter into conv1+conv2+out composition:
    #   conv1 ch0: 3x3 block A (top-left of filt), conv2 ch0: 3x3 block B (middle),
    #   out: 3x3 block C (bottom-right) -> effective 7x7 = A*B*C convolution.
    # Decompose filt (7x7) into three 3x3 blocks via a simple factorization:
    #   use rank-1 style: filt = u v^T with u,v of length 7 -> implement as
    #   conv1 (3x3) -> conv2 (3x3) -> out (3x3) using outer products.
    # Easier: implement the 7x7 filter via 3 stages of 3x3 with padding, using
    # the "separable outer product" trick:
    #   filt = a (7,) column vector times b (7,) row vector.
    #   conv1: 3x3 kernel = outer(a[0:3], b[0:3]) on channel 0 -> 3x3
    #   conv2: 3x3 kernel = outer(a[2:5], b[2:5]) applied to conv1 output
    #   out:   3x3 kernel = outer(a[4:7], b[4:7])
    # Composition of three 3x3 convs (with zero padding) gives the 7x7 outer
    # product a b^T exactly (since padding is zero and supports align).
    # We need a rank-1 approximation of filt: use SVD, keep top singular value.
    U, Sg, Vt = np.linalg.svd(filt, full_matrices=False)
    a = U[:, 0] * np.sqrt(Sg[0])
    b = Vt[0, :] * np.sqrt(Sg[0])

    def outer3(u, v):
        return np.outer(u, v)

    BIG = 50.0
    Wn = {}
    # conv1
    c1w = np.zeros((32, 2, 3, 3)); c1b = np.zeros(32)
    c1w[0, 0] = outer3(a[0:3], b[0:3])
    c1b[0] = BIG
    c1w[1, 1, 1, 1] = 1.0            # t/31 channel (center tap)
    c1b[1] = 0.0
    Wn['conv1.weight'] = c1w; Wn['conv1.bias'] = c1b
    # conv2
    c2w = np.zeros((64, 32, 3, 3)); c2b = np.zeros(64)
    c2w[0, 0] = outer3(a[2:5], b[2:5])
    c2b[0] = BIG
    c2w[1, 1, 1, 1] = 1.0
    Wn['conv2.weight'] = c2w; Wn['conv2.bias'] = c2b
    # conv3
    c3w = np.zeros((64, 64, 3, 3)); c3b = np.zeros(64)
    c3w[0, 0] = outer3(a[4:7], b[4:7])
    c3b[0] = BIG
    c3w[1, 1, 1, 1] = 1.0
    Wn['conv3.weight'] = c3w; Wn['conv3.bias'] = c3b
    # out
    ow = np.zeros((1, 64, 3, 3)); ob = np.zeros(1)
    ow[0, 0] = 1.0
    ob[0] = b0
    Wn['out.weight'] = ow; Wn['out.bias'] = ob

    cand_mse = mse(forward(x, t, Wn), y)

    # ---------------- fallback / alternative: ridge readout on cached features ----
    feats = np.asarray(f['features'], dtype=np.float64)   # (N,64,16,16)
    # design matrix for out layer: 3x3x64 taps per pixel -> 576 features
    # Build normal equations in chunks (memory safe)
    G = np.zeros((577, 577)); rhs = np.zeros(577)
    # feature extraction helper: for each pixel, 3x3 neighborhood of 64 channels
    def feat_rows(F):
        # F: (N,64,16,16) -> (N*256, 576)
        Nn = F.shape[0]
        Fp = np.pad(F, ((0, 0), (0, 0), (1, 1), (1, 1)))
        out = np.empty((Nn, 256, 576))
        for i in range(3):
            for j in range(3):
                blk = Fp[:, :, i:i + 16, j:j + 16]     # (N,64,16,16)
                out[:, :, (i * 3 + j) * 64:(i * 3 + j + 1) * 64] = blk.transpose(0, 2, 3, 1).reshape(Nn, 256, 64)
        return out
    Frows = feat_rows(feats)
    Yflat = y.reshape(N, 256)
    for st in range(0, N, 250):
        en = min(N, st + 250)
        Xb = Frows[st:en].reshape(-1, 576)
        Xb = np.concatenate([Xb, np.ones((Xb.shape[0], 1))], axis=1)
        Yb = Yflat[st:en].reshape(-1)
        G += Xb.T @ Xb
        rhs += Xb.T @ Yb
    del Frows
    for ridge in (1e-3, 1e-2, 1e-1, 1.0):
        Gr = G.copy()
        Gr[np.diag_indices_from(Gr)] += ridge * np.trace(G) / 577.0
        sol = np.linalg.solve(Gr, rhs)
        Wr = base.copy()
        Wr['out.weight'] = sol[:576].reshape(1, 64, 3, 3)
        Wr['out.bias'] = np.array([sol[576]])
        m = mse(forward(x, t, Wr), y)
        if m < cand_mse:
            cand_mse = m
            Wn = Wr

    # ---------------- also try: analytic global map realized via a *t-gated* scheme ----
    # Global 256x256 map cannot be realized by a 7x7 CNN, but we can improve by
    # using the *full* network capacity: fit a 7x7 filter per timestep-group is
    # impossible (weights static).  Instead try: analytic map applied to a
    # *smoothed* x (via conv stack) - skip.
    # Final: also try scaling the analytic filter by a global constant to
    # minimize MSE (closed-form scalar rescale of prediction).
    best = Wn
    best_mse = cand_mse
    # scalar rescale search on the final prediction
    pred = forward(x, t, best)
    num = float(np.sum(pred * y)); den = float(np.sum(pred * pred))
    if den > 0:
        s_opt = num / den
        for s_try in (s_opt,):
            Ws = {k: v.copy() for k, v in best.items()}
            Ws['out.weight'] = best['out.weight'] * s_try
            Ws['out.bias'] = best['out.bias'] * s_try
            m = mse(forward(x, t, Ws), y)
            if m < best_mse:
                best_mse = m; best = Ws

    # ---------------- package ----------------
    out = {}
    for k, v in base.items():
        out[k] = np.clip(np.asarray(best[k], dtype=np.float32), -1000.0, 1000.0)
    for k, v in out.items():
        if not np.all(np.isfinite(v)):
            out[k] = np.clip(np.asarray(base[k], dtype=np.float32), -1000.0, 1000.0)

    diagnostics = {
        'base_mse': float(base_mse),
        'new_mse_train': float(best_mse),
        'analytic_local_mse': float(cand_mse),
        'n_examples': int(N),
        'filter_rank1_sv': float(Sg[0]),
        'alphabar_min': float(abar.min()),
        'alphabar_max': float(abar.max()),
    }
    return {'weights': out, 'diagnostics': diagnostics}
