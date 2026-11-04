import numpy as np


def _relu(a):
    return np.maximum(a, 0.0)


def _conv3x3(x, W, b):
    """x: (N,C,H,W) -> (N,O,H,W), PyTorch cross-correlation, pad 1."""
    N, C, H, Wd = x.shape
    O = W.shape[0]
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    out = np.zeros((N, O, H, Wd), dtype=np.float32)
    for o in range(O):
        acc = np.zeros((N, H, Wd), dtype=np.float32)
        Wo = W[o]
        for c in range(C):
            for kr in range(3):
                for kc in range(3):
                    acc += Wo[c, kr, kc] * xp[:, c, kr:kr + H, kc:kc + Wd]
        out[:, o] = acc + b[o]
    return out


def _conv3x3_grad(x, W, b, gout):
    """Return (dW, db) for conv3x3 with given upstream gradient gout (N,O,H,W)."""
    N, C, H, Wd = x.shape
    O = W.shape[0]
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    dW = np.zeros_like(W)
    db = gout.sum(axis=(0, 2, 3))
    for o in range(O):
        go = gout[:, o]
        for c in range(C):
            for kr in range(3):
                for kc in range(3):
                    dW[o, c, kr, kc] = float((go * xp[:, c, kr:kr + H, kc:kc + Wd]).sum())
    return dW, db


def edit(w, f):
    w = {k: np.array(v, dtype=np.float32, copy=True) for k, v in w.items()}

    x = np.asarray(f["x"], dtype=np.float32)          # (N,1,16,16)
    t = np.asarray(f["t"]).astype(np.int64).ravel()    # (N,)
    y = np.asarray(f["y"], dtype=np.float32)          # (N,1,16,16)
    N = x.shape[0]
    H = x.shape[2]
    Wd = x.shape[3]
    P = H * Wd

    xf = x.reshape(N, P)
    yf = y.reshape(N, P)

    # ---------------- clean-image statistics (per pixel) ----------------
    clean = np.asarray(f["clean"], dtype=np.float32).reshape(N, P)
    mu_pix = clean.mean(axis=0)          # (P,)
    V_pix = clean.var(axis=0)            # (P,)
    Vbar = float(V_pix.mean())
    mubar = float(mu_pix.mean())

    # ---------------- alphabar schedule ----------------
    ab = np.asarray(f["alphabar"], dtype=np.float64).ravel()
    if ab.size != 32:
        ab = np.linspace(0.999, 1e-4, 32)
    ab = np.clip(ab, 1e-8, 1.0 - 1e-8)
    a_t = np.sqrt(ab)                    # (32,)
    b_t = np.sqrt(1.0 - ab)              # (32,)
    tn = t.astype(np.float64)

    # ---------------- closed-form linear epsilon model ----------------
    # eps ~ alpha(t)*x + beta(t) + per-pixel bias; fit by ridge on (x, t).
    # Design: per-example regressors [x (P), 1, t/31, (t/31)^2] -> 256+3 cols.
    # Solved per pixel-group via normal equations on pooled data.
    lam = 1e-3
    # Build normal equations for a global (shared across pixels) coefficient set
    # on features: [x_pixels (P), 1, t01, t01^2] predicting eps pixels.
    # Because pixels are exchangeable-ish, use per-pixel regressions with
    # shared time-basis coefficients:
    #   eps_p = sum_k c_k(t) * x_p + d_k(t)  ->  fit c_k(t), d_k(t) as
    #   polynomials in t01 (degree 2), solved by least squares over (N,P).
    B = np.stack([np.ones(N), tn / 31.0, (tn / 31.0) ** 2], axis=1)   # (N,3)
    # For each time-basis k: fit scalar c_k and d_k with x and 1 regressors.
    # Normal equations: for regressors [x, 1] per basis, using B weights.
    # Build: for each k, regress eps on [B_k * x, B_k] jointly.
    # Equivalent: solve for M (P x 3) and d (P x 3) minimizing
    #   sum_n || y_n - (M^T b_n) x_n - d^T b_n ||^2
    # -> treat per pixel p: regress y[:,p] on [B*x[:,p] (N,3), B (N,3)].
    # Do this for all pixels jointly with shared structure:
    #   y_n,p = sum_k B_nk (M_pk x_np + d_pk)
    # Solve for M (P,3) and d (P,3) via per-pixel 6x6 normal equations.
    n_pix = P
    # Precompute per-pixel sums
    # Sxx[k,l] = sum_n B_nk B_nl x_np^2 ; Sx[k] = sum_n B_nk B_nl x_np
    # We'll loop over pixels (256) with 6x6 solves -> cheap.
    BtB = np.zeros((3, 3))
    for k in range(3):
        for l in range(3):
            BtB[k, l] = float((B[:, k] * B[:, l]).sum())
    BtB_reg = BtB + lam * np.eye(3) * (BtB.trace() / 3.0 + 1e-12)

    M = np.zeros((n_pix, 3))
    D = np.zeros((n_pix, 3))
    for p in range(n_pix):
        xp = xf[:, p]
        yp = yf[:, p]
        # normal equations for [M0,M1,M2,D0,D1,D2]
        A = np.zeros((6, 6))
        rhs = np.zeros(6)
        # M block: sum over n,k,l B_nk B_nl x_n^2
        for k in range(3):
            for l in range(3):
                A[k, l] = float((B[:, k] * B[:, l] * xp * xp).sum())
        # M-D cross: sum B_nk B_nl x_n
        for k in range(3):
            for l in range(3):
                A[k, 3 + l] = float((B[:, k] * B[:, l] * xp).sum())
                A[3 + l, k] = A[k, 3 + l]
        # D block
        for k in range(3):
            for l in range(3):
                A[3 + k, 3 + l] = float((B[:, k] * B[:, l]).sum())
        for k in range(3):
            rhs[k] = float((B[:, k] * xp * yp).sum())
            rhs[3 + k] = float((B[:, k] * yp).sum())
        A[3:, 3:] += lam * np.eye(3) * (A[3:, 3:].trace() / 3.0 + 1e-12)
        A[:3, :3] += lam * np.eye(3) * (A[:3, :3].trace() / 3.0 + 1e-12)
        sol = np.linalg.solve(A, rhs)
        M[p] = sol[:3]
        D[p] = sol[3:]

    # ---------------- build network implementing this map ----------------
    # Architecture plan:
    #  conv1: 2 -> 32  (x channel filters + t-channel constant filters)
    #  conv2: 32 -> 64 (identity-ish mixing)
    #  conv3: 64 -> 64 (identity-ish mixing)
    #  out:   64 -> 1
    # To get a t-dependent linear map on x we use ReLU hinge gates:
    #   conv1 channel k computes  x_k(x_t) + c_k*(t/31) + b_k
    #   ReLU then conv2 mixes channels with (mostly) identity so that
    #   out = sum_k w_k * ReLU(x_k + c_k t + b_k)
    # which is piecewise-linear in x with t-dependent breakpoints.
    # We fit the final layer by least squares on the resulting features.

    # Step 1: choose conv1 filters (32 channels) as random-ish spatial filters
    # over x_t plus a t-dependent offset channel structure.
    rng = np.random.default_rng(1234)
    n1 = 32
    # spatial filters on x: 3x3, small scale
    Wx1 = (rng.standard_normal((n1, 1, 3, 3)) * 0.35).astype(np.float32)
    # t-channel filters: constant spatial (all-ones * scalar) -> constant offset
    Wt1 = np.zeros((n1, 1, 3, 3), dtype=np.float32)
    ct = rng.standard_normal(n1) * 0.6
    for k in range(n1):
        Wt1[k, 0] = ct[k] / 9.0
    b1 = (rng.standard_normal(n1) * 0.15).astype(np.float32)

    # Step 2: conv2, conv3 as near-identity channel mixing with small noise
    W2 = np.zeros((64, 32, 3, 3), dtype=np.float32)
    W3 = np.zeros((64, 64, 3, 3), dtype=np.float32)
    for j in range(32):
        W2[j, j] = 1.0
    for j in range(64):
        W3[j, j] = 1.0
    W2 += (rng.standard_normal(W2.shape) * 0.01).astype(np.float32)
    W3 += (rng.standard_normal(W3.shape) * 0.01).astype(np.float32)
    b2 = np.zeros(64, dtype=np.float32)
    b3 = np.zeros(64, dtype=np.float32)

    # Step 3: forward to get features (post-ReLU conv3 output) then fit out
    # by ridge least squares to y (per-pixel-shared convolution weights).
    def forward_feats(xin, tin):
        Nn = xin.shape[0]
        tt = (tin.astype(np.float32) / 31.0).reshape(Nn, 1, 1, 1)
        inp = np.concatenate([xin, np.broadcast_to(tt, (Nn, 1, H, Wd))], axis=1)
        h1 = _relu(_conv3x3(inp, np.concatenate([Wx1, Wt1], axis=1), b1))
        h2 = _relu(_conv3x3(h1, W2, b2))
        h3 = _relu(_conv3x3(h2, W3, b3))
        return h3

    feats = forward_feats(x, t)                       # (N,64,16,16)
    # Fit out.weight (1,64,3,3), out.bias (1) by ridge least squares:
    #   y[n,i,j] = sum_{c,dy,dx} W[0,c,dy,dx] * feats[n,c,i+dy-1,j+dx-1] + b
    # Build normal equations over 576+1 unknowns using im2col of feats.
    F = feats.reshape(N, 64, P)                        # (N,64,256)
    # design columns: for each output pixel (i,j) and each (c,dy,dx)
    # im2col on spatial dims per channel
    Fp = np.pad(F, ((0, 0), (0, 0), (1, 1)), mode="constant")
    # patches: (N, 64, 3, 3, 16, 16) -> too big; do per (dy,dx)
    # Build A^T A (577x577) and A^T y (577)
    npar = 576 + 1
    AtA = np.zeros((npar, npar))
    Aty = np.zeros(npar)
    # Precompute per-channel spatial correlation terms efficiently:
    # A^T A block for (c1,dy1,dx1),(c2,dy2,dx2) = sum_n sum_p F1_shift * F2_shift
    # Use FFT-free direct approach: for each channel pair and shift pair.
    # Cost: 64*64*9*9 correlations of 16x16 -> manageable if done as
    # matrix products over spatial dims.
    # Represent F as (N*64, 16,16) -> flatten spatial: (N*64, 256)
    Fm = F.reshape(N * 64, P)
    # For shift (dy,dx), shifted version: Fm shifted spatially.
    shifts = [(dy, dx) for dy in range(3) for dx in range(3)]
    # Build list of shifted matrices: (N*64, P) each -> 9 * 2000*64*256 floats = 9*32.7M*4B = 1.18GB
    # Too big; instead accumulate Gram per channel-pair using einsum on blocks.
    # Do it channel-pair wise with small memory.
    G = np.zeros((64, 64, 9, 9))
    # spatial correlation between channel c1 and c2 with shift pair
    # corr(c1,c2,dy1,dx1,dy2,dx2) = sum_n sum_p F1[n,c1,p+off1] * F2[n,c2,p+off2]
    # Compute via: for each pair of shifts, correlate F1 and F2 spatially.
    for i1, (dy1, dx1) in enumerate(shifts):
        for i2, (dy2, dx2) in enumerate(shifts):
            # sum over n,p of F1 shifted by (dy1,dx1) * F2 shifted by (dy2,dx2)
            # = sum over n of <shift(F1), shift(F2)>
            # Build shifted views: index p -> p + off
            oy1, ox1 = dy1 - 1, dx1 - 1
            oy2, ox2 = dy2 - 1, dx2 - 1
            # valid region where both in-bounds
            ys = slice(max(0, oy1, oy2), min(H, H + oy1, H + oy2))
            # simpler: pad and take slices
            def shifted(Fc, oy, ox):
                # Fc: (N,64,16,16)
                out = np.zeros_like(Fc)
                ys0 = max(0, -oy); ys1 = min(H, H - oy)
                xs0 = max(0, -ox); xs1 = min(Wd, Wd - ox)
                out[:, :, ys0 + oy:ys1 + oy, xs0 + ox:xs1 + ox] = Fc[:, :, ys0:ys1, xs0:xs1]
                return out
            A1 = shifted(F, oy1, ox1)
            A2 = shifted(F, oy2, ox2)
            # sum over n and spatial of A1*A2 per channel pair
            prod = (A1 * A2).sum(axis=(0, 2, 3))   # (64,)
            # need full channel-pair matrix: use einsum on (N,64,H,W)
            # cheaper: compute via reshape
            a = A1.reshape(N, 64, P)
            b2_ = A2.reshape(N, 64, P)
            # channel-pair gram: sum_n,p a[n,c1,p]*b2[n,c2,p]
            # = sum_p (a[:,c1,p] outer b2[:,c2,p])
            # do with einsum: 'ncp,nkp->ck'
            G[:, :, i1, i2] = np.einsum('ncp,nkp->ck', a, b2_)
    # Aty: for each (c,dy,dx): sum_n sum_p shifted_feat * y
    yf2 = yf  # (N,P)
    Aty = np.zeros(576)
    for i1, (dy1, dx1) in enumerate(shifts):
        oy1, ox1 = dy1 - 1, dx1 - 1
        ys0 = max(0, -oy1); ys1 = min(H, H - oy1)
        xs0 = max(0, -ox1); xs1 = min(Wd, Wd - ox1)
        sh = np.zeros((N, 64, P), dtype=np.float32)
        sh[:, :, ys0 * Wd + xs0: ys1 * Wd + xs1] = F[:, :, ys0 * Wd + xs0: ys1 * Wd + xs1]
        # careful: spatial slice -> flatten index mapping
        # rebuild properly
        sh = np.zeros((N, 64, H, Wd), dtype=np.float32)
        sh[:, :, ys0 + oy1:ys1 + oy1, xs0 + ox1:xs1 + ox1] = F[:, :, ys0:ys1, xs0:xs1]
        shf = sh.reshape(N, 64, P)
        Aty[i1::9] = np.einsum('ncp,np->c', shf, yf2)
    # assemble AtA
    idx = np.arange(576)
    AtA = np.zeros((577, 577))
    for i1 in range(9):
        for i2 in range(9):
            AtA[np.ix_(idx[i1::9], idx[i2::9])] = G[:, :, i1, i2]
    # bias column: sum of shifted features per (c,shift) and total
    for i1 in range(9):
        AtA[idx[i1::9], 576] = Aty[i1::9] * 0.0  # placeholder, computed below
    # bias column = sum_n sum_p shifted_feat
    bcol = np.zeros(576)
    for i1, (dy1, dx1) in enumerate(shifts):
        oy1, ox1 = dy1 - 1, dx1 - 1
        ys0 = max(0, -oy1); ys1 = min(H, H - oy1)
        xs0 = max(0, -ox1); xs1 = min(Wd, Wd - ox1)
        sh = np.zeros((N, 64, H, Wd), dtype=np.float32)
        sh[:, :, ys0 + oy1:ys1 + oy1, xs0 + ox1:xs1 + ox1] = F[:, :, ys0:ys1, xs0:xs1]
        bcol[i1::9] = sh.sum(axis=(0, 2, 3))
    AtA[idx, 576] = bcol
    AtA[576, idx] = bcol
    AtA[576, 576] = float(N * P)
    Aty = np.concatenate([Aty, np.array([yf2.sum()])])

    # ridge regularization (scale-aware)
    scale = np.trace(AtA[:576, :576]) / 576.0
    reg = 1e-4 * scale
    AtA[np.arange(576), np.arange(576)] += reg
    AtA[576, 576] += 1e-4 * float(N * P)

    sol = np.linalg.solve(AtA, Aty)
    Wo = sol[:576].reshape(64, 3, 3).astype(np.float32)[None, ...]
    bo = np.array([sol[576]], dtype=np.float32)

    # ---------------- assemble weights ----------------
    w["conv1.weight"] = np.concatenate([Wx1, Wt1], axis=1).astype(np.float32)
    w["conv1.bias"] = b1.astype(np.float32)
    w["conv2.weight"] = W2.astype(np.float32)
    w["conv2.bias"] = b2.astype(np.float32)
    w["conv3.weight"] = W3.astype(np.float32)
    w["conv3.bias"] = b3.astype(np.float32)
    w["out.weight"] = Wo.astype(np.float32)
    w["out.bias"] = bo.astype(np.float32)

    # ---------------- diagnostics ----------------
    def mse_of(weights):
        inp = np.concatenate(
            [x, np.broadcast_to((t.astype(np.float32) / 31.0).reshape(N, 1, 1, 1), (N, 1, H, Wd))],
            axis=1)
        h1 = _relu(_conv3x3(inp, weights["conv1.weight"], weights["conv1.bias"]))
        h2 = _relu(_conv3x3(h1, weights["conv2.weight"], weights["conv2.bias"]))
        h3 = _relu(_conv3x3(h2, weights["conv3.weight"], weights["conv3.bias"]))
        p = _conv3x3(h3, weights["out.weight"], weights["out.bias"])
        return float(((p - y) ** 2).mean())

    mse_new = mse_of(w)
    mse_old = float(((np.asarray(f["prediction"], dtype=np.float32) - y) ** 2).mean())

    # clip / sanitize
    for k in w:
        w[k] = np.clip(np.nan_to_num(w[k], nan=0.0, posinf=0.0, neginf=0.0), -1000.0, 1000.0).astype(np.float32)

    return {"weights": w,
            "diagnostics": {"mse_before": mse_old, "mse_after": mse_new,
                            "V_clean": Vbar, "mu_clean": mubar,
                            "alphabar_sum": float(ab.sum())}}
