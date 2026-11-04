import numpy as np

# ----------------------------------------------------------------------------
# Closed-form epsilon-prediction weight edit.
#  * No autograd / backprop / gradient estimation / iterative optimizers.
#  * Only NumPy linear algebra (ridge least squares, PCA-style projections).
#  * Architecture is preserved exactly (same keys, same shapes).
# ----------------------------------------------------------------------------

def _conv2d(x, W, b):
    """Cross-correlation conv2d, pad=1, stride=1. x:[N,Cin,H,W], W:[Cout,Cin,3,3]."""
    N, Cin, H, Wd = x.shape
    Cout = W.shape[0]
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    # im2col via stride tricks
    s0, s1, s2, s3 = xp.strides
    cols = np.lib.stride_tricks.as_strided(
        xp,
        shape=(N, Cin, 3, 3, H, Wd),
        strides=(s0, s1, s2, s3, s2, s3),
        writeable=False,
    )
    out = np.tensordot(cols, W, axes=([1, 2, 3], [1, 2, 3]))  # [N,Cout,H,Wd]
    out = out + b.reshape(1, -1, 1, 1)
    return out


def _relu(a):
    return np.maximum(a, 0.0)


def _forward(w, x, t):
    """Reproduce the fixed forward pass (conv1->ReLU->conv2->ReLU->conv3->ReLU->out)."""
    tb = (t.astype(np.float32) / 31.0).reshape(-1, 1, 1, 1)
    inp = np.concatenate([x, np.broadcast_to(tb, (x.shape[0], 1, x.shape[2], x.shape[3]))], axis=1)
    h1 = _relu(_conv2d(inp, w["conv1.weight"], w["conv1.bias"]))
    h2 = _relu(_conv2d(h1, w["conv2.weight"], w["conv2.bias"]))
    h3 = _relu(_conv2d(h2, w["conv3.weight"], w["conv3.bias"]))
    out = _conv2d(h3, w["out.weight"], w["out.bias"])
    return inp, h1, h2, h3, out


def _corr_maps(F, taps):
    """Zero-padded 2D correlation of channel maps.

    F: [C,N,H,W] -> for each pair (c,c') and lag (dy,dx) return
    sum_{n,p} F[c,n,p+dy,dx] * F[c',n,p+dy',dx']  with out-of-range = 0.
    Implemented via FFT on zero-padded maps (exact linear correlation).
    Returns array [C, C, len(taps_y), len(taps_x)].
    """
    C, N, H, Wd = F.shape
    P = 1
    while P < H + 2 * taps_y_max + 1:
        P *= 2
    Fp = np.zeros((C, N, P, P), dtype=np.float32)
    Fp[:, :, :H, :Wd] = F
    Fh = np.fft.rfft2(Fp, axes=(-2, -1))
    # corr[c,c'] = IFFT( conj(Fh[c]) * Fh[c'] )  -> lag convention: value at (dy,dx)
    prod = np.conj(Fh)[:, None] * Fh[None, :]           # [C,C,N,P,P/2+1]
    corr = np.fft.irfft2(prod, s=(P, P), axes=(-2, -1))  # [C,C,N,P,P]
    out = np.empty((C, C, len(taps_y), len(taps_x)), dtype=np.float64)
    for iy, dy in enumerate(taps_y):
        for ix, dx in enumerate(taps_x):
            out[:, :, iy, ix] = corr[:, :, :, dy % P, dx % P].sum(axis=2)
    return out


def _fit_readout(feat, y, ridge, n_sub=None, seed=0):
    """Closed-form ridge least squares for the 3x3 out conv (576 weights + bias).

    feat: [N,64,16,16] post-ReLU conv3 activations (the readout input).
    y:    [N,1,16,16] target epsilon.
    Returns (W[1,64,3,3], b[1]).
    """
    N, C, H, Wd = feat.shape
    F = feat.astype(np.float64)
    Y = y.astype(np.float64)[:, 0]                       # [N,H,Wd]
    if n_sub is not None and n_sub < N:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(N, size=n_sub, replace=False))
        F = F[idx]
        Y = Y[idx]
        N = n_sub

    taps = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    ty = sorted(set(t[0] for t in taps))
    tx = sorted(set(t[1] for t in taps))
    corr = _corr_maps(F, ty, tx)                        # [C,C,3,3]
    # Build normal equations for the 576-dim parameter vector
    # index (c, iy, ix) -> c*9 + iy*3 + ix
    A = np.zeros((C * 9, C * 9), dtype=np.float64)
    for iy, dy in enumerate(ty):
        for ix, dx in enumerate(tx):
            # corr[c,c',iy',ix'] gives sum F[c,p+dy,dx] F[c',p+dy',dx']
            # we need A[(c,iy,ix),(c',iy',ix')] = corr[c,c',iy',ix'] with dy=ty[iy], dx=tx[ix]
            blk = corr[:, :, iy, ix]                     # [C,C] -> sum F[c,p+dy,dx]F[c',p+dy',dx']
            # place into A: rows (c,iy,ix), cols (c',iy',ix')
            for iy2 in range(3):
                for ix2 in range(3):
                    # corr block for (iy2,ix2) gives sum F[c,p+dy2,dx2] F[c',p+dy2',dx2']
                    # we need cross term between tap (iy,ix) and (iy2,ix2):
                    #   sum F[c,p+ty[iy],tx[ix]] * F[c',p+ty[iy2],tx[ix2]]
                    # which is corr[c,c',iy2,ix2] with the *first* tap being (iy,ix)
                    # => need correlation at lag (ty[iy2]-ty[iy], tx[ix2]-tx[ix])
                    pass
            # direct computation below (clearer)
        # (unused)
    # --- direct, explicit construction (cheap enough: 9x9 taps) ---
    A = np.zeros((C * 9, C * 9), dtype=np.float64)
    # precompute shifted maps for each tap: S[c,iy,ix,n,p]
    S = np.zeros((C, 3, 3, N, H, Wd), dtype=np.float64)
    for iy, dy in enumerate(ty):
        for ix, dx in enumerate(tx):
            # S[c,iy,ix,n,p] = F[c,n,p+dy,dx] (zero outside)
            src_y0, src_y1 = max(0, -dy), min(H, H - dy)
            src_x0, src_x1 = max(0, -dx), min(Wd, Wd - dx)
            if src_y1 <= src_y0 or src_x1 <= src_x0:
                continue
            S[:, iy, ix, :, src_y0:src_y1, src_x0:src_x1] = \
                F[:, :, src_y0 + dy:src_y1 + dy, src_x0 + dx:src_x1 + dx]
    # A[(c,iy,ix),(c',jy,jx)] = sum_{n,p} S[c,iy,ix] S[c',jy,jx]
    # = sum over channel pairs of elementwise product sums
    for c in range(C):
        for jy in range(3):
            for jx in range(3):
                row = c * 9 + jy * 3 + jx
                Sc = S[c, jy, jx]                        # [N,H,Wd]
                # block over c'
                for c2 in range(C):
                    for iy2 in range(3):
                        for ix2 in range(3):
                            col = c2 * 9 + iy2 * 3 + ix2
                            A[row, col] = np.einsum('nhw,nhw->', Sc, S[c2, iy2, ix2])
    # rhs: b[(c,iy,ix)] = sum_{n,p} S[c,iy,ix] * Y[n,p]
    rhs = np.zeros(C * 9, dtype=np.float64)
    for c in range(C):
        for iy in range(3):
            for ix in range(3):
                rhs[c * 9 + iy * 3 + ix] = np.einsum('nhw,nhw->', S[c, iy, ix], Y)
    # bias column (constant 1 per pixel) -> augment
    n_pix = N * H * Wd
    ones = np.ones(n_pix, dtype=np.float64)
    # A_bb = n_pix ; A_b,k = sum_{n,p} S[k]
    Ab = np.empty(C * 9, dtype=np.float64)
    for c in range(C):
        for iy in range(3):
            for ix in range(3):
                Ab[c * 9 + iy * 3 + ix] = S[c, iy, ix].sum()
    Aaug = np.zeros((C * 9 + 1, C * 9 + 1), dtype=np.float64)
    Aaug[:C * 9, :C * 9] = A
    Aaug[:C * 9, C * 9] = Ab
    Aaug[C * 9, :C * 9] = Ab
    Aaug[C * 9, C * 9] = n_pix
    baug = np.zeros(C * 9 + 1, dtype=np.float64)
    baug[:C * 9] = rhs
    baug[C * 9] = Y.sum()
    lam = ridge * max(1.0, np.trace(Aaug) / (C * 9 + 1))
    Areg = Aaug + lam * np.eye(C * 9 + 1)
    sol = np.linalg.solve(Areg, baug)
    Wout = sol[:C * 9].reshape(C, 3, 3)[None, :, :, :].astype(np.float32)
    bout = np.array([sol[C * 9]], dtype=np.float32)
    return Wout, bout


def edit(w, f):
    w = {k: np.array(v, dtype=np.float32, copy=True) for k, v in w.items()}
    x = np.asarray(f["x"], dtype=np.float32)
    t = np.asarray(f["t"], dtype=np.int64)
    y = np.asarray(f["y"], dtype=np.float32)
    alphabar = np.asarray(f["alphabar"], dtype=np.float32)

    diag = {}

    # ------------------------------------------------------------------
    # 0) Identify the cached "features" array: post-ReLU conv3 output
    #    (== layer.out.input0) or pre-activation conv3 output.
    # ------------------------------------------------------------------
    feats = np.asarray(f["features"], dtype=np.float32)
    bidx = np.asarray(f["batch.ids"], dtype=np.int64)
    out_in = np.asarray(f["layer.out.input0"], dtype=np.float32)
    conv3_out = np.asarray(f["layer.conv3.output"], dtype=np.float32)
    if bidx.size and bidx.max() < feats.shape[0]:
        d_post = np.abs(feats[bidx] - out_in).mean()
        d_pre = np.abs(feats[bidx] - conv3_out).mean()
        if d_pre < d_post:
            feats = np.maximum(feats, 0.0)          # convert pre-activation to post-ReLU
        diag["feat_match_post"] = float(d_post)
        diag["feat_match_pre"] = float(d_pre)
    else:
        feats = np.maximum(feats, 0.0)

    # ------------------------------------------------------------------
    # 1) Re-shape conv1 so that its 32 channels carry *useful* per-pixel
    #    information about x_t: identity-like taps (so ReLU keeps the
    #    signal) plus a set of thresholds (so the readout can build
    #    piecewise-linear shrinkage functions of x_t).
    #    This is a fixed, hand-designed (closed-form) construction.
    # ------------------------------------------------------------------
    W1 = np.zeros((32, 2, 3, 3), dtype=np.float32)
    b1 = np.zeros(32, dtype=np.float32)
    # channels 0..3 : identity / smoothed identity / neighbor-average on x_t
    W1[0, 0, 1, 1] = 1.0
    W1[1, 0, 1, 1] = 0.5
    W1[1, 0, 0, 1] = 0.25
    W1[1, 0, 2, 1] = 0.25
    W1[2, 0, 1, 1] = 0.5
    W1[2, 0, 0, 0] = 0.25
    W1[2, 0, 2, 2] = 0.25
    W1[3, 0, 1, 1] = 0.5
    W1[3, 0, 0, 1] = 0.25
    W1[3, 0, 1, 0] = 0.25
    # channel 4 : t channel (broadcast) -> lets the net modulate by noise level
    W1[4, 1, 1, 1] = 1.0
    # channels 5..31 : thresholds on x_t: ReLU(x_t - a_k) with a_k spread
    # over the plausible range of x_t (roughly [-1.2, 1.2]).
    a_vals = np.linspace(-1.2, 1.2, 27).astype(np.float32)
    for k, a in enumerate(a_vals):
        W1[5 + k, 0, 1, 1] = 1.0
        b1[5 + k] = -float(a)
    # keep the original random structure for a few channels as "texture" probes
    w1_orig = np.asarray(w["conv1.weight"], dtype=np.float32)
    b1_orig = np.asarray(w["conv1.bias"], dtype=np.float32)
    for k in range(5):
        W1[5 + 27 + k] = w1_orig[k]
        b1[5 + 27 + k] = b1_orig[k]
    w["conv1.weight"] = W1
    w["conv1.bias"] = b1

    # ------------------------------------------------------------------
    # 2) conv2 / conv3 : make them near-identity (channel-wise) so that the
    #    post-ReLU features remain a rich, information-preserving set.
    #    A small amount of the original random weights is kept for diversity.
    # ------------------------------------------------------------------
    def near_identity(Cout, Cin, orig, frac=0.15):
        W = np.zeros((Cout, Cin, 3, 3), dtype=np.float32)
        m = min(Cout, Cin)
        for c in range(m):
            W[c, c, 1, 1] = 1.0
        # blend a little of the original random filters (scaled down)
        scale = 0.25
        W = W + scale * orig
        return W

    w["conv2.weight"] = near_identity(64, 32, np.asarray(w["conv2.weight"], dtype=np.float32))
    w["conv2.bias"] = np.asarray(w["conv2.bias"], dtype=np.float32) * 0.25
    w["conv3.weight"] = near_identity(64, 64, np.asarray(w["conv3.weight"], dtype=np.float32))
    w["conv3.bias"] = np.asarray(w["conv3.bias"], dtype=np.float32) * 0.25

    # ------------------------------------------------------------------
    # 3) Recompute the (now changed) hidden features for all training
    #    examples and refit the readout in closed form.
    # ------------------------------------------------------------------
    inp, h1, h2, h3, _ = _forward(w, x, t)
    feat_new = h3                                   # post-ReLU conv3 output

    # sanity: compare with cached features (they are stale now, but the
    # scale should be comparable)
    diag["feat_new_std"] = float(feat_new.std())
    diag["feat_cached_std"] = float(feats.std())

    # closed-form ridge fit of out.weight / out.bias on the new features
    Wout, bout = _fit_readout(feat_new, y, ridge=1e-3)
    w["out.weight"] = Wout.astype(np.float32)
    w["out.bias"] = bout.astype(np.float32)

    # ------------------------------------------------------------------
    # 4) Diagnostics: training MSE of the new predictor (per-example
    #    pixel-mean), plus per-timestep MSE, and comparison with the
    #    baseline (cached prediction).
    # ------------------------------------------------------------------
    _, _, _, _, pred_new = _forward(w, x, t)
    mse_new = float(((pred_new - y) ** 2).reshape(y.shape[0], -1).mean(axis=1).mean())
    pred_old = np.asarray(f["prediction"], dtype=np.float32)
    mse_old = float(((pred_old - y) ** 2).reshape(y.shape[0], -1).mean(axis=1).mean())
    diag["mse_new"] = mse_new
    diag["mse_old_cached"] = mse_old
    per_t = {}
    for tt in range(32):
        m = (t == tt)
        if m.sum() > 0:
            per_t[str(tt)] = float(((pred_new[m] - y[m]) ** 2).reshape(int(m.sum()), -1).mean(axis=1).mean())
    diag["mse_per_t"] = per_t
    diag["pred_std"] = float(pred_new.std())
    diag["y_std"] = float(y.std())

    # ------------------------------------------------------------------
    # 5) Safety: finite, float32, bounded magnitudes, shapes preserved.
    # ------------------------------------------------------------------
    for k in list(w.keys()):
        arr = np.asarray(w[k], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=100.0, neginf=-100.0)
        np.clip(arr, -1000.0, 1000.0, out=arr)
        w[k] = arr.astype(np.float32)

    return {"weights": w, "diagnostics": diag}
