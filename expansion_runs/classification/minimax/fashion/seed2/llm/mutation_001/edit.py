import numpy as np

# ---------------------------------------------------------------- utilities
def _conv2d(x, W, b, pad=1):
    """x:[N,C,H,W]  W:[O,C,kh,kw] -> [N,O,H,W]  cross-correlation, zero pad."""
    N, C, H, Wd = x.shape
    O, C2, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    Ho = xp.shape[2] - kh + 1
    Wo = xp.shape[3] - kw + 1
    # im2col
    s = xp.strides
    patches = np.lib.stride_tricks.as_strided(
        xp,
        shape=(N, C, kh, kw, Ho, Wo),
        strides=(s[0], s[1], s[2], s[3], s[2], s[3]),
        writeable=False)
    cols = patches.transpose(0, 4, 5, 1, 2, 3).reshape(N * Ho * Wo, C * kh * kw)
    out = cols @ W.reshape(O, C * kh * kw).T
    out = out + b[None, :]
    return out.reshape(N, Ho, Wo, O).transpose(0, 3, 1, 2)


def _maxpool2(h):
    N, C, H, Wd = h.shape
    H2, W2 = H // 2, Wd // 2
    r = h[:, :, :H2 * 2, :W2 * 2].reshape(N, C, H2, 2, W2, 2)
    return r.max(axis=(3, 5))


def _forward(x, cw1, cb1, cw2, cb2, hw, hb):
    h = _maxpool2(np.maximum(_conv2d(x, cw1, cb1, 1), 0.0))
    h = _maxpool2(np.maximum(_conv2d(h, cw2, cb2, 1), 0.0))
    f = h.reshape(h.shape[0], -1)
    return f, f @ hw.T + hb


def _rot_kernel(theta_deg, f, phase, sigma=1.0):
    """Continuous Gabor sampled on the 3x3 pixel grid (bilinear)."""
    th = np.deg2rad(theta_deg)
    ct, st = np.cos(th), np.sin(th)
    xs = np.array([-1.0, 0.0, 1.0])
    gx, gy = np.meshgrid(xs, xs)          # row=y, col=x
    u = gx * ct + gy * st
    v = -gx * st + gy * ct
    env = np.exp(-(gx ** 2 + gy ** 2) / (2.0 * sigma ** 2))
    return env * np.cos(2.0 * np.pi * f * u + phase)


def _oriented_line(theta_deg):
    """3-tap oriented line (even) template, rotated with bilinear sampling."""
    th = np.deg2rad(theta_deg)
    ct, st = np.cos(th), np.sin(th)
    xs = np.array([-1.0, 0.0, 1.0])
    gx, gy = np.meshgrid(xs, xs)
    u = gx * ct + gy * st
    v = -gx * st + gy * ct
    # line along v: sample a 1-D profile in u
    prof = np.cos(np.pi * u)          # +1,-1,+1 across the line
    # bilinear resample of the continuous template onto pixel centres
    return _bilinear_sample(prof, u, v)


def _bilinear_sample(func_vals, u, v):
    """Sample a separable template defined on continuous coords (u,v) at pixel grid."""
    # build template on a fine continuous grid then bilinear-sample at pixel centres
    lin = np.linspace(-1.5, 1.5, 61)
    U, V = np.meshgrid(lin, lin)
    T = func_vals(U, V)
    # pixel coords: row=y, col=x ; u = x ct + y st ; v = -x st + y ct
    th = np.deg2rad(_CURRENT_THETA[0])
    ct, st = np.cos(th), np.sin(th)
    xs = np.array([-1.0, 0.0, 1.0])
    gx, gy = np.meshgrid(xs, xs)
    u = gx * ct + gy * st
    v = -gx * st + gy * ct
    # invert: x = u ct - v st ; y = u st + v ct
    px = u * ct - v * st
    py = u * st + v * ct
    fx = (px + 1.5) / 3.0 * 60.0
    fy = (py + 1.5) / 3.0 * 60.0
    fx = np.clip(fx, 0, 60 - 1e-6)
    fy = np.clip(fy, 0, 60 - 1e-6)
    x0 = np.floor(fx).astype(int); y0 = np.floor(fy).astype(int)
    x1 = np.minimum(x0 + 1, 59); y1 = np.minimum(y0 + 1, 59)
    wx = fx - x0; wy = fy - y0
    T00 = T[y0, x0]; T01 = T[y0, x1]; T10 = T[y1, x0]; T11 = T[y1, x1]
    return (T00 * (1 - wx) * (1 - wy) + T01 * wx * (1 - wy) +
            T10 * (1 - wx) * wy + T11 * wx * wy)


_CURRENT_THETA = [0.0]


def _oriented_even(theta_deg, f=0.5):
    """Even-symmetric oriented (line) filter."""
    _CURRENT_THETA[0] = theta_deg
    th = np.deg2rad(theta_deg)
    ct, st = np.cos(th), np.sin(th)

    def T(U, V):
        u = U * ct + V * st
        v = -U * st + V * ct
        env = np.exp(-(U ** 2 + V ** 2) / (2.0 * 1.0 ** 2))
        return env * np.cos(2.0 * np.pi * f * u)

    return _bilinear_sample(T, None, None)


def _oriented_odd(theta_deg, f=0.5):
    """Odd-symmetric oriented (edge) filter."""
    _CURRENT_THETA[0] = theta_deg
    th = np.deg2rad(theta_deg)
    ct, st = np.cos(th), np.sin(th)

    def T(U, V):
        u = U * ct + V * st
        v = -U * st + V * ct
        env = np.exp(-(U ** 2 + V ** 2) / (2.0 * 1.0 ** 2))
        return env * np.sin(2.0 * np.pi * f * u)

    return _bilinear_sample(T, None, None)


def _dog(sigma_c, sigma_s):
    lin = np.linspace(-1.5, 1.5, 61)
    U, V = np.meshgrid(lin, lin)
    T = np.exp(-(U ** 2 + V ** 2) / (2 * sigma_c ** 2)) - \
        np.exp(-(U ** 2 + V ** 2) / (2 * sigma_s ** 2))
    xs = np.array([-1.0, 0.0, 1.0])
    gx, gy = np.meshgrid(xs, xs)
    fx = np.clip((gx + 1.5) / 3.0 * 60.0, 0, 60 - 1e-6)
    fy = np.clip((gy + 1.5) / 3.0 * 60.0, 0, 60 - 1e-6)
    x0 = np.floor(fx).astype(int); y0 = np.floor(fy).astype(int)
    x1 = np.minimum(x0 + 1, 59); y1 = np.minimum(y0 + 1, 59)
    wx = fx - x0; wy = fy - y0
    return (T[y0, x0] * (1 - wx) * (1 - wy) + T[y0, x1] * wx * (1 - wy) +
            T[y1, x0] * (1 - wx) * wy + T[y1, x1] * wx * wy)


def _design_conv1():
    """16 structured first-layer filters (4 orientations x 4 filter types)."""
    f1 = np.zeros((16, 1, 3, 3), dtype=np.float64)
    ors = [0.0, 45.0, 90.0, 135.0]
    for i, th in enumerate(ors):
        f1[4 * i + 0, 0] = _oriented_odd(th, f=0.5)     # edge, odd
        f1[4 * i + 1, 0] = _oriented_even(th, f=0.5)    # line, even
        f1[4 * i + 2, 0] = _oriented_odd(th, f=0.85)    # finer edge, odd
        f1[4 * i + 3, 0] = _oriented_even(th, f=0.85)   # finer line, even
    # keep scale sane
    s = f1.std()
    f1 = f1 / max(s, 1e-9) * 0.30
    return f1.astype(np.float32)


def _design_conv2():
    """32 structured second-layer filters over the 16 conv1 channels."""
    f2 = np.zeros((32, 16, 3, 3), dtype=np.float64)
    # channel layout of conv1: ch = 4*orient + type  (type: odd, even, odd_fine, even_fine)
    def ch(o, t):
        return 4 * o + t
    for o in range(4):
        # (a) parallel-edge / texture: same orientation, two taps separated by 2 px
        for off in [(0, 2), (2, 0), (1, 1), (2, 2)]:
            pass
        # (b) corner / junction: product of two orientations at neighbouring taps
        for o2 in range(o + 1, 4):
            idx = 4 * o + 4 * o2
            if idx >= 32:
                continue
            f2[idx, ch(o, 0), 1, 1] = 1.0
            f2[idx, ch(o2, 0), 1, 1] = 1.0
            f2[idx, ch(o, 1), 0, 1] = 0.5
            f2[idx, ch(o2, 1), 1, 0] = 0.5
    # texture detectors: oriented channel correlated with itself at offset
    k = 0
    for o in range(4):
        for (dr, dc) in [(0, 2), (2, 0), (1, 1), (2, 2)]:
            if k >= 32:
                break
            f2[k, ch(o, 0), 1 + dr // 2, 1 + dc // 2] = 1.0
            f2[k, ch(o, 0), 1 - dr // 2, 1 - dc // 2] = 1.0
            f2[k, ch(o, 1), 1 + dr // 2, 1 + dc // 2] = 0.7
            f2[k, ch(o, 1), 1 - dr // 2, 1 - dc // 2] = 0.7
            k += 1
    # local energy / smoothing filters per orientation (even channel, box sum)
    for o in range(4):
        if k >= 32:
            break
        for (dr, dc) in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            if k >= 32:
                break
            f2[k, ch(o, 1), 1 + dr, 1 + dc] = 1.0
            f2[k, ch(o, 1), 1 - dr, 1 - dc] = 1.0
            f2[k, ch(o, 0), 1, 1] = 0.5
            k += 1
    # remaining slots: isotropic DoG-like channel mixes
    while k < 32:
        f2[k, ch(k % 4, 1), 1, 1] = 1.0
        f2[k, ch((k + 1) % 4, 0), 0, 0] = 0.5
        f2[k, ch((k + 2) % 4, 3), 2, 2] = 0.5
        k += 1
    s = f2.std()
    f2 = f2 / max(s, 1e-9) * 0.12
    return f2.astype(np.float32)


def _ridge_fit(F, Y, lam):
    """Closed-form ridge (normal equations) with intercept handled by centering."""
    N, D = F.shape
    mu = F.mean(axis=0)
    Fc = F - mu
    Yc = Y - Y.mean(axis=0)
    G = Fc.T @ Fc
    G.flat[:: D + 1] += lam
    W = np.linalg.solve(G, Fc.T @ Yc)
    b = Y.mean(axis=0) - mu @ W
    return W.astype(np.float32), b.astype(np.float32)


def _ridge_fit_loo(F, Y, lam):
    """Ridge fit + closed-form leave-one-out errors via eigendecomposition."""
    N, D = F.shape
    mu = F.mean(axis=0)
    Fc = F - mu
    Yc = Y - Y.mean(axis=0)
    G = Fc.T @ Fc
    evals, evecs = np.linalg.eigh(G)
    evals = np.clip(evals, 0.0, None)
    VtY = evecs.T @ (Fc.T @ Yc)              # D x K
    FcV = Fc @ evecs                         # N x D
    d = evals + lam
    # LOO hat diagonal for ridge (in feature space): h_ii = sum_j u_ij^2/(s_j+lam)
    # u_ij = (Fc V)_ij / sqrt(s_j)
    Z = FcV / np.sqrt(np.maximum(evals, 1e-12))
    hdiag = (Z ** 2 / d[None, :]).sum(axis=1)
    hdiag = np.clip(hdiag, 0.0, 1.0 - 1e-9)
    Yhat = (Z ** 2 / d[None, :]) @ VtY      # N x K
    resid = Yc - Yhat
    loo = resid / (1.0 - hdiag)[:, None]
    pred_loo = Y.mean(axis=0)[None, :] + Yhat + loo
    W = evecs @ (VtY / d[:, None])
    b = Y.mean(axis=0) - mu @ W
    acc_loo = float((pred_loo.argmax(axis=1) == Y.argmax(axis=1)).mean())
    return W.astype(np.float32), b.astype(np.float32), acc_loo


def edit(w, f):
    x = np.asarray(f["x"], dtype=np.float32)
    y = np.asarray(f["y"]).astype(np.int64)
    N = x.shape[0]

    cw1 = np.asarray(w["conv1.weight"], dtype=np.float32)
    cb1 = np.asarray(w["conv1.bias"], dtype=np.float32)
    cw2 = np.asarray(w["conv2.weight"], dtype=np.float32)
    cb2 = np.asarray(w["conv2.bias"], dtype=np.float32)
    hw = np.asarray(w["head.weight"], dtype=np.float32)
    hb = np.asarray(w["head.bias"], dtype=np.float32)

    diag = {}

    # ---- validate my forward implementation against cached activations ----
    idx = np.asarray(f["batch.ids"], dtype=np.int64)
    fb, _ = _forward(x[idx], cw1, cb1, cw2, cb2, hw, hb)
    ref = np.asarray(f["layer.head.input0"], dtype=np.float32)
    diag["fwd_check_max_abs_feat_err"] = float(np.abs(fb - ref).max())
    lg, _ = _forward(x[idx], cw1, cb1, cw2, cb2, hw, hb)
    refl = np.asarray(f["layer.head.output"], dtype=np.float32)
    diag["fwd_check_max_abs_logit_err"] = float(np.abs(lg - refl).max())

    # ---- design new conv stack ----
    ncw1 = _design_conv1()
    ncb1 = np.zeros(16, dtype=np.float32)
    ncw2 = _design_conv2()
    ncb2 = np.zeros(32, dtype=np.float32)

    # ---- compute pooled features for the whole permitted pool ----
    feats = np.empty((N, 1568), dtype=np.float32)
    B = 250
    for i in range(0, N, B):
        xb = x[i:i + B]
        h = _maxpool2(np.maximum(_conv2d(xb, ncw1, ncb1, 1), 0.0))
        h = _maxpool2(np.maximum(_conv2d(h, ncw2, ncb2, 1), 0.0))
        feats[i:i + B] = h.reshape(h.shape[0], -1)
    diag["feat_mean"] = float(feats.mean())
    diag["feat_std"] = float(feats.std())
    diag["feat_zero_frac"] = float((feats == 0).mean())

    # ---- closed-form ridge head, lambda chosen by closed-form LOO (PRESS) ----
    Y = np.zeros((N, 10), dtype=np.float64)
    Y[np.arange(N), y] = 1.0
    F = feats.astype(np.float64)
    sc = F.std(axis=0)
    sc[sc < 1e-8] = 1.0
    Fs = F / sc

    lams = [1e1, 1e2, 1e3, 1e4, 1e5]
    best = None
    loo_accs = []
    for lam in lams:
        Wl, bl, al = _ridge_fit_loo(Fs, Y, lam)
        loo_accs.append((float(lam), float(al)))
        if best is None or al > best[2]:
            best = (Wl, bl, al, lam)
    Wl, bl, acc_loo, lam_star = best
    diag["loo_grid"] = loo_accs
    diag["lambda_star"] = float(lam_star)
    diag["loo_acc"] = float(acc_loo)

    # rescale ridge weights back to raw feature units
    headW = (Wl / sc[:, None]).astype(np.float32)
    headB = bl.astype(np.float32)

    # train accuracy of the closed-form head
    logits = feats @ headW.T + headB
    diag["train_acc"] = float((logits.argmax(axis=1) == y).mean())
    diag["train_acc_per_class"] = [float((logits[y == k].argmax(axis=1) == k).mean())
                                   for k in range(10)]
    diag["pred_hist"] = np.bincount(logits.argmax(axis=1), minlength=10).tolist()

    # ---- write back ----
    w["conv1.weight"] = ncw1.astype(np.float32)
    w["conv1.bias"] = ncb1.astype(np.float32)
    w["conv2.weight"] = ncw2.astype(np.float32)
    w["conv2.bias"] = ncb2.astype(np.float32)
    w["head.weight"] = headW
    w["head.bias"] = headB

    for k in w:
        arr = np.asarray(w[k], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1000.0, neginf=-1000.0)
        np.clip(arr, -1000.0, 1000.0, out=arr)
        w[k] = arr

    return {"weights": w, "diagnostics": diag}
