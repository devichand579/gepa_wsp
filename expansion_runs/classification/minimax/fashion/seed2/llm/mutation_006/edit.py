def edit(w, f):
    import numpy as np

    # ---------------- helpers ----------------
    def conv2d(x, W, b, pad=1):
        # x: (N,C,H,W), W: (O,C,kh,kw) -> (N,O,H',W')
        N, C, H, Wd = x.shape
        O, _, kh, kw = W.shape
        xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        Ho = H + 2 * pad - kh + 1
        Wo = Wd + 2 * pad - kw + 1
        s = xp.strides
        cols = np.lib.stride_tricks.as_strided(
            xp, shape=(N, C, kh, kw, Ho, Wo),
            strides=(s[0], s[1], s[2], s[3], s[2], s[3]), writeable=False)
        cols = cols.reshape(N, C * kh * kw, Ho * Wo)
        out = np.einsum('ncp,oc->nop', cols, W.reshape(O, -1), optimize=True)
        out = out.reshape(N, O, Ho, Wo) + b[None, :, None, None]
        return out

    def maxpool2(x):
        N, C, H, Wd = x.shape
        h = H // 2
        r = x[:, :, :2 * h, :2 * Wd // 2].reshape(N, C, h, 2, Wd // 2, 2)
        return r.max(axis=(3, 5))

    def forward_feats(x, c1w, c1b, c2w, c2b):
        h = np.maximum(conv2d(x, c1w, c1b), 0.0)
        h = maxpool2(h)
        h = np.maximum(conv2d(h, c2w, c2b), 0.0)
        h = maxpool2(h)
        return h.reshape(h.shape[0], -1)

    def softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def fit_head(F, y, K, lam, mode):
        N, D = F.shape
        mu = F.mean(axis=0)
        Fc = F - mu
        Y = np.zeros((N, K), dtype=np.float64)
        Y[np.arange(N), y] = 1.0
        Yc = Y - Y.mean(axis=0, keepdims=True)
        if mode == 'ridge':
            A = Fc.T @ Fc + lam * N / D * np.eye(D)
            B = Fc.T @ Yc
            W = np.linalg.solve(A, B)
        else:  # regularized LDA (shared within-class covariance)
            Sw = np.zeros((D, D))
            for c in range(K):
                Fc_c = F[y == c] - F[y == c].mean(axis=0, keepdims=True)
                Sw += Fc_c.T @ Fc_c
            Sw /= N
            tr = np.trace(Sw) / D
            Sw = (1.0 - lam) * Sw + lam * tr * np.eye(D)
            Mc = np.zeros((K, D))
            for c in range(K):
                Mc[c] = F[y == c].mean(axis=0)
            gm = Mc.mean(axis=0)
            W = np.linalg.solve(Sw, (Mc - gm).T).T
            b = -0.5 * np.sum(Mc * W, axis=1) + np.log(np.bincount(y, minlength=K) / N)
            return W.astype(np.float32), b.astype(np.float32), mu.astype(np.float32)
        b = Yc.mean(axis=0) - W.T @ mu
        return W.astype(np.float32), b.astype(np.float32), mu.astype(np.float32)

    def apply_head(F, W, b, mu):
        return (F - mu) @ W.T + b

    def acc(logits, y):
        return float((logits.argmax(axis=1) == y).mean())

    # ---------------- data ----------------
    x = np.asarray(f["x"], dtype=np.float32)
    y = np.asarray(f["y"]).astype(np.int64)
    N = x.shape[0]
    K = 10
    xf = x.reshape(N, 1, 28, 28)

    c1w0 = np.asarray(w["conv1.weight"], dtype=np.float32)
    c1b0 = np.asarray(w["conv1.bias"], dtype=np.float32)
    c2w0 = np.asarray(w["conv2.weight"], dtype=np.float32)
    c2b0 = np.asarray(w["conv2.bias"], dtype=np.float32)

    # sanity: reproduce cached features with current weights
    F0 = forward_feats(xf, c1w0, c1b0, c2w0, c2b0)
    F0c = np.asarray(f["features"], dtype=np.float32)
    repro_err = float(np.abs(F0 - F0c).max()) if F0.shape == F0c.shape else -1.0

    # ---------------- candidate conv designs ----------------
    # patch matrix for PCA-based designs
    patches = []
    for i in range(N):
        im = xf[i, 0]
        p = np.lib.stride_tricks.as_strided(
            im, shape=(28, 28, 3, 3), strides=(im.strides[0], im.strides[1], im.strides[0], im.strides[1]),
            writeable=False)
        patches.append(p.reshape(784, 9))
    P = np.concatenate(patches, axis=0).astype(np.float64)
    Pm = P.mean(axis=0)
    Pc = P - Pm
    # top PCA directions of 3x3 patches (9 dims max)
    U, S, Vt = np.linalg.svd(Pc, full_matrices=False)
    patch_dirs = Vt[:9]                     # (9,9) orthonormal basis of patch space

    # 5x5 patch PCA (for conv2 realization)
    patches5 = []
    for i in range(N):
        im = xf[i, 0]
        p = np.lib.stride_tricks.as_strided(
            im, shape=(26, 26, 5, 5),
            strides=(im.strides[0], im.strides[1], im.strides[0], im.strides[1]), writeable=False)
        patches5.append(p.reshape(676, 25))
    P5 = np.concatenate(patches5, axis=0).astype(np.float64)
    P5m = P5.mean(axis=0)
    P5c = P5 - P5m
    U5, S5, Vt5 = np.linalg.svd(P5c, full_matrices=False)
    filt5 = Vt5[:25]                        # (25,25) top PCA filters of 5x5 patches

    # conv1 design A: 9 smoothed deltas + 7 extra smoothed patch-PCA filters
    def smooth_kernel():
        return np.ones((3, 3)) / 9.0

    def build_conv1():
        W1 = np.zeros((16, 1, 3, 3), dtype=np.float64)
        sk = smooth_kernel()
        for c in range(9):
            r, cc = divmod(c, 3)
            W1[c, 0, r, cc] = 1.0
            W1[c, 0] = W1[c, 0] * 0.0
            W1[c, 0, r, cc] = 1.0
        # smoothed deltas: conv output at (i,j) = sum over 3x3 window of image * kernel
        for c in range(9):
            r, cc = divmod(c, 3)
            W1[c, 0] = 0.0
            W1[c, 0, r, cc] = 1.0
        # extra channels: smoothed versions of the 7 remaining patch-PCA directions
        for j in range(7):
            d = patch_dirs[9 + j] if 9 + j < 9 else patch_dirs[j % 9]
            # project onto non-negative smoothing: use |d| smoothed
            dd = np.abs(d).reshape(3, 3)
            W1[9 + j, 0] = dd / max(dd.sum(), 1e-6)
        return W1.astype(np.float32)

    def realize_5x5_in_conv2(g5, W1):
        """express 5x5 filter g5 as sum over conv1 channels c and 3x3 offsets (dy,dx)
        of a[c,dy,dx] * shift(conv1_c) ; returns (16,3,3) coefficients."""
        # basis: for each (c,dy,dx) the 5x5 filter = shift of W1[c] placed at (dy+1,dx+1)
        basis = np.zeros((16 * 9, 25))
        for c in range(16):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    idx = (c * 9) + (dy + 1) * 3 + (dx + 1)
                    k = W1[c, 0]
                    for r in range(3):
                        for cc in range(3):
                            basis[idx, (r + dy + 1) * 5 + (cc + dx + 1)] = k[r, cc]
        a, *_ = np.linalg.lstsq(basis, g5.reshape(25), rcond=None)
        return a.reshape(16, 3, 3)

    W1A = build_conv1()
    W2A = np.zeros((32, 16, 3, 3), dtype=np.float64)
    for k in range(25):
        W2A[k] = realize_5x5_in_conv2(filt5[k], W1A)
    for k in range(25, 32):
        W2A[k] = realize_5x5_in_conv2(np.roll(filt5[k % 25], 1), W1A)
    W2A = W2A.astype(np.float32)
    b1A = np.zeros(16, dtype=np.float32)
    b2A = np.zeros(32, dtype=np.float32)

    # ---------------- validation split (stratified, deterministic) ----------------
    rng = np.random.RandomState(0)
    tr_idx = []
    va_idx = []
    for c in range(K):
        idx = np.where(y == c)[0]
        idx = idx[rng.permutation(len(idx))]
        ntr = int(round(0.8 * len(idx)))
        tr_idx.append(idx[:ntr])
        va_idx.append(idx[ntr:])
    tr_idx = np.concatenate(tr_idx)
    va_idx = np.concatenate(va_idx)

    def eval_design(F, tag):
        res = {}
        best = (-1.0, None)
        for mode in ("ridge", "lda"):
            for lam in (0.02, 0.1, 0.3, 0.6, 0.9):
                Wf, bf, muf = fit_head(F[tr_idx], y[tr_idx], K, lam, mode)
                lg = apply_head(F[va_idx], Wf, bf, muf)
                a = acc(lg, y[va_idx])
                res["%s_%s_%.2f" % (tag, mode, lam)] = round(a, 4)
                if a > best[0]:
                    best = (a, (mode, lam))
        return best, res

    cands = []
    b0, r0 = eval_design(F0, "base")
    cands.append(("base", F0, b0[1], b0[0]))
    diag = {"repro_max_abs_err": round(repro_err, 6), "val": {}}
    diag["val"].update(r0)

    FA = forward_feats(xf, W1A, b1A, W2A, b2A)
    bA, rA = eval_design(FA, "pca")
    cands.append(("pca", FA, bA[1], bA[0]))
    diag["val"].update(rA)

    # also: base features but with conv2 replaced by PCA-5x5 filters over ORIGINAL conv1
    # (realize 5x5 filters using the original conv1 basis if it spans enough)
    try:
        W2B = np.zeros((32, 16, 3, 3), dtype=np.float64)
        for k in range(25):
            W2B[k] = realize_5x5_in_conv2(filt5[k], c1w0.astype(np.float64))
        for k in range(25, 32):
            W2B[k] = realize_5x5_in_conv2(np.roll(filt5[k % 25], 1), c1w0.astype(np.float64))
        FB = forward_feats(xf, c1w0, c1b0, W2B.astype(np.float32), np.zeros(32, dtype=np.float32))
        bB, rB = eval_design(FB, "mix")
        cands.append(("mix", FB, bB[1], bB[0]))
        diag["val"].update(rB)
    except Exception:
        pass

    tag, Fbest, (mode, lam), vacc = max(cands, key=lambda t: t[3])
    diag["chosen_design"] = tag
    diag["chosen_val_acc"] = round(vacc, 4)
    diag["chosen_head"] = "%s lam=%.2f" % (mode, lam)

    # ---------------- final fit on ALL permitted samples ----------------
    Wf, bf, muf = fit_head(Fbest, y, K, lam, mode)
    lg = apply_head(Fbest, Wf, bf, muf)
    p = softmax(lg)
    diag["train_acc"] = round(acc(lg, y), 4)
    diag["train_ce"] = round(float(-np.log(np.clip(p[np.arange(N), y], 1e-9, 1)).mean()), 4)
    diag["pred_hist"] = np.bincount(lg.argmax(axis=1), minlength=K).tolist()
    diag["true_hist"] = np.bincount(y, minlength=K).tolist()
    diag["feat_mean_abs"] = round(float(np.abs(Fbest).mean()), 5)
    diag["head_w_std"] = round(float(Wf.std()), 5)

    # ---------------- write back ----------------
    if tag == "base":
        w["conv1.weight"] = c1w0.copy()
        w["conv1.bias"] = c1b0.copy()
        w["conv2.weight"] = c2w0.copy()
        w["conv2.bias"] = c2b0.copy()
    elif tag == "pca":
        w["conv1.weight"] = W1A.copy()
        w["conv1.bias"] = b1A.copy()
        w["conv2.weight"] = W2A.copy()
        w["conv2.bias"] = b2A.copy()
    else:
        w["conv1.weight"] = c1w0.copy()
        w["conv1.bias"] = c1b0.copy()
        w["conv2.weight"] = W2B.astype(np.float32).copy()
        w["conv2.bias"] = np.zeros(32, dtype=np.float32)

    w["head.weight"] = Wf.astype(np.float32)
    w["head.bias"] = bf.astype(np.float32)

    # safety: finite, bounded
    for k in list(w.keys()):
        arr = np.asarray(w[k], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=100.0, neginf=-100.0)
        np.clip(arr, -1000.0, 1000.0, out=arr)
        w[k] = arr.astype(np.float32)

    return {"weights": w, "diagnostics": diag}
