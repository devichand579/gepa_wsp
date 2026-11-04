import numpy as np
import math

def edit(w, f):
    # ---------------- helpers ----------------
    def conv2d(x, W, b, pad=1):
        # x: [N,Cin,H,W], W: [Cout,Cin,kh,kw]
        N, Cin, H, Wd = x.shape
        Cout, _, kh, kw = W.shape
        if pad > 0:
            xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        else:
            xp = x
        Hp, Wp = xp.shape[2], xp.shape[3]
        oh, ow = Hp - kh + 1, Wp - kw + 1
        s = xp.strides
        cols = np.lib.stride_tricks.as_strided(
            xp, shape=(N, Cin, kh, kw, oh, ow),
            strides=(s[0], s[1], s[2], s[3], s[2], s[3]), writeable=False)
        cols = cols.reshape(N, Cin * kh * kw, oh * ow)
        out = np.einsum('oc,ncp->nop', W.reshape(Cout, -1), cols)
        out = out + b.reshape(1, Cout, 1)
        return out.reshape(N, Cout, oh, ow)

    def maxpool2(x):
        N, C, H, Wd = x.shape
        h2, w2 = H // 2, Wd // 2
        xr = x[:, :, :h2 * 2, :w2 * 2].reshape(N, C, h2, 2, w2, 2)
        return xr.max(axis=(3, 5))

    def forward(x, c1w, c1b, c2w, c2b, hw, hb):
        h = maxpool2(np.maximum(conv2d(x, c1w, c1b), 0.0))
        h = maxpool2(np.maximum(conv2d(h, c2w, c2b), 0.0))
        f = h.reshape(h.shape[0], -1)
        logits = f @ hw.T + hb
        return f, logits

    def softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    # ---------------- data ----------------
    X = np.asarray(f["x"], dtype=np.float32)
    Y = np.asarray(f["y"], dtype=np.int64)
    N = X.shape[0]
    Xf = X.reshape(N, -1).astype(np.float64)
    y = Y
    K = 10
    onehot = np.zeros((N, K))
    onehot[np.arange(N), y] = 1.0

    # ---------------- conv1: PCA of 3x3 patches ----------------
    # build patch matrix from training images (stride 1, all positions)
    P = np.pad(Xf.reshape(N, 28, 28), ((0, 0), (1, 1), (1, 1)))
    patches = np.empty((N, 26 * 26, 9), dtype=np.float64)
    idx = 0
    for r in range(26):
        for c in range(26):
            patches[:, idx, :] = P[:, r:r + 3, c:c + 3].reshape(N, 9)
            idx += 1
    pm = patches.reshape(-1, 9).mean(axis=0)
    pc = patches.reshape(-1, 9) - pm
    cov = (pc.T @ pc) / pc.shape[0]
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]
    evals = np.maximum(evals[order], 0.0)
    # 16 filters: 9 PCA directions + 7 extra (oriented edge/derivative filters)
    basis = evecs[:, :9].T.copy()          # [9,9]
    # extra filters: oriented first derivatives (Sobel-like) and laplacian
    extra = []
    for th in np.linspace(0, np.pi, 6, endpoint=False):
        dx, dy = math.cos(th), math.sin(th)
        g = np.zeros((3, 3))
        g[1, 0], g[1, 2] = -dx, dx
        g[0, 1], g[2, 1] = -dy, dy
        extra.append(g.reshape(-1))
    lap = np.zeros((3, 3)); lap[1, 1] = -4.0
    lap[0, 1] = lap[2, 1] = lap[1, 0] = lap[1, 2] = 1.0
    extra.append(lap.reshape(-1))
    extra.append(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64))
    extra = np.array(extra)
    # orthonormalize the whole set (Gram-Schmidt) for stability
    B = np.concatenate([basis, extra], axis=0)     # [16,9]
    Q, _ = np.linalg.qr(B.T)                        # [9,16]
    filt = Q.T                                      # [16,9]
    # scale so that each filter has unit response std on data
    resp_std = np.sqrt(np.maximum((filt ** 2) @ np.diag(cov), 1e-12))
    filt = filt / resp_std[None, :]
    c1w = filt.reshape(16, 1, 3, 3).astype(np.float32)
    c1b = np.zeros(16, dtype=np.float32)

    # ---------------- conv2: class-difference matched filters ----------------
    cls_mean = np.stack([Xf[y == c].mean(axis=0) for c in range(K)])   # [K,784]
    glob = cls_mean.mean(axis=0)
    diff = cls_mean - glob                                            # [K,784]
    # local templates: 4x4 blocks (conv2 effective pixel footprint ~4x4)
    # build template bank: for each class, several 4x4 local templates
    templates = []
    tclass = []
    for c in range(K):
        d = diff[c].reshape(28, 28)
        for (r0, c0) in [(2, 2), (6, 6), (10, 10), (14, 14), (18, 18), (22, 22),
                         (2, 12), (12, 2), (12, 12), (22, 22), (6, 18), (18, 6)]:
            t = d[r0:r0 + 4, c0:c0 + 4].reshape(-1)
            if np.linalg.norm(t) < 1e-8:
                continue
            templates.append(t)
            tclass.append(c)
    templates = np.array(templates)
    tclass = np.array(tclass)
    # each template lives in a 4x4 pixel footprint; conv1 filters are 3x3 patches.
    # Represent template as a 3x3 patch located at the template's 2x2-subsampled
    # position inside the conv2 3x3 window: use the 3x3 patch of the 4x4 template
    # that is centred (rows 0..2 / cols 0..2 of the 4x4) -> approximate.
    T3 = templates[:, :9]                              # [T,9] top-left 3x3 of each template
    # solve for conv1-space coefficients: want sum_k g_k * f_k ~= T3 (patch space)
    # f_k are 3x3 filters -> their 9-dim vectors; solve least squares in patch space
    Fmat = filt                                        # [16,9]
    # g [16] with Fmat^T g = T3^T  ->  g = pinv(Fmat^T) @ T3^T
    G = np.linalg.pinv(Fmat.T)                         # [9,16]
    gcoef = (G @ T3.T).T                               # [T,16]
    # build conv2 filters: 32 channels; each channel = one template detector
    # channel c uses spatial 3x3 window with the template applied at centre
    # (we approximate by placing the 3x3 template patch at the centre of the
    #  3x3 conv window, i.e. offset (0,0) of the window)
    T = templates.shape[0]
    c2w = np.zeros((32, 16, 3, 3), dtype=np.float64)
    c2b = np.zeros(32, dtype=np.float64)
    for t in range(T):
        ch = t % 32
        for k in range(16):
            c2w[ch, k, 1, 1] += gcoef[t, k]
        # bias: threshold so relu keeps only strong matches (one-vs-rest gate)
        c2b[ch] = 0.0
    # scale conv2 filters to a sensible dynamic range
    c2w = c2w / 3.0
    c2b = c2b.astype(np.float32)

    # ---------------- compute features with new conv stack ----------------
    Xb = X[:N]
    Fnew, Lnew = forward(Xb, c1w, c1b, c2w.astype(np.float32), c2b,
                         np.zeros((K, 1568), dtype=np.float32),
                         np.zeros(K, dtype=np.float32))

    # ---------------- closed-form head: regularized LDA ----------------
    def fit_lda(F, y, lam_rel):
        Nf, D = F.shape
        mu = F.mean(axis=0)
        Fc = F - mu
        means = np.stack([F[y == c].mean(axis=0) for c in range(K)])
        Sw = np.zeros((D, D))
        for c in range(K):
            Z = Fc[y == c]
            Sw += Z.T @ Z
        Sw /= max(Nf - K, 1)
        tr = np.trace(Sw) / D
        Sw += lam_rel * max(tr, 1e-12) * np.eye(D)
        try:
            M = np.linalg.solve(Sw, means.T)          # [D,K]
        except np.linalg.LinAlgError:
            M = np.linalg.lstsq(Sw, means.T, rcond=None)[0]
        W = M.T                                        # [K,D]
        b = -0.5 * np.sum(means * M.T, axis=1)
        b = b + np.log(np.bincount(y, minlength=K) / Nf)
        return W, b

    def fit_ridge(F, y, lam_rel):
        Nf, D = F.shape
        mu = F.mean(axis=0)
        Yc = onehot - onehot.mean(axis=0)
        Fm = F - mu
        A = Fm.T @ Fm
        tr = np.trace(A) / D
        A = A + lam_rel * max(tr, 1e-12) * np.eye(D)
        W = np.linalg.solve(A, Fm.T @ Yc)
        b = onehot.mean(axis=0) - mu @ W.T
        return W, b

    def acc(F, W, b):
        pred = np.argmax(F @ W.T + b, axis=1)
        return float((pred == y).mean())

    # candidate heads (closed-form, tiny fixed set)
    cands = []
    for lam in (0.3, 1.0, 3.0, 10.0, 30.0):
        Wl, bl = fit_lda(Fnew, y, lam)
        cands.append(('lda_new_%g' % lam, Wl, bl))
        Wr, br = fit_ridge(Fnew, y, lam)
        cands.append(('ridge_new_%g' % lam, Wr, br))
    # also original features with original convs (baseline)
    c1w0 = np.asarray(w["conv1.weight"], dtype=np.float32)
    c1b0 = np.asarray(w["conv1.bias"], dtype=np.float32)
    c2w0 = np.asarray(w["conv2.weight"], dtype=np.float32)
    c2b0 = np.asarray(w["conv2.bias"], dtype=np.float32)
    Fold, _ = forward(Xb, c1w0, c1b0, c2w0, c2b0,
                      np.zeros((K, 1568), dtype=np.float32), np.zeros(K, dtype=np.float32))
    for lam in (1.0, 10.0):
        Wl, bl = fit_lda(Fold, y, lam)
        cands.append(('lda_old_%g' % lam, Wl, bl))

    # honest selection: split pool into fit/holdout (deterministic)
    rng = np.random.RandomState(0)
    perm = rng.permutation(N)
    ntr = int(0.75 * N)
    tr_idx, va_idx = perm[:ntr], perm[ntr:]

    def fit_eval(F, kind, lam):
        if kind == 'lda':
            W, b = fit_lda(F[tr_idx], y[tr_idx], lam)
        else:
            W, b = fit_ridge(F[tr_idx], y[tr_idx], lam)
        pred = np.argmax(F[va_idx] @ W.T + b, axis=1)
        return float((pred == y[va_idx]).mean()), W, b

    best = None
    for name, _, _ in cands:
        kind = 'lda' if name.startswith('lda') else 'ridge'
        lam = float(name.split('_')[-1])
        Fmat = Fold if 'old' in name else Fnew
        va, W, b = fit_eval(Fmat, kind, lam)
        if best is None or va > best[0]:
            best = (va, name, W, b, Fmat)

    va_acc, name, W, b, Fmat = best

    # refit chosen head on the FULL pool
    kind = 'lda' if name.startswith('lda') else 'ridge'
    lam = float(name.split('_')[-1])
    if kind == 'lda':
        Wf, bf = fit_lda(Fmat, y, lam)
    else:
        Wf, bf = fit_ridge(Fmat, y, lam)

    # ---------------- write back ----------------
    w["conv1.weight"] = np.clip(c1w, -1000, 1000).astype(np.float32)
    w["conv1.bias"] = np.clip(c1b, -1000, 1000).astype(np.float32)
    w["conv2.weight"] = np.clip(c2w.astype(np.float32), -1000, 1000).astype(np.float32)
    w["conv2.bias"] = np.clip(c2b, -1000, 1000).astype(np.float32)
    w["head.weight"] = np.clip(Wf, -1000, 1000).astype(np.float32)
    w["head.bias"] = np.clip(bf, -1000, 1000).astype(np.float32)

    # diagnostics
    train_acc = float((np.argmax(Fmat @ Wf.T + bf, axis=1) == y).mean())
    old_head_acc = float((np.argmax(np.asarray(f["logits"], dtype=np.float64), axis=1) == y).mean())
    P = softmax(Fmat @ Wf.T + bf)
    ce = float(-np.log(np.maximum(P[np.arange(N), y], 1e-12)).mean())
    diag = {
        "chosen_head": name,
        "holdout_acc_75_25": round(va_acc, 4),
        "train_acc_full": round(train_acc, 4),
        "old_acc": round(old_head_acc, 4),
        "cross_entropy_new": round(ce, 4),
        "feat_mean_new": round(float(Fnew.mean()), 5),
        "feat_std_new": round(float(Fnew.std()), 5),
        "feat_mean_old": round(float(Fold.mean()), 5),
        "feat_std_old": round(float(Fold.std()), 5),
        "head_w_std": round(float(Wf.std()), 5),
        "head_w_absmax": round(float(np.abs(Wf).max()), 4),
        "conv1_absmax": round(float(np.abs(c1w).max()), 4),
        "conv2_absmax": round(float(np.abs(c2w).max()), 4),
        "n_train": int(N),
        "n_classes": int(K),
    }
    return {"weights": w, "diagnostics": diag}
