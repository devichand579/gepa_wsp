def edit(w, f):
    import numpy as np

    # ---------------- helpers -------------------------------------------------
    def conv2d(x, W, b, pad=1):
        # x: [N,C,H,W], W: [O,C,kh,kw] -> [N,O,H,W]  (cross-correlation, stride 1)
        N, C, H, Wd = x.shape
        O, C2, kh, kw = W.shape
        xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        cols = np.empty((N, C * kh * kw, H * Wd), dtype=np.float32)
        idx = 0
        for c in range(C):
            for i in range(kh):
                for j in range(kw):
                    cols[:, idx, :] = xp[:, c, i:i + H, j:j + Wd].reshape(N, -1)
                    idx += 1
        out = np.einsum('ok,nkp->nop', W.reshape(O, -1), cols, optimize=True)
        out += b.reshape(1, O, 1)
        return out.reshape(N, O, H, Wd)

    def maxpool2(x):
        N, C, H, Wd = x.shape
        H2, W2 = H // 2, Wd // 2
        v = x[:, :, :H2 * 2, :W2 * 2].reshape(N, C, H2, 2, W2, 2)
        return v.max(axis=(3, 5))

    def features_from(x, cw1, cb1, cw2, cb2, chunk=250):
        outs = []
        for s in range(0, x.shape[0], chunk):
            xb = x[s:s + chunk]
            h = np.maximum(conv2d(xb, cw1, cb1), 0.0)
            h = maxpool2(h)
            h = np.maximum(conv2d(h, cw2, cb2), 0.0)
            h = maxpool2(h)
            outs.append(h.reshape(h.shape[0], -1))
        return np.concatenate(outs, 0).astype(np.float32)

    def ridge_fit(F, Y, lam):
        # closed-form ridge with intercept (centered)
        N = F.shape[0]
        Fm = F.mean(0, keepdims=True)
        Ym = Y.mean(0, keepdims=True)
        Fc = F - Fm
        Yc = Y - Ym
        G = Fc.T @ Fc
        G[np.diag_indices_from(G)] += lam * N
        A = np.linalg.solve(G, Fc.T @ Yc)
        b = Ym - Fm @ A
        return A, b

    def acc(F, Y, A, b):
        return float((np.argmax(F @ A + b, 1) == Y.argmax(1)).mean())

    def loo_select(F, Y, lams):
        # closed-form LOO for ridge via SVD of centered design
        N, D = F.shape
        Fm = F.mean(0, keepdims=True)
        Fc = F - Fm
        Ym = Y.mean(0, keepdims=True)
        Yc = Y - Ym
        U, s, Vt = np.linalg.svd(Fc, full_matrices=False)
        UtY = U.T @ Yc
        best = (None, -1.0, None)
        for lam in lams:
            d = s ** 2
            inv = 1.0 / (d + lam * N + 1e-12)
            A = Vt.T @ (inv[:, None] * (UtY))
            b = Ym - Fm @ A
            h = (d * inv).sum()
            resid = Yc - Fc @ A
            # LOO residual for each output column: e_i/(1-h_ii)
            hh = (U ** 2) @ inv
            denom = np.maximum(1.0 - hh, 1e-6)
            loo = (resid / denom[:, None]) ** 2
            score = float(loo.mean())
            if score > best[1]:
                best = (lam, score, (A, b))
        return best

    # ---------------- data ----------------------------------------------------
    x = np.asarray(f["x"], dtype=np.float32)
    y = np.asarray(f["y"], dtype=np.int64)
    N = x.shape[0]
    Y = np.zeros((N, 10), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # ---------------- design conv filters ------------------------------------
    # conv1: 16 filters of 3x3 on the raw image.
    #  8 DoG/edge-like (center-surround + oriented first derivatives)
    #  8 Gabor-like (oriented sine gratings at 2 scales)
    c1 = np.zeros((16, 1, 3, 3), dtype=np.float32)
    idx = 0
    # center-surround (DoG) at 2 scales
    for s in (1.0, 2.0):
        k = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], np.float32)
        c1[idx, 0] = (k / k.sum()) - (np.ones((3, 3), np.float32) / 9.0) * (1.0 / s)
        idx += 1
    # oriented first derivatives (Sobel-like) at 4 orientations
    sob = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], np.float32)
    for _ in range(4):
        c1[idx, 0] = np.rot90(sob, _).copy()
        idx += 1
    # Gabor-like oriented gratings at 2 scales x 2 orientations
    yy, xx = np.mgrid[-1:2, -1:2].astype(np.float32)
    for sc in (1.0, 2.0):
        for th in (0.0, np.pi / 4):
            g = np.cos(2 * np.pi * sc * (xx * np.cos(th) + yy * np.sin(th)) / 4.0)
            g = g - g.mean()
            c1[idx, 0] = g
            idx += 1
    # remaining 4: isotropic high-pass + smoothed variants
    hp = np.array([[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]], np.float32)
    c1[idx, 0] = hp; idx += 1
    c1[idx, 0] = np.array([[1.0, 1.0, 1.0], [1.0, -8.0, 1.0], [1.0, 1.0, 1.0]], np.float32); idx += 1
    c1[idx, 0] = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], np.float32); idx += 1
    c1[idx, 0] = np.array([[1.0, 0.0, 1.0], [0.0, -4.0, 0.0], [1.0, 0.0, 1.0]], np.float32); idx += 1
    c1 /= (np.abs(c1).max(axis=(1, 2, 3), keepdims=True) + 1e-8)
    b1 = np.zeros(16, dtype=np.float32)

    # conv2: 32 filters over the 16 conv1 channels (3x3), scale-balanced so that
    # every channel has comparable response magnitude (the old bank had a few
    # dominant channels and many near-dead ones).
    rng = np.random.default_rng(12345)
    c2 = rng.normal(0.0, 1.0, size=(32, 16, 3, 3)).astype(np.float32)
    # make each conv2 filter a random combination of all 16 conv1 channels
    for o in range(32):
        mix = rng.normal(0.0, 1.0, size=(16, 3, 3)).astype(np.float32)
        c2[o] = mix
    # balance: unit-norm per filter, then equalize expected response energy
    nrm = np.sqrt((c2 ** 2).sum(axis=(1, 2, 3), keepdims=True)) + 1e-8
    c2 /= nrm
    b2 = np.zeros(32, dtype=np.float32)

    # ---------------- candidate evaluation (closed-form fits only) -----------
    # split pool: fit on 70%, validate on 30% for model selection
    idxs = np.arange(N)
    rng2 = np.random.default_rng(7)
    rng2.shuffle(idxs)
    ntr = int(0.7 * N)
    tr, va = idxs[:ntr], idxs[ntr:]

    cands = []
    # A: keep original conv stack (features from feedback), refit head
    cands.append(("orig", None, None, None, None))
    # B: designed conv1 + scale-balanced random conv2
    cands.append(("design", c1, b1, c2, b2))
    # C: designed conv1 + original conv2 (keeps the learned-ish mixing)
    cands.append(("design_origmix", c1, b1,
                  np.asarray(w["conv2.weight"], np.float32),
                  np.asarray(w["conv2.bias"], np.float32)))

    lam_grid = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

    best = None
    for name, W1, B1, W2, B2 in cands:
        if name == "orig":
            F = np.asarray(f["features"], dtype=np.float32)
        else:
            F = features_from(x, W1, B1, W2, B2)
        Fd = F.astype(np.float64)
        # ridge fit on train split for each lambda -> validation accuracy
        bestva = -1.0
        bestlam = lam_grid[0]
        for lam in lam_grid:
            A, b = ridge_fit(Fd[tr], Y[tr], lam)
            a = acc(Fd[va], Y[va], A, b)
            if a > bestva:
                bestva, bestlam, bestA, bestb = a, lam, A, b
        # also LDA-style (class-mean whitened) head on the same features
        mu = np.stack([Fd[tr][Y[tr].argmax(1) == c].mean(0) for c in range(10)])
        Fc = Fd[tr] - Fd[tr].mean(0, keepdims=True)
        S = Fc.T @ Fc / max(len(tr) - 1, 1)
        dg = np.diag(S).copy()
        dg[dg < 1e-10] = 1e-10
        for gam in (0.0, 0.05, 0.2, 0.5):
            Sg = (1.0 - gam) * S + gam * np.diag(dg)
            Sg[np.diag_indices_from(Sg)] += 1e-3 * np.trace(Sg) / Sg.shape[0]
            try:
                Wi = np.linalg.solve(Sg, mu.T).T
            except np.linalg.LinAlgError:
                continue
            bi = -0.5 * np.einsum('cd,de,ce->c', mu, Wi, mu)
            a = acc(Fd[va], Y[va], Wi, bi)
            if a > bestva:
                bestva, bestlam = a, ("lda", gam)
                bestA, bestb = Wi, bi
        if best is None or bestva > best[0]:
            best = (bestva, name, bestlam, bestA, bestb, (W1, B1, W2, B2))

    bestva, name, lam, A, b, (W1, B1, W2, B2) = best

    # ---------------- final fit on the whole pool ----------------------------
    if name == "orig":
        F = np.asarray(f["features"], dtype=np.float32).astype(np.float64)
    else:
        F = features_from(x, W1, B1, W2, B2).astype(np.float64)

    if isinstance(lam, tuple) and lam[0] == "lda":
        mu = np.stack([F[Y.argmax(1) == c].mean(0) for c in range(10)])
        Fc = F - F.mean(0, keepdims=True)
        S = Fc.T @ Fc / max(N - 1, 1)
        dg = np.diag(S).copy(); dg[dg < 1e-10] = 1e-10
        gam = lam[1]
        Sg = (1.0 - gam) * S + gam * np.diag(dg)
        Sg[np.diag_indices_from(Sg)] += 1e-3 * np.trace(Sg) / Sg.shape[0]
        A = np.linalg.solve(Sg, mu.T).T
        b = -0.5 * np.einsum('cd,de,ce->c', mu, A, mu)
    else:
        A, b = ridge_fit(F, Y, lam)

    train_acc = acc(F, Y, A, b)

    # ---------------- write back ---------------------------------------------
    out = dict(w)
    if name != "orig":
        out["conv1.weight"] = np.asarray(W1, np.float32)
        out["conv1.bias"] = np.asarray(B1, np.float32)
        out["conv2.weight"] = np.asarray(W2, np.float32)
        out["conv2.bias"] = np.asarray(B2, np.float32)
    out["head.weight"] = A.astype(np.float32)
    out["head.bias"] = b.astype(np.float32)

    # safety: finite, bounded, correct shapes
    for k in ("conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias",
              "head.weight", "head.bias"):
        t = np.asarray(out[k], np.float32)
        t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        t = np.clip(t, -1000.0, 1000.0)
        out[k] = t.astype(np.float32)

    # per-class accuracy of the final head
    pred = np.argmax(F @ A + b, 1)
    per_class = [float((pred[y == c] == c).mean()) for c in range(10)]

    diagnostics = {
        "design": name,
        "lambda": (lam if not isinstance(lam, tuple) else str(lam)),
        "val_acc_selected": float(bestva),
        "train_acc_final": float(train_acc),
        "per_class_acc": per_class,
        "head_w_absmax": float(np.abs(out["head.weight"]).max()),
        "head_w_std": float(out["head.weight"].std()),
        "feat_mean": float(F.mean()),
        "feat_std": float(F.std()),
        "feat_zero_frac": float((F == 0).mean()),
        "conv1_absmax": float(np.abs(out["conv1.weight"]).max()),
        "conv2_absmax": float(np.abs(out["conv2.weight"]).max()),
        "note": "conv stack redesigned (DoG/Sobel/Gabor conv1 + scale-balanced conv2) "
                "and head refit by closed-form ridge/LDA; features recomputed for all 2000 images"
    }
    return {"weights": out, "diagnostics": diagnostics}
