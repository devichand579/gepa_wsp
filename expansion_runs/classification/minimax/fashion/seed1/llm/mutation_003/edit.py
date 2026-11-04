def edit(w, f):
    import numpy as np

    # ------------------------------------------------------------------ utils
    def conv2d(x, W, b):
        """cross-correlation, padding=1, stride 1. x:[N,C,H,W] W:[O,C,3,3]"""
        N, C, H, Wd = x.shape
        O = W.shape[0]
        xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
        s_h, s_w = xp.strides
        win = np.lib.stride_tricks.as_strided(
            xp, shape=(N, C, H, Wd, 3, 3),
            strides=(s_h * C, s_h, s_w, xp.strides[3], s_w, xp.strides[3]),
            writeable=False)
        cols = win.reshape(N * C * H * Wd, 9)
        wm = W.reshape(O, C * 9).T
        out = cols @ wm
        out = out.reshape(N, H, Wd, O).transpose(0, 3, 1, 2)
        return out + b[None, :, None, None]

    def maxpool2(x):
        N, C, H, Wd = x.shape
        v = x.reshape(N, C, H // 2, 2, Wd // 2, 2).transpose(0, 1, 2, 4, 3, 5)
        return v.reshape(N, C, H // 2, Wd // 2, 4).max(axis=-1)

    def forward_feats(x, c1w, c1b, c2w, c2b):
        h = maxpool2(np.maximum(conv2d(x, c1w, c1b), 0.0))
        h = maxpool2(np.maximum(conv2d(h, c2w, c2b), 0.0))
        return h.reshape(h.shape[0], -1)

    # ------------------------------------------------- analytic conv1 filters
    def oriented_deriv(theta):
        """3x3 first-derivative-of-Gaussian-like kernel at angle theta (radians)."""
        r = np.arange(-1, 2)
        dr, dc = np.meshgrid(r, r, indexing="ij")
        ct, st = np.cos(theta), np.sin(theta)
        u = dr * ct + dc * st
        v = -dr * st + dc * ct
        k = -u * np.exp(-0.5 * (u * u + v * v))
        return k / (np.abs(k).max() + 1e-12)

    def oriented_second(theta):
        r = np.arange(-1, 2)
        dr, dc = np.meshgrid(r, indexing="ij")
        ct, st = np.cos(theta), np.sin(theta)
        u = dr * ct + dc * st
        v = -dr * st + dc * ct
        k = (u * u - v * v) * np.exp(-0.5 * (u * u + v * v))
        return k / (np.abs(k).max() + 1e-12)

    c1 = []
    for k in range(8):                       # 8 oriented first derivatives
        c1.append(oriented_deriv(k * np.pi / 8.0))
    for k in range(4):                       # 4 oriented second derivatives
        c1.append(oriented_second(k * np.pi / 4.0))
    c1.append(np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]))   # laplacian
    c1.append(np.array([[-1.0, -1.0, -1.0], [-1.0, 9.0, -1.0], [-1.0, -1.0, -1.0]]) / 9.0)
    c1.append(np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0)  # blur
    c1.append(np.array([[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]) / 3.0)  # x edge
    c1.append(np.array([[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]) / 3.0)  # y edge
    c1.append(np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]]) / 3.0)  # diag edge
    c1.append(np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]))          # identity
    c1w = np.stack(c1).astype(np.float32)[:, None, :, :]      # [16,1,3,3]
    c1b = np.zeros(16, dtype=np.float32)

    # ------------------------------------------------- conv2: fixed mixing bank
    rng = np.random.default_rng(20240917)
    c2w = np.zeros((32, 16, 3, 3), dtype=np.float32)
    # 16 "identity + local smoothing" filters (one per conv1 channel)
    for i in range(16):
        c2w[i, i, 1, 1] = 1.0
        c2w[i, i, 1, 0] = 0.25
        c2w[i, i, 1, 2] = 0.25
        c2w[i, i, 0, 1] = 0.25
        c2w[i, i, 2, 1] = 0.25
    # 16 seeded random mixtures of all conv1 channels (texture/energy detectors)
    mix = rng.normal(0.0, 1.0, size=(16, 16, 3, 3)).astype(np.float32)
    mix /= np.sqrt((mix ** 2).sum(axis=(1, 2, 3), keepdims=True) + 1e-12)
    c2w[16:] = mix
    c2b = np.zeros(32, dtype=np.float32)

    # ------------------------------------------------------ forward + features
    x = np.ascontiguousarray(f["x"], dtype=np.float32)
    y = np.asarray(f["y"]).astype(np.int64)
    N = x.shape[0]
    X = forward_feats(x, c1w, c1b, c2w, c2b).astype(np.float64)   # [N,1568]

    # --------------------------------------------- closed-form ridge head fit
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Z = (X - mu) / sd
    Y = np.zeros((N, 10), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    G = Z.T @ Z
    b = Z.T @ Y
    dg = np.diag(G).copy()
    tr = float(dg.sum())
    p = G.shape[0]

    # closed-form GCV selection of the ridge level (vectorised over a fixed grid)
    alphas = tr * np.logspace(-4.0, 1.0, 21)
    best = (None, -1.0)
    for a in alphas:                       # closed-form solutions, no training loop
        Ai = G + a * np.eye(p)
        try:
            Wi = np.linalg.solve(Ai, b)
        except np.linalg.LinAlgError:
            continue
        r = Y - Z @ Wi
        rss = float((r ** 2).sum())
        df = float((dg / (dg + a)).sum())
        den = max(1.0 - df / N, 1e-6)
        gcv = (rss / N) / (den * den)
        if gcv < best[1]:
            best = (a, gcv)
    alpha = alphas[0] if best[0] is None else float(best[0])

    A = G + alpha * np.eye(p)
    Wz = np.linalg.solve(A, b)
    bz = Y.mean(axis=0) - Z.mean(axis=0) @ Wz

    # back to raw-feature space
    Wraw = Wz / sd[:, None]
    braw = bz - Wraw.T @ mu

    # --------------------------------------------------- diagnostics (in-sample)
    logits = X @ Wraw + braw
    pred = logits.argmax(axis=1)
    acc = float((pred == y).mean())
    pmax = np.max(logits, axis=1)
    lse = np.log(np.exp(logits - pmax[:, None]).sum(axis=1)) + pmax
    ce = float((-lse[np.arange(N), y]).mean())

    # 5-fold CV of the same closed-form procedure (fold assignment is deterministic)
    fold = np.arange(N) % 5
    cv_correct = 0
    for k in range(5):
        tr = fold != k
        te = ~tr
        Xtr = X[tr]; Ytr = Y[tr]
        m = Xtr.mean(axis=0); s = Xtr.std(axis=0); s[s < 1e-8] = 1.0
        Ztr = (Xtr - m) / s
        Gtr = Ztr.T @ Ztr
        btr = Ztr.T @ Ytr
        dtr = np.diag(Gtr).copy()
        atr = float(dtr.sum()) * 1e-3
        Wtr = np.linalg.solve(Gtr + atr * np.eye(p), btr)
        btr2 = Ytr.mean(axis=0) - Ztr.mean(axis=0) @ Wtr
        Zte = (X[te] - m) / s
        lg = Zte @ Wtr + btr2
        cv_correct += int((lg.argmax(axis=1) == y[te]).sum())
    cv_acc = cv_correct / float(N)

    # ------------------------------------- verify forward pass consistency (batch)
    bidx = np.asarray(f["batch.ids"]).astype(np.int64)
    bx = np.ascontiguousarray(f["batch.x"], dtype=np.float32)
    c1o = conv2d(bx, c1w, c1b)
    c1p = maxpool2(np.maximum(c1o, 0.0))
    c2o = conv2d(c1p, c2w, c2b)
    c2p = maxpool2(np.maximum(c2o, 0.0))
    bf = c2p.reshape(c2p.shape[0], -1)
    blog = bf @ Wraw + braw
    ref_c1 = np.asarray(f["layer.conv1.output"], dtype=np.float32)
    ref_c2 = np.asarray(f["layer.conv2.output"], dtype=np.float32)
    ref_h = np.asarray(f["layer.head.output"], dtype=np.float32)
    err_c1 = float(np.abs(c1o - ref_c1).max())
    err_c2 = float(np.abs(c2o - ref_c2).max())
    err_h = float(np.abs(blog - ref_h).max())

    # ------------------------------------------------------------- write weights
    out = {k: np.array(v, dtype=np.float32, copy=True) for k, v in w.items()}
    out["conv1.weight"] = c1w.astype(np.float32)
    out["conv1.bias"] = c1b.astype(np.float32)
    out["conv2.weight"] = c2w.astype(np.float32)
    out["conv2.bias"] = c2b.astype(np.float32)
    out["head.weight"] = Wraw.T.astype(np.float32)
    out["head.bias"] = braw.astype(np.float32)

    for k in out:
        out[k] = np.clip(np.nan_to_num(out[k], nan=0.0, posinf=0.0, neginf=0.0), -1000.0, 1000.0).astype(np.float32)

    diag = {
        "train_acc": round(acc, 4),
        "train_ce": round(ce, 4),
        "cv5_acc": round(cv_acc, 4),
        "ridge_alpha": float(alpha),
        "ridge_alpha_over_trace": float(alpha / max(tr, 1e-12)),
        "gcv": float(best[1]) if best[0] is not None else None,
        "feat_dim": int(p),
        "n_train": int(N),
        "fwd_err_conv1_batch": round(err_c1, 5),
        "fwd_err_conv2_batch": round(err_c2, 5),
        "fwd_err_head_batch": round(err_h, 5),
        "batch_acc_new": float((blog.argmax(axis=1) == np.asarray(f["batch.y"])).mean()),
        "head_w_absmean": round(float(np.abs(out["head.weight"]).mean()), 6),
        "head_w_absmax": round(float(np.abs(out["head.weight"]).max()), 6),
        "conv1_absmax": round(float(np.abs(out["conv1.weight"]).max()), 6),
        "conv2_absmax": round(float(np.abs(out["conv2.weight"]).max()), 6),
        "feat_mean": round(float(mu.mean()), 5),
        "feat_std_mean": round(float(sd.mean()), 5),
        "feat_nearconst": int((sd < 1e-8).sum()),
        "note": "conv1=oriented derivative/edge/centre-surround bank; conv2=16 identity-smooth + 16 seeded random mixtures; head=closed-form ridge on one-hot with GCV-selected alpha (no gradient/iterative training)"
    }
    return {"weights": out, "diagnostics": diag}
