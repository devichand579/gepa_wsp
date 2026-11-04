def edit(w, f):
    # ---------------- numpy forward pass (matches the stated architecture) --------
    def conv2d(x, W, b, pad=1):
        # x: [N,C,H,W], W: [O,C,kh,kw] -> [N,O,H,W]
        N, C, H, Wd = x.shape
        O, C2, kh, kw = W.shape
        xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        Hout = H + 2 * pad - kh + 1
        Wout = Wd + 2 * pad - kw + 1
        cols = np.empty((N, C, kh, kw, Hout, Wout), dtype=np.float32)
        for i in range(kh):
            for j in range(kw):
                cols[:, :, i, j, :, :] = xp[:, :, i:i + Hout, j:j + Wout]
        out = np.einsum('nckhw,ock->nohw', cols, W, optimize=True)
        out += b.reshape(1, O, 1, 1)
        return out

    def maxpool2(x):
        N, C, H, Wd = x.shape
        H2, W2 = H // 2, Wd // 2
        v = x[:, :, :2 * H2, :2 * W2].reshape(N, C, H2, 2, W2, 2)
        return v.max(axis=(3, 5))

    def forward(x, c1w, c1b, c2w, c2b):
        h = maxpool2(np.maximum(conv2d(x, c1w, c1b), 0.0))
        h = maxpool2(np.maximum(conv2d(h, c2w, c2b), 0.0))
        return h.reshape(h.shape[0], -1)

    x = np.asarray(f["x"], dtype=np.float32)
    y = np.asarray(f["y"]).astype(np.int64)
    N = x.shape[0]
    xf = x.reshape(N, 1, 28, 28)

    # ---------------- conv1: PCA of 3x3 patches, used as +/- pairs ---------------
    # patch matrix: all 28*28 positions of all images
    patches = np.empty((N * 784, 9), dtype=np.float64)
    xp = np.pad(xf, ((0, 0), (0, 0), (1, 1), (1, 1)))
    for i in range(3):
        for j in range(3):
            patches[:, i * 3 + j] = xp[:, 0, i:i + 28, j:j + 28].reshape(-1)
    pm = patches.mean(axis=0)
    pc = patches - pm
    cov = (pc.T @ pc) / max(1, pc.shape[0] - 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]
    evals = np.maximum(evals[order], 1e-12)
    K = 8
    V = evecs[:, :K]                      # [9,8] orthonormal patch directions
    c1w = np.zeros((16, 1, 3, 3), dtype=np.float32)
    c1b = np.zeros(16, dtype=np.float32)
    for k in range(K):
        v = V[:, k]
        c1w[2 * k, 0] = v.reshape(3, 3)
        c1w[2 * k + 1, 0] = -v.reshape(3, 3)
        c1b[2 * k] = -float(pm @ v)
        c1b[2 * k + 1] = float(pm @ v)
    # scale so conv1 pre-activations have unit std (keeps relu balanced)
    s1 = np.sqrt((c1w.reshape(16, 9) ** 2).sum(axis=1) * (cov[0, 0] + cov[1, 1] + cov[2, 2] +
                                                          cov[3, 3] + cov[4, 4] + cov[5, 5] +
                                                          cov[6, 6] + cov[7, 7] + cov[8, 8]) / 9.0)
    s1 = np.where(s1 > 1e-8, s1, 1.0)
    c1w = (c1w.reshape(16, 9) / s1[:, None]).reshape(16, 1, 3, 3).astype(np.float32)
    c1b = (c1b / s1).astype(np.float32)

    # ---------------- conv2: whitened random mixing of the 16 conv1 channels -----
    rng = np.random.default_rng(12345)
    c2w = rng.standard_normal((32, 16, 3, 3)).astype(np.float32)
    c2w *= (2.0 / np.sqrt(16 * 9.0)).astype(np.float32)
    c2b = np.zeros(32, dtype=np.float32)

    # ---------------- recompute features for the whole permitted pool ------------
    feats = forward(xf, c1w, c1b, c2w, c2b).astype(np.float64)
    mu = feats.mean(axis=0)
    sd = feats.std(axis=0)
    sd_safe = np.where(sd > 1e-8, sd, 1.0)
    Z = (feats - mu) / sd_safe
    active = sd > 1e-8
    Z = Z[:, active]

    # ---------------- one-hot targets, centered ---------------------------------
    Y = np.zeros((N, 10), dtype=np.float64)
    Y[np.arange(N), y] = 1.0
    ybar = Y.mean(axis=0)
    Yc = Y - ybar

    # ---------------- ridge via single SVD; lambda by closed-form GCV -------------
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    UtY = U.T @ Yc
    d = s ** 2
    s2_sum = float((s ** 2).sum())
    p_eff = float(Z.shape[1])
    n_eff = float(N)

    def gcv(lam):
        df = float((d / (d + lam)).sum())
        resid = float(((s ** 2) / (d + lam) * (UtY ** 2).sum(axis=1)).sum())
        denom = max(1e-8, (1.0 - df / n_eff) ** 2)
        return resid / denom, df, resid

    lam_grid = np.concatenate([
        np.logspace(-6, 0, 61),
        np.logspace(0, 3, 61)
    ])
    lam_grid = np.unique(np.concatenate([lam_grid, [1e-8, 1e8]]))
    best = (np.inf, None, None, None)
    for lam in lam_grid:
        g, df, res = gcv(float(lam))
        if g < best[0]:
            best = (g, float(lam), df, res)
    gcv_best, lam_star, df_star, res_star = best

    # closed-form ridge coefficients on standardized features
    shrink = d / (d + lam_star)
    W_std = (Vt.T * shrink) @ (UtY / np.maximum(s, 1e-12)[:, None])  # [P,10]
    b_std = ybar - W_std.T @ (mu[active] / sd_safe[active])

    # ---------------- write back into head (un-standardize) --------------------
    head_w = np.zeros((10, feats.shape[1]), dtype=np.float64)
    head_w[:, active] = (W_std / sd_safe[active][None, :]).T
    head_b = b_std.astype(np.float64)

    # ---------------- diagnostics ------------------------------------------------
    logits = feats @ head_w.T + head_b
    pred = logits.argmax(axis=1)
    acc = float((pred == y).mean())
    mx = logits.max(axis=1, keepdims=True)
    lse = mx + np.log(np.exp(logits - mx).sum(axis=1, keepdims=True))
    ce = float((-lse[np.arange(N), y]).mean())
    per_class = []
    for c in range(10):
        m = (y == c)
        per_class.append(float((pred[m] == c).mean()) if m.any() else 0.0)
    # effective rank of the head (participation ratio of singular values)
    sg = np.linalg.svd(head_w, compute_uv=False)
    pr = float((sg.sum() ** 2) / max(1e-12, (sg ** 2).sum()))

    w["conv1.weight"] = c1w.astype(np.float32)
    w["conv1.bias"] = c1b.astype(np.float32)
    w["conv2.weight"] = c2w.astype(np.float32)
    w["conv2.bias"] = c2b.astype(np.float32)
    w["head.weight"] = head_w.astype(np.float32)
    w["head.bias"] = head_b.astype(np.float32)

    # clip for safety
    for k in w:
        a = np.asarray(w[k], dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=1000.0, neginf=-1000.0)
        w[k] = np.clip(a, -1000.0, 1000.0).astype(np.float32)

    diagnostics = {
        "hypothesis": "PCA patch filter bank (conv1, +/- pairs) + whitened random mixing (conv2) + ridge head on standardized features, lambda by closed-form GCV",
        "pool_accuracy": acc,
        "pool_cross_entropy": ce,
        "lambda_star": float(lam_star),
        "gcv_best": float(gcv_best),
        "effective_df": float(df_star),
        "n_features_active": int(active.sum()),
        "head_participation_ratio": pr,
        "per_class_recall": per_class,
        "feature_mean": float(feats.mean()),
        "feature_std": float(feats.std()),
        "logit_std": float(logits.std()),
        "conv1_filter_norms": [float(v) for v in np.sqrt((c1w.reshape(16, 9) ** 2).sum(axis=1))],
        "patch_pca_evals": [float(v) for v in evals[:8]],
        "head_weight_absmax": float(np.abs(head_w).max()),
        "head_bias": [float(v) for v in head_b],
    }
    return {"weights": w, "diagnostics": diagnostics}
