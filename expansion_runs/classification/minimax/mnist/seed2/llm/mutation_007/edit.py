def edit(w, f):
    import numpy as np

    # ------------------------------------------------------------------ data
    X = np.asarray(f["x"], dtype=np.float64).reshape(-1, 784)
    y = np.asarray(f["y"], dtype=np.int64)
    N, D = X.shape
    K = 10
    Y1 = np.zeros((N, K))
    Y1[np.arange(N), y] = 1.0

    rng = np.random.default_rng(12345)

    # ------------------------------------------------- PCA (centered, full)
    mu = X.mean(axis=0)
    Xc = X - mu
    # economical eigendecomposition of the covariance
    C = (Xc.T @ Xc) / max(N - 1, 1)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 0.0)
    evecs = evecs[:, order]
    d_pca = 64
    V = evecs[:, :d_pca]                      # principal directions
    Z = Xc @ V                                # N x d_pca  (scores)
    Zmu = Z.mean(axis=0)
    Zc = Z - Zmu

    # ------------------------------------------- per-class shrunk covariances
    cls_idx = [np.where(y == c)[0] for c in range(K)]
    # pooled within-class covariance (shared part)
    Sw = np.zeros((d_pca, d_pca))
    mus = np.zeros((K, d_pca))
    for c in range(K):
        idx = cls_idx[c]
        mus[c] = Z[idx].mean(axis=0)
        Dc = Z[idx] - mus[c]
        Sw += Dc.T @ Dc
    Sw /= max(N - K, 1)
    tr = np.trace(Sw) / d_pca
    eye = np.eye(d_pca)

    # shrinkage level for the class-specific part (fixed, closed form choice)
    lam = 0.35
    Sig = np.zeros((K, d_pca, d_pca))
    for c in range(K):
        idx = cls_idx[c]
        Dc = Z[idx] - mus[c]
        Sc = (Dc.T @ Dc) / max(len(idx) - 1, 1)
        S = (1.0 - lam) * Sc + lam * Sw + 1e-3 * tr * eye
        Sig[c] = S

    # ------------------------------------------------ Mahalanobis distances
    # d_c(x) = (z-mu_c)^T Sig_c^{-1} (z-mu_c)
    dist = np.zeros((N, K))
    for c in range(K):
        Si = np.linalg.inv(Sig[c])
        Dc = Z - mus[c]
        dist[:, c] = np.einsum("ij,jk,ik->i", Dc, Si, Dc)

    # choose offsets a_c so that ReLU(a_c - d_c) is (almost) always active
    a = dist.max(axis=0) + 1.0
    # safety margin scaled by spread so held-out outliers rarely truncate
    spread = dist.std(axis=0) + 1e-9
    a = a + 6.0 * spread

    H1 = np.maximum(a[None, :] - dist, 0.0)          # N x K   (base units)
    H2 = np.maximum(dist - a[None, :], 0.0)          # N x K   (excess units)

    # ------------------------------------------- random ReLU units (whitened)
    # whitened PCA space: scale each principal coordinate by 1/sqrt(eig)
    sc = 1.0 / np.sqrt(np.maximum(evals[:d_pca], 1e-8))
    Zw = Zc * sc[None, :]
    s = Zw.std(axis=0)
    s[s < 1e-9] = 1.0
    Zw = Zw / s[None, :]

    n_rand = 108
    W_rand = rng.normal(0.0, 1.0, size=(n_rand, d_pca)) / np.sqrt(d_pca)
    b_rand = rng.uniform(0.0, 1.0, size=n_rand) * 1.0
    R = np.maximum(Zw @ W_rand.T + b_rand[None, :], 0.0)   # N x n_rand

    # ------------------------------------------------------------- assemble
    H = np.concatenate([H1, H2, R], axis=1)              # N x 128
    assert H.shape[1] == 128

    # ------------------------------------------- closed-form ridge head + LOO
    Hm = H.mean(axis=0)
    Hc = H - Hm
    Yc = Y1 - Y1.mean(axis=0)
    G = Hc.T @ Hc
    B = Hc.T @ Yc
    # closed-form LOO error for ridge: e_i/(1-h_ii)
    evals_g, evecs_g = np.linalg.eigh(G)
    evals_g = np.maximum(evals_g, 0.0)

    def fit_ridge(lam):
        Gi = evecs_g @ np.diag(1.0 / (evals_g + lam)) @ evecs_g.T
        W = Gi @ B
        # intercept: mean target - Hm @ W
        b = Y1.mean(axis=0) - Hm @ W
        return W, b, Gi

    def loo_mse(lam):
        W, b, Gi = fit_ridge(lam)
        # H_ii = Hm_i^2 + G_ii
        hdiag = (Hm ** 2) + np.diag(G)
        Rres = Yc - Hc @ W
        # inverse of (I - H G^{-1}) applied to residuals
        # (I - Hc Gi Hc^T) ; use identity: (I - Hc Gi Hc^T)^{-1} = I + Hc Gi Hc^T/(1-h_ii)
        # compute per-sample leverage via  Hc Gi Hc^T  diagonal
        # efficient: diag = sum((Hc @ Gi) * Hc, axis=1)
        diag_h = np.einsum("ij,ij->i", Hc @ Gi, Hc)
        inv_den = 1.0 - diag_h
        # LOO residual = (Rres - h_ii * Rres_i) / (1 - h_ii)  (standard PRESS)
        loo = (Rres - diag_h[:, None] * Rres) / inv_den[:, None]
        return float(np.mean(loo ** 2)), W, b

    lams = np.array([0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0])
    best = None
    for lam in lams:
        m, W, b = loo_mse(lam)
        if best is None or m < best[0]:
            best = (m, lam, W, b)
    loo_best, lam_best, W2, b2 = best

    # ------------------------------------------------- logits & calibration
    def logits_of(W, b):
        return H @ W + b

    Zl = logits_of(W2, b2)

    def softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def ce_of(z):
        p = softmax(z)
        return float(-np.mean(np.log(np.maximum(p[np.arange(N), y], 1e-12))))

    def acc_of(z):
        return float(np.mean(np.argmax(z, axis=1) == y))

    # single global scale (temperature) chosen from a closed-form candidate set
    scales = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 14.0, 20.0, 30.0, 50.0])
    ces = np.array([ce_of(Zl * s) for s in scales])
    accs = np.array([acc_of(Zl * s) for s in scales])
    # prefer the smallest scale among those within 0.5% of the best CE
    ce_min = ces.min()
    ok = np.where(ces <= ce_min * 1.005)[0]
    j = ok[0]
    scale = float(scales[j])

    # ---------------------------------------------------------------- fc1
    # fc1.weight is [out=128, in=784]; hidden unit j computes w_j . x + b_j
    W1 = np.zeros((128, 784))
    b1 = np.zeros(128)
    # base Mahalanobis units: h_c = a_c - (z-mu_c)^T Sig_c^{-1} (z-mu_c)
    #   z = (x-mu) V  ->  d_c(x) = x^T A_c x + b_c^T x + c_c
    for c in range(K):
        Si = np.linalg.inv(Sig[c])
        A = V @ Si @ V.T                       # 784 x 784
        lin = -2.0 * (V @ Si @ mus[c])          # 784
        const = mus[c] @ Si @ mus[c]
        # h_c = a_c - d_c = -x^T A x - lin^T x + (a_c - const)
        W1[c] = -A[0] - lin
        b1[c] = a[c] - const + A[0] @ mu + lin @ mu
        # excess unit: h = d_c - a_c = x^T A x + lin^T x + (const - a_c)
        W1[K + c] = A[0] + lin
        b1[K + c] = const - a[c] - (A[0] @ mu + lin @ mu)
    # random ReLU units in whitened PCA space
    #   z_w = (x-mu) V diag(sc/s) ;  h = z_w . W_rand^T + b_rand
    Ww = V * (sc / s)[None, :]                # 784 x d_pca
    W1[2 * K:] = W_rand @ Ww.T
    b1[2 * K:] = b_rand - W_rand @ (Ww.T @ mu)

    # ---------------------------------------------------------------- fc2
    # head: logits = H @ W2 + b2, scaled by `scale`
    Wf2 = (scale * W2).astype(np.float32)     # [10,128]
    bf2 = (scale * b2).astype(np.float32)     # [10]

    # --------------------------------------------------------------- sanity
    def forward(W1_, b1_, W2_, b2_):
        h = np.maximum(X @ W1_.T + b1_[None, :], 0.0)
        return h @ W2_.T + b2_[None, :]

    z_new = forward(W1, b1, Wf2.astype(np.float64), bf2.astype(np.float64))
    acc_new = float(np.mean(np.argmax(z_new, axis=1) == y))
    ce_new = ce_of(z_new)

    # --------------------------------------------------------- write weights
    out = {}
    for k, v in w.items():
        out[k] = np.array(v, dtype=np.float32, copy=True)
    out["fc1.weight"] = np.clip(W1, -1000.0, 1000.0).astype(np.float32)
    out["fc1.bias"] = np.clip(b1, -1000.0, 1000.0).astype(np.float32)
    out["fc2.weight"] = np.clip(Wf2, -1000.0, 1000.0).astype(np.float32)
    out["fc2.bias"] = np.clip(bf2, -1000.0, 1000.0).astype(np.float32)

    diagnostics = {
        "method": "closed-form PCA + per-class Mahalanobis(QDA) units + random ReLU, ridge-LS head with LOO lambda, global logit scale",
        "n_samples": int(N),
        "pca_dim": int(d_pca),
        "n_maha_base": int(K),
        "n_maha_excess": int(K),
        "n_random_relu": int(n_rand),
        "shrinkage_lambda": float(lam),
        "ridge_lambda": float(lam_best),
        "loo_mse": float(loo_best),
        "scale": scale,
        "train_acc_new": acc_new,
        "train_ce_new": ce_new,
        "train_acc_old": float(np.mean(np.argmax(np.asarray(f["logits"], dtype=np.float64), axis=1) == y)),
        "train_ce_old": float(-np.mean(np.log(np.maximum(np.asarray(f["probs"], dtype=np.float64)[np.arange(N), y], 1e-12)))),
        "feature_zero_frac": float(np.mean(H == 0.0)),
        "feature_mean": float(H.mean()),
        "fc1_weight_absmax": float(np.abs(out["fc1.weight"]).max()),
        "fc1_weight_std": float(out["fc1.weight"].std()),
        "fc1_bias_absmax": float(np.abs(out["fc1.bias"]).max()),
        "fc2_weight_absmax": float(np.abs(out["fc2.weight"]).max()),
        "fc2_bias": [float(v) for v in out["fc2.bias"]],
        "logit_std_new": float(z_new.std()),
        "maha_a": [float(v) for v in a],
    }
    return {"weights": out, "diagnostics": diagnostics}
