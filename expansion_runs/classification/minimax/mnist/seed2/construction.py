import numpy as np

def edit(w, f):
    # ---------------- data ----------------
    X = np.asarray(f["x"], dtype=np.float64).reshape(-1, 784)
    y = np.asarray(f["y"], dtype=np.int64)
    N, D = X.shape
    C = 10

    # ---------------- class statistics ----------------
    mu = np.zeros((C, D))
    for c in range(C):
        m = y == c
        mu[c] = X[m].mean(axis=0)
    mubar = X.mean(axis=0)

    # within-class scatter
    Sw = np.zeros((D, D))
    for c in range(C):
        m = y == c
        Z = X[m] - mu[c]
        Sw += Z.T @ Z
    Sw /= N

    # regularized inverse (ridge on the scatter matrix)
    tr = np.trace(Sw) / D
    Sw_reg = Sw + (0.10 * tr) * np.eye(D)
    Sw_inv = np.linalg.inv(Sw_reg)

    # Fisher / LDA directions and class-mean template directions
    d = (mu - mubar) @ Sw_inv.T                      # C x D
    t = mu - mubar                                   # C x D

    def unit_norm(v):
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n < 1e-12] = 1.0
        return v / n

    d = unit_norm(d)
    t = unit_norm(t)

    # standardized scores for every training sample
    s_lda = X @ d.T                                  # N x C
    s_tpl = X @ t.T

    def zscale(v):
        mu_v = v.mean(axis=0)
        sd = v.std(axis=0)
        sd[sd < 1e-9] = 1.0
        return (v - mu_v) / sd, mu_v, sd

    z_lda, m_lda, sd_lda = zscale(s_lda)
    z_tpl, m_tpl, sd_tpl = zscale(s_tpl)

    # ---------------- hidden unit design ----------------
    rows, biases, tags = [], [], []

    # (a) multi-threshold soft binning of each class's LDA score
    for c in range(C):
        for tau in (0.0, 1.0, 2.0):
            rows.append(d[c])
            biases.append(tau * sd_lda[c] + m_lda[c])
            tags.append("lda")
    # (b) multi-threshold soft binning of each class's template score
    for c in range(C):
        for tau in (0.0, 1.0):
            rows.append(t[c])
            biases.append(tau * sd_tpl[c] + m_tpl[c])
            tags.append("tpl")
    # (c) random He-initialized pixel-space projections (generic nonlinearity)
    rng = np.random.default_rng(0)
    n_rand = 128 - len(rows)
    Wrand = rng.standard_normal((n_rand, D)) * np.sqrt(2.0 / D)
    brand = rng.standard_normal(n_rand) * 0.10
    for i in range(n_rand):
        rows.append(Wrand[i])
        biases.append(brand[i])
        tags.append("rand")

    W1 = np.asarray(rows, dtype=np.float64)          # 128 x 784
    b1 = np.asarray(biases, dtype=np.float64)         # 128

    # ---------------- recompute hidden features ----------------
    H = np.maximum(X @ W1.T + b1, 0.0)                # N x 128
    H = np.clip(H, 0.0, 1e6)

    # ---------------- ridge readout (closed form) ----------------
    Y1 = np.zeros((N, C))
    Y1[np.arange(N), y] = 1.0
    Hm = H.mean(axis=0)
    Hc = H - Hm
    G = Hc.T @ Hc
    b = Hc.T @ Y1
    trG = np.trace(G) / G.shape[0]

    def solve_ridge(lam):
        A = G + lam * np.eye(G.shape[0])
        W2 = np.linalg.solve(A, b)
        c2 = Y1.mean(axis=0) - Hm @ W2
        return W2, c2

    def loo_err(lam):
        W2, c2 = solve_ridge(lam)
        R = Hc @ W2 - (Y1 - Y1.mean(axis=0))
        A = G + lam * np.eye(G.shape[0])
        Ainv = np.linalg.inv(A)
        Hinv = np.einsum('ij,jk->ik', Hc, Ainv)
        hdiag = np.einsum('ij,ij->i', Hc, Hinv)
        denom = np.maximum(1.0 - hdiag, 1e-6)
        e = (R ** 2).sum(axis=1)
        loo = (e / denom ** 2).sum()
        return loo, W2, c2

    # closed-form LOO selection over a handful of ridge parameters
    cands = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
    best = None
    for lam in cands:
        loo, W2, c2 = loo_err(lam)
        if best is None or loo < best[0]:
            best = (loo, lam, W2, c2)
    loo_best, lam, W2, c2 = best

    # ---------------- exact end-to-end check (ReLU applied) ----------------
    def forward(Xq, W1q, b1q, W2q, c2q):
        Hq = np.maximum(Xq @ W1q.T + b1q, 0.0)
        return Hq @ W2q + c2q

    def acc(Xq, yq, W1q, b1q, W2q, c2q):
        lg = forward(Xq, W1q, b1q, W2q, c2q)
        return float((lg.argmax(axis=1) == yq).mean())

    acc_in = acc(X, y, W1, b1, W2, c2)

    # ---------------- write back ----------------
    w["fc1.weight"][...] = W1.astype(np.float32)
    w["fc1.bias"][...] = b1.astype(np.float32)
    w["fc2.weight"][...] = W2.astype(np.float32)
    w["fc2.bias"][...] = c2.astype(np.float32)

    # ---------------- diagnostics ----------------
    lg = forward(X, y * 0 + 0, W1, b1, W2, c2) if False else forward(X, W1, b1, W2, c2)
    p = np.exp(lg - lg.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    ce = float(-np.log(np.clip(p[np.arange(N), y], 1e-12, None)).mean())
    pred = lg.argmax(axis=1)
    per_class = {int(c): float((pred[y == c] == c).mean()) for c in range(C)}
    nz = float((H > 0).mean())
    hstd = float(H.std())
    w1_std = float(W1.std())
    w2_absmax = float(np.abs(W2).max())

    diag = {
        "hypothesis": ("fc1 was random noise; rebuilt 128 hidden units as class-discriminant "
                       "(LDA) + class-template gated detectors with data-derived thresholds, "
                       "plus random He projections; fc2 solved by closed-form ridge on the "
                       "recomputed ReLU features."),
        "lambda_ridge": float(lam),
        "loo_mse": float(loo_best),
        "acc_train_pool": acc_in,
        "cross_entropy_train": ce,
        "per_class_acc": per_class,
        "hidden_zero_frac": 1.0 - nz,
        "hidden_mean": float(H.mean()),
        "hidden_std": hstd,
        "fc1_w_std": w1_std,
        "fc1_w_absmax": float(np.abs(W1).max()),
        "fc1_bias_absmax": float(np.abs(b1).max()),
        "fc2_w_std": float(W2.std()),
        "fc2_w_absmax": w2_absmax,
        "fc2_bias": [float(v) for v in c2],
        "n_units_lda": int(sum(1 for g in tags if g == "lda")),
        "n_units_tpl": int(sum(1 for g in tags if g == "tpl")),
        "n_units_rand": int(sum(1 for g in tags if g == "rand")),
        "lda_train_acc": float(((X @ d.T).argmax(axis=1) == y).mean()),
        "tpl_train_acc": float(((X @ t.T).argmax(axis=1) == y).mean()),
    }
    return {"weights": w, "diagnostics": diag}
