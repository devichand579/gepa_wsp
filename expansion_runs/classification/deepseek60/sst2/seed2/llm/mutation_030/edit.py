import numpy as np

def edit(w, f):
    X = f["x"].astype(np.int64)          # [N,64] token ids
    y = f["y"].astype(np.int64)          # [N]
    N, T = X.shape
    V = w["embed.weight"].shape[0]       # 2000
    D = w["embed.weight"].shape[1]       # 32

    mask = (X != 0).astype(np.float32)   # [N,64]
    L = mask.sum(axis=1)                 # [N]
    L = np.maximum(L, 1.0)

    # length-normalized bag-of-words
    BOW = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), T)
    cols = X.reshape(-1)
    np.add.at(BOW, (rows, cols), 1.0)
    BOW /= L[:, None]
    BOW[:, 0] = 0.0                      # PAD never contributes

    t = np.where(y == 1, 1.0, -1.0)      # +/-1 targets

    # closed-form ridge on normalized BOW, lambda chosen by 5-fold CV
    def ridge_fit(Xtr, ttr, lam):
        A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
        b = Xtr.T @ ttr
        return np.linalg.solve(A, b)

    rng = np.random.default_rng(0)
    idx = rng.permutation(N)
    folds = np.array_split(idx, 5)
    best_lam, best_acc = 1.0, -1.0
    for lam in [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
        accs = []
        for k in range(5):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(5) if j != k])
            wk = ridge_fit(BOW[tr], t[tr], lam)
            pred = np.where(BOW[te] @ wk >= 0, 1, -1)
            accs.append((pred == t[te]).mean())
        a = float(np.mean(accs))
        if a > best_acc:
            best_acc, best_lam = a, lam

    w_lin = ridge_fit(BOW, t, best_lam)
    b_lin = float(np.mean(t - BOW @ w_lin))

    # ---- realize w_lin as d @ LayerNorm(embed[t]) ----
    # fixed zero-mean orthonormal frame u,v in R^D
    u = np.zeros(D); u[0] = 1.0; u[1] = -1.0; u /= np.linalg.norm(u)
    v = np.zeros(D); v[0] = 1.0; v[1] = 1.0; v[2] = -2.0; v /= np.linalg.norm(v)

    wmax = float(np.max(np.abs(w_lin)))
    if wmax < 1e-8:
        wmax = 1e-8
    R = wmax / (np.sqrt(D) * 0.99)       # scale so arccos arg stays < 1
    c = np.clip(w_lin / (np.sqrt(D) * R), -0.999, 0.999)
    theta = np.arccos(c)                 # [V]

    Z = np.sqrt(D) * (np.cos(theta)[:, None] * u[None, :]
                      + np.sin(theta)[:, None] * v[None, :])   # [V,32]
    Z[0, :] = 0.0                        # PAD row irrelevant (never pooled)

    # LayerNorm(embed[t]) == z_t  <=>  embed[t] = z_t (already zero-mean unit-var)
    w["embed.weight"] = Z.astype(np.float32)
    w["pos.weight"][:] = 0.0

    # d = R * u  (so d @ z_t = sqrt(D) R cos(theta_t) = w_lin[t])
    d = R * u
    w["head.weight"][0, :] = (-0.5 * d).astype(np.float32)
    w["head.weight"][1, :] = (0.5 * d).astype(np.float32)
    w["head.bias"][0] = np.float32(-0.5 * b_lin)
    w["head.bias"][1] = np.float32(0.5 * b_lin)

    # keep norm as identity LayerNorm, blocks zero (pure linear read-out)
    w["norm.weight"][:] = 1.0
    w["norm.bias"][:] = 0.0

    # verify realized scores match the fitted classifier
    Zf = Z.astype(np.float64)
    pooled = np.zeros((N, D))
    np.add.at(pooled, rows, Zf[cols])
    pooled /= L[:, None]
    score = pooled @ d + b_lin
    train_acc = float(((score >= 0).astype(np.int64) == y).mean())

    diag = {
        "best_lam": best_lam,
        "cv_acc_bow": best_acc,
        "train_acc_realized": train_acc,
        "w_lin_absmax": wmax,
        "R": float(R),
        "b_lin": b_lin,
    }
    return {"weights": w, "diagnostics": diag}
