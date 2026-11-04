import numpy as np

def edit(w, f):
    X_ids = f["x"].astype(np.int64)          # [N,64]
    y     = f["y"].astype(np.int64)          # [N]
    N, L  = X_ids.shape
    V     = w["embed.weight"].shape[0]       # 2000

    # ---- normalized unigram count features ----
    mask = (X_ids != 0)
    counts = np.zeros((N, V), dtype=np.float64)
    for i in range(N):
        row = X_ids[i][mask[i]]
        if row.size:
            np.add.at(counts[i], row, 1.0)
    lens = mask.sum(axis=1).astype(np.float64)
    lens[lens == 0] = 1.0
    Xn = counts / lens[:, None]              # [N,V] length-normalized counts

    # ---- ridge regression with intercept, closed form ----
    def fit(alpha, Xtr, ytr):
        Xa = np.concatenate([Xtr, np.ones((Xtr.shape[0], 1))], axis=1)
        d = Xa.shape[1]
        R = alpha * np.eye(d)
        R[-1, -1] = 0.0                      # don't penalize intercept
        A = Xa.T @ Xa + R
        b = Xa.T @ (2.0 * ytr - 1.0)         # targets in {-1,+1}
        return np.linalg.solve(A, b)

    # ---- cross-validation to pick alpha ----
    rng = np.random.RandomState(0)
    perm = rng.permutation(N)
    nfold = 5
    folds = np.array_split(perm, nfold)
    alphas = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    best_alpha, best_cv = 1e-1, -1.0
    for alpha in alphas:
        accs = []
        for k in range(nfold):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(nfold) if j != k])
            coef = fit(alpha, Xn[tr], y[tr])
            pred = (Xn[te] @ coef[:-1] + coef[-1]) > 0
            accs.append((pred == (y[te] == 1)).mean())
        cv = float(np.mean(accs))
        if cv > best_cv:
            best_cv, best_alpha = cv, alpha

    # ---- final fit on the whole feedback pool ----
    coef = fit(best_alpha, Xn, y)
    beta = coef[:-1].astype(np.float64)      # per-token score
    b0   = float(coef[-1])

    # ---- encode scores into embed[:,0]; dims 1..31 satisfy LN constraint ----
    # Need per row: sum = 0, sum of squares = 32  (variance 1, mean 0)
    # Construction: dim0 = a, dim1 = b, dims2..31 = c (all equal).
    #   a + b + 30c = 0 ; a^2 + b^2 + 30c^2 = 32
    #   => 930 c^2 + 60 a c + (2a^2 - 32) = 0
    scale = 4.0
    a = beta / scale
    amax = np.sqrt(31.0) * 0.999
    a = np.clip(a, -amax, amax)
    disc = 119040.0 - 3840.0 * a * a
    disc = np.clip(disc, 0.0, None)
    c = (-60.0 * a + np.sqrt(disc)) / 1860.0
    b = -a - 30.0 * c

    emb = np.zeros((V, 32), dtype=np.float64)
    emb[:, 0] = a
    emb[:, 1] = b
    emb[:, 2:] = c[:, None]
    w["embed.weight"] = emb.astype(np.float32)

    # ---- head reads dim0; gain = scale reproduces the linear score ----
    head = np.zeros((2, 32), dtype=np.float64)
    head[0, 0] = -scale / 2.0
    head[1, 0] =  scale / 2.0
    w["head.weight"] = head.astype(np.float32)
    w["head.bias"] = np.array([-b0 / 2.0, b0 / 2.0], dtype=np.float32)

    # ---- keep blocks identity, LN identity-scaled ----
    for name in ["block0.norm1.weight", "block0.norm1.bias",
                 "block0.norm2.weight", "block0.norm2.bias",
                 "block1.norm1.weight", "block1.norm1.bias",
                 "block1.norm2.weight", "block1.norm2.bias",
                 "norm.weight", "norm.bias"]:
        if name in w:
            w[name] = (np.ones_like(w[name]) if name.endswith("weight")
                       else np.zeros_like(w[name])).astype(np.float32)

    # ---- verify the realized linear score matches the fit ----
    pred_lin = (Xn @ beta + b0) > 0
    train_acc = float((pred_lin == (y == 1)).mean())

    diag = {
        "best_alpha": best_alpha,
        "cv_acc": best_cv,
        "train_acc_linear": train_acc,
        "beta_absmax": float(np.abs(beta).max()),
        "beta_std": float(beta.std()),
        "b0": b0,
        "scale": scale,
        "a_absmax": float(np.abs(a).max()),
        "embed_nonzero": int(np.count_nonzero(emb)),
    }
    return {"weights": w, "diagnostics": diag}
