import numpy as np

def edit(w, f):
    X = f["x"]                      # [N,64] int64
    y = f["y"].astype(np.float64)   # [N]
    N, L = X.shape
    V = w["embed.weight"].shape[0]  # 2000
    D = w["embed.weight"].shape[1]  # 32

    # ---- 1. bag of words (counts) ----
    BOW = np.zeros((N, V), dtype=np.float64)
    for i in range(N):
        ids = X[i]
        ids = ids[ids != 0]
        if ids.size:
            np.add.at(BOW[i], ids, 1.0)

    # normalised BOW (matches masked_mean pooling)
    lens = np.maximum((X != 0).sum(axis=1), 1).astype(np.float64)
    F = BOW / lens[:, None]          # [N, V]

    # ---- 2. ridge fit with CV-selected lambda ----
    t = 2.0 * y - 1.0                # {-1,+1}
    lam_grid = [0.03, 0.1, 0.3, 1.0, 3.0]

    def ridge_fit(Ftr, ttr, lam):
        mu = Ftr.mean(axis=0)
        Fc = Ftr - mu
        G = Fc.T @ Fc + lam * np.eye(V)
        s = np.linalg.solve(G, Fc.T @ ttr)
        c = ttr.mean() - mu @ s
        return s, c

    # 5-fold CV
    rng = np.random.RandomState(0)
    idx = rng.permutation(N)
    folds = np.array_split(idx, 5)
    best_lam, best_acc = lam_grid[0], -1.0
    for lam in lam_grid:
        accs = []
        for k in range(5):
            va = folds[k]
            tr = np.concatenate([folds[j] for j in range(5) if j != k])
            s, c = ridge_fit(F[tr], t[tr], lam)
            pred = np.sign(F[va] @ s + c)
            accs.append((pred == t[va]).mean())
        a = float(np.mean(accs))
        if a > best_acc:
            best_acc, best_lam = a, lam

    s, c = ridge_fit(F, t, best_lam)
    s = s.astype(np.float64)

    # ---- 3. build embed rows: zero-mean / unit-variance, col0 = s ----
    E = np.zeros((V, D), dtype=np.float64)
    E[:, 0] = s
    # fill remaining columns with small deterministic orthogonal-ish noise
    rs = np.random.RandomState(12345)
    for v in range(V):
        row = rs.randn(D).astype(np.float64)
        row[0] = s[v]
        row -= row.mean()
        sd = row.std()
        if sd < 1e-8:
            row = np.zeros(D); row[0] = s[v]
            row -= row.mean(); sd = row.std()
        row /= sd
        E[v] = row
    # re-orthogonalise col0 exactly to s after centering (keep LN identity)
    E[:, 0] = s
    # enforce zero mean / unit var per row exactly
    for v in range(V):
        row = E[v]
        row -= row.mean()
        sd = row.std()
        if sd > 1e-8:
            row /= sd
        E[v] = row

    # ---- 4. head reads col0 (and a second view for scaling) ----
    H = np.zeros((2, D), dtype=np.float64)
    H[0, 0] = -1.0
    H[1, 0] = 1.0
    # small second view on col1 to reduce CE (keeps sign of logit)
    H[0, 1] = -0.05
    H[1, 1] = 0.05
    b = np.array([-c, c], dtype=np.float64)

    # ---- 5. write back ----
    w["embed.weight"] = E.astype(np.float32)
    w["head.weight"] = H.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

    # zero out all other modules (keep shapes)
    for k in list(w.keys()):
        if k in ("embed.weight", "head.weight", "head.bias"):
            continue
        arr = w[k]
        if k.endswith("norm1.weight") or k.endswith("norm2.weight") or k == "norm.weight":
            arr[...] = 1.0
        else:
            arr[...] = 0.0
        w[k] = arr.astype(np.float32)

    # ---- diagnostics ----
    logits = F @ s + c
    pred = (logits > 0).astype(np.int64)
    acc = float((pred == y).mean())
    # CE
    p = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    ce = float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))

    diag = {
        "method": "BOW ridge + LayerNorm-identity embedding",
        "best_lambda": best_lam,
        "cv_accuracy": best_acc,
        "train_accuracy": acc,
        "train_ce": ce,
        "s_norm": float(np.linalg.norm(s)),
        "intercept": float(c),
        "embed_row_mean_maxabs": float(np.abs(E.mean(axis=1)).max()),
        "embed_row_std_maxdev": float(np.abs(E.std(axis=1) - 1.0).max()),
    }
    return {"weights": w, "diagnostics": diag}
