import numpy as np

def edit(w, f):
    X = f["x"].astype(np.int64)          # [4000,64]
    y = f["y"].astype(np.int64)          # [4000]
    N, L = X.shape
    V = w["embed.weight"].shape[0]       # 2000

    # ---- length-normalised bag-of-words counts (exclude PAD=0) ----
    mask = (X != 0).astype(np.float64)                 # [N,64]
    lens = mask.sum(1, keepdims=True)                  # [N,1]
    lens[lens == 0] = 1.0
    C = np.zeros((N, V), dtype=np.float64)
    for i in range(N):
        np.add.at(C[i], X[i][mask[i] > 0], 1.0)
    C /= lens                                          # mean count per token

    # ---- ridge-regularised linear classifier (closed form) ----
    # centre features, fit on centred targets, choose lambda by 5-fold CV
    Cm = C.mean(0, keepdims=True)
    Cc = C - Cm
    yc = y.astype(np.float64) - y.mean()
    G = Cc.T @ Cc                                      # [V,V]
    b = Cc.T @ yc                                      # [V]

    def fit(lam):
        return np.linalg.solve(G + lam * np.eye(V), b)

    # small fixed grid, evaluated with 5-fold CV on the pool
    folds = np.arange(N) % 5
    best_lam, best_acc = 1.0, -1.0
    for lam in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        acc = 0.0
        for k in range(5):
            tr = folds != k
            va = ~tr
            Gk = Cc[tr].T @ Cc[tr]
            bk = Cc[tr].T @ yc[tr]
            wk = np.linalg.solve(Gk + lam * np.eye(V), bk)
            pred = (Cc[va] @ wk) > 0
            acc += (pred == (y[va] == 1)).sum()
        acc /= N
        if acc > best_acc:
            best_acc, best_lam = acc, lam

    w_tok = fit(best_lam)                              # [V]
    # intercept on the centred scale
    intercept = y.mean() - (Cm @ w_tok)[0]

    # ---- embed the score so LayerNorm is the identity on each row ----
    # e(token) = [s, c, c, ..., c] with mean 0 and std 1  ->  c = -s/31
    s = w_tok.astype(np.float64)
    s = s / (np.abs(s).max() + 1e-12)                  # keep magnitudes tame
    c = -s / 31.0
    E = np.empty((V, 32), dtype=np.float64)
    E[:, 0] = s
    E[:, 1:] = c[:, None]
    # enforce exact mean 0 / std 1 (guards against rounding)
    E -= E.mean(1, keepdims=True)
    sd = E.std(1, keepdims=True)
    sd[sd < 1e-12] = 1.0
    E /= sd

    w["embed.weight"] = E.astype(np.float32)
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)

    # ---- head: read coordinate 0 of the pooled feature ----
    scale = 1.0
    head = np.zeros((2, 32), dtype=np.float32)
    head[1, 0] = scale
    head[0, 0] = 0.0
    w["head.weight"] = head
    w["head.bias"] = np.array([0.0, intercept * scale], dtype=np.float32)

    # ---- keep the blocks as exact identities ----
    for name in list(w.keys()):
        if name.endswith(".weight") and ("qkv" in name or "proj" in name or "ff1" in name or "ff2" in name):
            w[name] = np.zeros_like(w[name], dtype=np.float32)
        if name.endswith(".bias") and ("qkv" in name or "proj" in name or "ff1" in name or "ff2" in name):
            w[name] = np.zeros_like(w[name], dtype=np.float32)

    # ---- final LayerNorm identity ----
    w["norm.weight"] = np.ones_like(w["norm.weight"], dtype=np.float32)
    w["norm.bias"] = np.zeros_like(w["norm.bias"], dtype=np.float32)

    # ---- clip to the allowed range ----
    for k in w:
        w[k] = np.clip(np.nan_to_num(w[k], nan=0.0, posinf=1000.0, neginf=-1000.0),
                       -1000.0, 1000.0).astype(np.float32)

    diag = {
        "best_lambda": float(best_lam),
        "cv_acc": float(best_acc),
        "train_acc": float(((Cc @ w_tok) > 0).mean() == (y == 1).mean() and 0.0 or 0.0),
        "w_norm": float(np.linalg.norm(w_tok)),
        "w_absmax": float(np.abs(w_tok).max()),
        "intercept": float(intercept),
        "embed_row_mean_absmax": float(np.abs(w["embed.weight"].mean(1)).max()),
        "embed_row_std_min": float(w["embed.weight"].std(1).min()),
        "head_norm": float(np.linalg.norm(w["head.weight"])),
    }
    return {"weights": w, "diagnostics": diag}
