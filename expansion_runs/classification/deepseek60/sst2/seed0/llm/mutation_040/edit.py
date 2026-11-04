import numpy as np

def edit(w, f):
    X = np.asarray(f["x"]).astype(np.int64)          # [N,64]
    y = np.asarray(f["y"]).astype(np.float64)        # [N]
    N, T = X.shape
    V = w["embed.weight"].shape[0]                   # 2000

    # ---- build length-normalised bag-of-words features ----
    mask = (X != 0)                                  # exclude PAD
    L = np.maximum(mask.sum(axis=1), 1).astype(np.float64)
    M = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), T)
    cols = X.reshape(-1)
    keep = mask.reshape(-1)
    np.add.at(M, (rows[keep], cols[keep]), 1.0)
    M /= L[:, None]                                  # mean token counts

    # ---- ridge regression with CV over lambda (closed form) ----
    rng = np.random.default_rng(0)
    idx = rng.permutation(N)
    nfold = 5
    folds = np.array_split(idx, nfold)
    ybar = y.mean()
    mbar = M.mean(axis=0)
    Mc = M - mbar
    yc = y - ybar
    G = Mc.T @ Mc
    bvec = Mc.T @ yc
    best_lam, best_acc = None, -1.0
    for lam in [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]:
        A = G + lam * np.eye(V)
        try:
            wv = np.linalg.solve(A, bvec)
        except np.linalg.LinAlgError:
            wv = np.linalg.lstsq(A, bvec, rcond=None)[0]
        b = ybar - mbar @ wv
        # quick CV accuracy
        acc = 0.0
        for k in range(nfold):
            te = folds[k]
            sc = M[te] @ wv + b
            acc += ((sc > 0.5).astype(np.float64) == y[te]).mean()
        acc /= nfold
        if acc > best_acc:
            best_acc, best_lam = acc, lam
    A = G + best_lam * np.eye(V)
    try:
        wv = np.linalg.solve(A, bvec)
    except np.linalg.LinAlgError:
        wv = np.linalg.lstsq(A, bvec, rcond=None)[0]
    b = ybar - mbar @ wv

    # ---- construct embeddings so that delta . e_t = c_t ----
    R = np.sqrt(32.0)                                # fixed norm -> LayerNorm ~ identity
    c = wv.copy()                                    # per-token scores (c_0 unused)
    cmax = np.max(np.abs(c)) + 1e-9
    D = cmax / (0.99 * R)                            # scale of delta
    # zero-mean orthonormal directions u (score axis) and v (norm filler)
    u = np.zeros(32, dtype=np.float64); u[0] = 1.0; u[1] = -1.0; u /= np.linalg.norm(u)
    v = np.zeros(32, dtype=np.float64); v[0] = 1.0; v[1] = 1.0; v[2] = -2.0
    v -= (v @ u) * u; v /= np.linalg.norm(v)

    alpha = np.clip(c / D, -0.99 * R, 0.99 * R)      # projection onto u
    beta = np.sqrt(np.maximum(R * R - alpha * alpha, 0.0))
    E = alpha[:, None] * u[None, :] + beta[:, None] * v[None, :]   # [V,32]
    # pad token (id 0) gets a harmless valid vector
    E[0] = 0.0
    E = E.astype(np.float32)

    w["embed.weight"] = E
    w["pos.weight"] = np.zeros_like(w["pos.weight"])

    # ---- head: delta = D*u, bias difference = b ----
    delta = (D * u).astype(np.float32)
    head = np.zeros((2, 32), dtype=np.float32)
    head[0] = -0.5 * delta
    head[1] = 0.5 * delta
    w["head.weight"] = head
    w["head.bias"] = np.array([-0.5 * b, 0.5 * b], dtype=np.float32)

    # ---- keep blocks identity (already zero) ----
    for k in ["block0", "block1"]:
        for nm in ["qkv.weight", "qkv.bias", "proj.weight", "proj.bias",
                   "ff1.weight", "ff1.bias", "ff2.weight", "ff2.bias"]:
            w[f"{k}.{nm}"] = np.zeros_like(w[f"{k}.{nm}"])

    # ---- diagnostics ----
    sc = M @ wv + b
    acc = ((sc > 0.5).astype(np.float64) == y).mean()
    diag = {
        "best_lambda": float(best_lam),
        "cv_acc": float(best_acc),
        "train_acc": float(acc),
        "bias": float(b),
        "score_absmax": float(np.max(np.abs(c))),
        "delta_norm": float(np.linalg.norm(delta)),
        "embed_norm_mean": float(np.linalg.norm(E, axis=1).mean()),
        "embed_mean": float(E.mean()),
    }
    return {"weights": w, "diagnostics": diag}
