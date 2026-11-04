import numpy as np

def edit(w, f):
    X_ids = np.asarray(f["x"], dtype=np.int64)      # [N,64]
    y     = np.asarray(f["y"], dtype=np.int64)      # [N]
    N, T  = X_ids.shape
    V     = w["embed.weight"].shape[0]              # 2000
    D     = w["embed.weight"].shape[1]              # 32

    # ---- 1. Build normalized bag-of-words counts (exclude pad id 0) ----
    Cn = np.zeros((N, V), dtype=np.float64)
    for i in range(N):
        ids = X_ids[i]
        ids = ids[ids != 0]
        if ids.size:
            np.add.at(Cn[i], ids, 1.0)
    L = Cn.sum(axis=1, keepdims=True)
    L[L == 0] = 1.0
    Xn = Cn / L                                      # [N,V] token frequencies

    # ---- 2. Ridge least-squares linear classifier on Xn ----
    t = np.where(y == 1, 1.0, -1.0)
    mu = Xn.mean(axis=0, keepdims=True)
    Xc = Xn - mu
    lam = 5.0
    A = Xc.T @ Xc + lam * np.eye(V)
    g = np.linalg.solve(A, Xc.T @ t)                 # [V]
    c = -float(g @ mu.ravel())

    # scale so |g| <= 4 (needed for the LN fixed-point construction)
    gmax = np.abs(g).max()
    s = 4.0 / gmax if gmax > 4.0 else 1.0
    g = g * s
    c = c * s

    # ---- 3. Construct embed rows: zero-mean, ||row||^2 = D, first col = g ----
    # row = [g_k, -g_k, beta_k * p0]  with p0 a fixed zero-mean unit-norm vector
    # of length D-2; beta_k = sqrt(D - 2 g_k^2).  mean=0, ||row||^2 = D.
    p0 = np.zeros(D - 2, dtype=np.float64)
    if D - 2 > 0:
        p0[:] = 1.0
        p0 -= p0.mean()
        nrm = np.linalg.norm(p0)
        if nrm > 0:
            p0 /= nrm
    beta = np.sqrt(np.maximum(D - 2.0 * g * g, 1e-6))
    E = np.zeros((V, D), dtype=np.float64)
    E[:, 0] = g
    if D >= 2:
        E[:, 1] = -g
    if D > 2:
        E[:, 2:] = beta[:, None] * p0[None, :]

    w["embed.weight"] = E.astype(np.float32)
    w["pos.weight"]   = np.zeros_like(w["pos.weight"])

    # ---- 4. head: decision = features[:,0] + c  =>  u = e_0 ----
    u = np.zeros(D, dtype=np.float64)
    u[0] = 1.0
    head = np.zeros((2, D), dtype=np.float64)
    head[1] = 0.5 * u
    head[0] = -0.5 * u
    bias = np.array([-0.5 * c, 0.5 * c], dtype=np.float64)
    w["head.weight"] = head.astype(np.float32)
    w["head.bias"]   = bias.astype(np.float32)

    # ---- 5. diagnostics ----
    feats = Xn @ E                                   # predicted pooled features
    dec   = feats[:, 0] + c
    pred  = (dec > 0).astype(np.int64)
    acc   = float((pred == y).mean())
    diag = {
        "train_acc_on_counts": acc,
        "g_norm": float(np.linalg.norm(g)),
        "g_absmax": float(np.abs(g).max()),
        "c": float(c),
        "embed_row_norm_sq_mean": float((E * E).sum(axis=1).mean()),
        "embed_row_mean_absmax": float(np.abs(E.mean(axis=1)).max()),
        "n_nonzero_tokens": int((np.abs(g) > 1e-6).sum()),
    }
    return {"weights": w, "diagnostics": diag}
