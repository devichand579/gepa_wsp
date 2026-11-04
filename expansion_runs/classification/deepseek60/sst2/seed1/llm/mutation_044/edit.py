import numpy as np

def edit(w, f):
    # ---- 1. Build normalized count features from the full feedback pool ----
    X = np.asarray(f["x"])            # [N,64] int64
    y = np.asarray(f["y"]).astype(np.int64)  # [N]
    N, T = X.shape
    V = w["embed.weight"].shape[0]    # 2000

    # counts of non-pad tokens (id != 0)
    C = np.zeros((N, V), dtype=np.float64)
    for t in range(T):
        ids = X[:, t]
        valid = ids != 0
        np.add.at(C, (np.nonzero(valid)[0], ids[valid]), 1.0)
    L = C.sum(axis=1)                 # number of non-pad tokens
    L = np.maximum(L, 1.0)
    P = C / L[:, None]                # normalized counts = what the model computes

    # ---- 2. Closed-form ridge fit of linear classifier on P ----
    t = 2.0 * y.astype(np.float64) - 1.0   # targets in {-1,+1}
    # center features to reduce bias/regularization coupling
    mu = P.mean(axis=0, keepdims=True)
    Pc = P - mu
    # solve (Pc^T Pc + lam I) u = Pc^T t  (no reg on intercept)
    lam = 1e-2
    G = Pc.T @ Pc
    G[np.diag_indices_from(G)] += lam
    u = np.linalg.solve(G, Pc.T @ t)
    b = float(t.mean() - (mu.ravel() @ u))

    # ---- 3. Build embed rows f(j): zero-mean, unit-variance, f(j)[0] encodes u_j ----
    s = float(np.max(np.abs(u)))
    if s < 1e-12:
        s = 1.0
    # keep |a_j| safely below sqrt(31) ~ 5.568
    a = (u / s) * 5.4
    # solve 31*b^2 + 2*a*b + 31*a^2 - 960 = 0  ->  b = (-a + sqrt(29760 - 960 a^2))/31
    disc = np.maximum(29760.0 - 960.0 * a * a, 0.0)
    bb = (-a + np.sqrt(disc)) / 31.0
    cc = -(a + bb) / 30.0
    F = np.zeros((V, 32), dtype=np.float64)
    F[:, 0] = a
    F[:, 1] = bb
    F[:, 2:32] = cc[:, None]
    # sanity: zero-mean, unit-var
    F = F.astype(np.float32)

    w["embed.weight"] = F
    w["pos.weight"] = np.zeros_like(w["pos.weight"])

    # keep blocks as identity (all zeros)
    for k in list(w.keys()):
        if k.startswith("block0.") or k.startswith("block1."):
            w[k] = np.zeros_like(w[k])

    # final norm: identity LayerNorm
    w["norm.weight"] = np.ones_like(w["norm.weight"])
    w["norm.bias"] = np.zeros_like(w["norm.bias"])

    # ---- 4. Head: score = (1/L) sum_t u_t + b ----
    hw = np.zeros((2, 32), dtype=np.float32)
    hw[1, 0] = float(s / 5.4)          # so head[1] . pooled = (1/L) sum u_t
    hb = np.zeros(2, dtype=np.float32)
    hb[1] = float(b)
    w["head.weight"] = hw
    w["head.bias"] = hb

    # ---- diagnostics ----
    score = P @ u + b
    pred = (score > 0).astype(np.int64)
    acc = float((pred == y).mean())
    diag = {
        "method": "ridge_on_normalized_counts_via_LN_embed",
        "train_accuracy": acc,
        "lam": lam,
        "u_absmax": float(np.max(np.abs(u))),
        "bias": float(b),
        "mean_len": float(L.mean()),
    }
    return {"weights": w, "diagnostics": diag}
