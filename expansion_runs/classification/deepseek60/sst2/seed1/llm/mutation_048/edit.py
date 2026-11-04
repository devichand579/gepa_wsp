import numpy as np

def edit(w, f):
    X_ids = np.asarray(f["x"], dtype=np.int64)      # [N,64]
    y     = np.asarray(f["y"], dtype=np.int64)      # [N]
    N, T  = X_ids.shape
    V     = w["embed.weight"].shape[0]              # 2000

    # ---- normalized bag-of-words counts P[i,j] = count_j / L_i ----
    mask  = (X_ids != 0).astype(np.float64)         # [N,64]
    L     = mask.sum(axis=1)                        # [N]
    L[L == 0] = 1.0
    P     = np.zeros((N, V), dtype=np.float64)
    rows  = np.repeat(np.arange(N), T)
    cols  = X_ids.reshape(-1)
    np.add.at(P, (rows, cols), 1.0)
    P    /= L[:, None]

    # ---- closed-form ridge on the exact feature the model computes ----
    t     = (2.0 * y - 1.0).astype(np.float64)      # {-1,+1}
    X     = np.concatenate([P, np.ones((N, 1))], axis=1)   # [N, V+1]
    lam   = 1.0
    G     = X.T @ X
    G    += lam * np.eye(G.shape[0])
    rhs   = X.T @ t
    sol   = np.linalg.solve(G, rhs)
    u     = sol[:V].astype(np.float64)
    b     = float(sol[V])

    # ---- build LayerNorm-compatible embed rows ----
    # want z_j with sum=0, ||z_j||^2=32, z_j[1] = s_j (head reads channel 1)
    umax  = float(np.max(np.abs(u)))
    c     = max(1.0, umax / 5.0)                    # keep |s_j| <= 5
    s     = u / c                                   # target z_j[1]

    Z = np.zeros((V, 32), dtype=np.float64)
    # indices 2..30 are 'a' (29 slots), index 31 is 'b'
    # equations: s + 29a + b = 0 ; s^2 + 29a^2 + b^2 = 32
    # => 870 a^2 + 58 s a + (2 s^2 - 32) = 0
    disc = 111360.0 - 3596.0 * s * s
    disc = np.maximum(disc, 0.0)
    a    = (-58.0 * s + np.sqrt(disc)) / 1740.0
    bb   = -s - 29.0 * a
    Z[:, 1]  = s
    Z[:, 2:31] = a[:, None]
    Z[:, 31] = bb

    # pad row: keep it finite / harmless (excluded from pooling anyway)
    Z[0, :] = 0.0
    Z[0, 1] = 0.0

    # ---- write weights ----
    w["embed.weight"] = Z.astype(np.float32)
    w["pos.weight"]   = np.zeros_like(w["pos.weight"], dtype=np.float32)

    # blocks stay identity
    for k in ("block0", "block1"):
        for name in ("norm1.weight","norm1.bias","qkv.weight","qkv.bias",
                     "proj.weight","proj.bias","norm2.weight","norm2.bias",
                     "ff1.weight","ff1.bias","ff2.weight","ff2.bias"):
            key = f"{k}.{name}"
            if key in w:
                w[key] = np.zeros_like(w[key], dtype=np.float32)

    # final norm: standard LayerNorm
    w["norm.weight"] = np.ones_like(w["norm.weight"], dtype=np.float32)
    w["norm.bias"]   = np.zeros_like(w["norm.bias"], dtype=np.float32)

    # head: read channel 1 with scale c, bias b
    hw = np.zeros_like(w["head.weight"], dtype=np.float32)
    hw[1, 1] = np.float32(c)
    w["head.weight"] = hw
    w["head.bias"]   = np.array([0.0, b], dtype=np.float32)

    # ---- diagnostics ----
    logits = (P @ u + b)
    pred   = (logits > 0).astype(np.int64)
    acc    = float((pred == y).mean())
    diag = {
        "method": "ridge_on_normalized_counts_LN_embed",
        "train_acc": acc,
        "lam": lam,
        "u_absmax": umax,
        "u_norm": float(np.linalg.norm(u)),
        "c": c,
        "bias": b,
        "mean_len": float(L.mean()),
        "embed_absmax": float(np.abs(Z).max()),
        "head_scale": c,
    }
    return {"weights": w, "diagnostics": diag}
