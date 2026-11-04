import numpy as np

def edit(w, f):
    X = f["x"].astype(np.int64)          # [N,64]
    y = f["y"].astype(np.int64)          # [N]
    N, L = X.shape
    V = w["embed.weight"].shape[0]       # 2000
    D = w["embed.weight"].shape[1]       # 32

    mask = (X != 0).astype(np.float64)   # [N,64]
    counts = np.zeros((V,), dtype=np.float64)
    # accumulate token counts
    flat = X.reshape(-1)
    np.add.at(counts, flat, 1.0)
    # exclude pad from counts for weight estimation
    counts_np = counts.copy()
    counts_np[0] = 0.0

    # ---- closed-form LDA-style token weights on bag-of-words ----
    # per-class token frequency
    c0 = np.zeros(V); c1 = np.zeros(V)
    for i in range(N):
        ids = X[i][mask[i] > 0]
        if y[i] == 0:
            np.add.at(c0, ids, 1.0)
        else:
            np.add.at(c1, ids, 1.0)
    n0 = max((y == 0).sum(), 1)
    n1 = max((y == 1).sum(), 1)
    p0 = (c0 + 1.0) / (c0.sum() + V)
    p1 = (c1 + 1.0) / (c1.sum() + V)
    # log-odds weights
    tw = np.log(p1) - np.log(p0)         # [V]
    tw[0] = 0.0
    # normalize to moderate scale
    s = np.std(tw[1:])
    if s > 1e-8:
        tw = tw / s
    tw = np.clip(tw, -8.0, 8.0)

    # ---- design embedding: 2 informative columns ----
    E = np.zeros((V, D), dtype=np.float64)
    E[:, 0] = tw                          # token weight direction
    E[:, 1] = 1.0                         # constant column (length/direction info)
    # small random-ish orthogonal filler to spread LN directions (deterministic)
    idx = np.arange(V, dtype=np.float64)
    E[:, 2] = np.sin(idx * 0.017) * 0.5
    E[:, 3] = np.cos(idx * 0.013) * 0.5
    E[0, :] = 0.0                         # pad row stays zero

    # ---- recompute pooled features with new embedding (blocks identity) ----
    # per-token LayerNorm over last dim (eps=1e-5), weight=1 bias=0
    def ln(x, eps=1e-5):
        m = x.mean(axis=-1, keepdims=True)
        v = x.var(axis=-1, keepdims=True)
        return (x - m) / np.sqrt(v + eps)

    Et = E[X]                             # [N,64,D]
    LNo = ln(Et)                          # [N,64,D]
    m = mask[..., None]
    F = (LNo * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1.0)  # [N,D]

    # ---- closed-form ridge fit of head on recomputed features ----
    Y = np.zeros((N, 2), dtype=np.float64)
    Y[np.arange(N), y] = 1.0
    lam = 1e-2
    A = F.T @ F + lam * np.eye(D)
    B = F.T @ Y
    W = np.linalg.solve(A, B).T           # [2,D]
    b = Y.mean(axis=0) - W @ F.mean(axis=0)

    # ---- write back ----
    w["embed.weight"] = E.astype(np.float32)
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)
    w["norm.weight"] = np.ones(D, dtype=np.float32)
    w["norm.bias"] = np.zeros(D, dtype=np.float32)
    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

    # keep blocks identity
    for k in list(w.keys()):
        if k.startswith("block"):
            w[k] = np.zeros_like(w[k], dtype=np.float32)

    # clip for safety
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)

    # diagnostics
    pred = np.argmax(F @ W.T + b, axis=1)
    train_acc = float((pred == y).mean())
    diag = {
        "train_accuracy": train_acc,
        "head_weight_norm": float(np.linalg.norm(W)),
        "tw_std": float(np.std(tw[1:])),
        "feature_mean_abs": float(np.mean(np.abs(F))),
    }
    return {"weights": w, "diagnostics": diag}
