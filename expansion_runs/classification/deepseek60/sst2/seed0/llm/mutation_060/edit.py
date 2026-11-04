import numpy as np

def edit(w, f):
    X = f["x"]                      # [N,64] int64
    y = f["y"].astype(np.float64)   # [N]
    N, L = X.shape
    V = w["embed.weight"].shape[0]  # 2000

    mask = (X != 0).astype(np.float64)          # [N,64]
    counts = mask.sum(1)                        # [N]
    counts[counts == 0] = 1.0

    # design matrix: length-normalised one-hot counts  [N, V]
    A = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), L)
    cols = X.reshape(-1)
    np.add.at(A, (rows, cols), 1.0)
    A /= counts[:, None]
    A[:, 0] = 0.0                                # padding never contributes

    # held-out split for choosing ridge strength (closed form, no iteration)
    rng = np.random.RandomState(0)
    perm = rng.permutation(N)
    ntr = int(0.8 * N)
    tr, va = perm[:ntr], perm[ntr:]

    best = None
    for lam in (1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0):
        AtA = A[tr].T @ A[tr] + lam * np.eye(V)
        Aty = A[tr].T @ (y[tr] - 0.5)
        s = np.linalg.solve(AtA, Aty)
        b = 0.5 - float(np.mean(A[tr] @ s))
        pred = (A[va] @ s + b) > 0.5
        acc = float(np.mean(pred == (y[va] > 0.5)))
        if best is None or acc > best[0]:
            best = (acc, lam, s, b)

    _, lam, s, b = best

    # centre scores (LN removes the mean of the embedding, so only the
    # zero-mean part of s survives the pooling) and rescale
    s = s - s.mean()
    smax = float(np.max(np.abs(s)))
    if smax < 1e-8:
        smax = 1.0
    s = s / smax

    # two fixed orthonormal zero-mean directions in R^32
    p = np.zeros(32); p[0] = 1.0;  p[1] = -1.0
    p /= np.linalg.norm(p)
    q = np.zeros(32); q[2] = 1.0;  q[3] = -1.0
    q /= np.linalg.norm(q)

    theta = np.arccos(np.clip(s, -1.0, 1.0))     # [V]
    emb = np.cos(theta)[:, None] * p[None, :] + np.sin(theta)[:, None] * q[None, :]

    w["embed.weight"] = emb.astype(np.float32)
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)

    # keep all transformer blocks exactly identity
    for k in list(w.keys()):
        if k.startswith("block0.") or k.startswith("block1."):
            w[k] = np.zeros_like(w[k], dtype=np.float32)
    for k in list(w.keys()):
        if k.endswith("norm1.weight") or k.endswith("norm2.weight") or k == "norm.weight":
            w[k] = np.ones_like(w[k], dtype=np.float32)
        if k.endswith("norm1.bias") or k.endswith("norm2.bias") or k == "norm.bias":
            w[k] = np.zeros_like(w[k], dtype=np.float32)

    # head: row0 = 0, row1 = smax * p / sqrt(32)  ->  head[1].LN(emb[t]) = s_t
    head = np.zeros((2, 32), dtype=np.float32)
    head[1] = (smax * p / np.sqrt(32.0)).astype(np.float32)
    w["head.weight"] = head
    bias = np.zeros(2, dtype=np.float32)
    bias[1] = np.float32(b - 0.5)
    w["head.bias"] = bias

    # diagnostics
    diag = {
        "lam": float(lam),
        "val_acc": float(best[0]),
        "smax": float(smax),
        "bias": float(b - 0.5),
        "n_nonzero_s": int(np.sum(np.abs(s) > 1e-6)),
        "embed_norm_mean": float(np.mean(np.linalg.norm(w["embed.weight"], axis=1))),
    }
    return {"weights": w, "diagnostics": diag}
