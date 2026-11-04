import numpy as np

def edit(w, f):
    x = np.asarray(f["x"], dtype=np.int64)          # [N,64]
    y = np.asarray(f["y"], dtype=np.int64).astype(np.float64)
    N, L = x.shape
    V = w["embed.weight"].shape[0]

    # ---- masked-mean one-hot design matrix (excluding pad id 0) ----
    mask = (x != 0).astype(np.float64)               # [N,64]
    lens = np.maximum(mask.sum(1), 1.0)              # [N]
    X = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), L)
    cols = x.reshape(-1)
    vals = (mask.reshape(-1) / np.repeat(lens, L))
    np.add.at(X, (rows, cols), vals)
    # drop pad column (id 0) -> keep it at zero
    X[:, 0] = 0.0

    # ---- ridge least squares for per-token weights ----
    lam = 30.0
    G = X.T @ X
    G[np.diag_indices_from(G)] += lam
    rhs = X.T @ (2.0 * y - 1.0)                      # target +1 / -1
    s = np.linalg.solve(G, rhs)                      # [V]
    s[0] = 0.0

    # centre the scores (bias handled by head.bias)
    b = 0.0
    # choose a global scale for numerical safety
    amax = float(np.max(np.abs(s)))
    if amax < 1e-8:
        amax = 1.0
    s = s / amax
    # bias: optimal threshold via mean of class means
    pos = (2.0 * y - 1.0)
    # refit bias so that sign(bias + s·X) matches y as well as possible
    raw = X @ s
    # simple closed-form threshold: midpoint of class means
    b = -0.5 * (raw[pos > 0].mean() + raw[pos < 0].mean())

    # ---- realise s_t in the embedding via angle encoding ----
    d = 32
    p = np.zeros(d, dtype=np.float64); p[0] = 1.0/np.sqrt(2); p[1] = -1.0/np.sqrt(2)
    q = np.zeros(d, dtype=np.float64); q[0] = 1.0/np.sqrt(6); q[1] = 1.0/np.sqrt(6); q[2] = -2.0/np.sqrt(6)
    # p,q orthonormal, zero-mean, unit-norm
    theta = np.arccos(np.clip(s, -1.0, 1.0))         # [V]
    emb = np.cos(theta)[:, None] * p[None, :] + np.sin(theta)[:, None] * q[None, :]
    # scale so that LN(emb) = sqrt(32) * (cos p + sin q); keep unit norm
    emb = emb.astype(np.float32)
    w["embed.weight"][:] = emb

    # head: h * p, plus bias; head[1]-head[0] = h*p  => use head rows
    h = 1.0
    head = np.zeros((2, d), dtype=np.float32)
    head[0] = -0.5 * h * p.astype(np.float32)
    head[1] = 0.5 * h * p.astype(np.float32)
    w["head.weight"][:] = head
    w["head.bias"][:] = np.array([-0.5 * b * np.sqrt(32.0) * h,
                                  0.5 * b * np.sqrt(32.0) * h], dtype=np.float32)

    # keep everything else zero / identity (blocks already zero, LN identity)
    for k in ["pos.weight"]:
        w[k][:] = 0.0
    for blk in ["block0", "block1"]:
        for sub in ["qkv.weight", "qkv.bias", "proj.weight", "proj.bias",
                    "ff1.weight", "ff1.bias", "ff2.weight", "ff2.bias"]:
            w[f"{blk}.{sub}"][:] = 0.0
        for nm in ["norm1.weight", "norm2.weight"]:
            w[f"{blk}.{nm}"][:] = 1.0
        for nm in ["norm1.bias", "norm2.bias"]:
            w[f"{blk}.{nm}"][:] = 0.0
    w["norm.weight"][:] = 1.0
    w["norm.bias"][:] = 0.0

    # diagnostics
    pred = (raw + b) > 0
    acc = float((pred.astype(np.int64) == y.astype(np.int64)).mean())
    diag = {
        "train_acc": acc,
        "s_absmax": float(np.max(np.abs(s))),
        "s_std": float(np.std(s)),
        "n_nonzero_s": int(np.count_nonzero(np.abs(s) > 1e-6)),
        "bias": float(b),
        "lam": lam,
    }
    return {"weights": w, "diagnostics": diag}
