import numpy as np

def edit(w, f):
    X = f["x"]; Y = f["y"]
    N, T = X.shape
    V = 2000

    # ---- length-normalised bag-of-words design matrix Z (N x V) ----
    mask = (X != 0).astype(np.float64)
    L = mask.sum(axis=1)
    L[L == 0] = 1.0
    Z = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), T)
    cols = X.reshape(-1)
    np.add.at(Z, (rows, cols), 1.0)
    Z /= L[:, None]

    # ---- closed-form ridge fit of per-token weights w ----
    y = Y.astype(np.float64) * 2.0 - 1.0          # {-1,+1}
    lam = 1.0
    A = Z.T @ Z
    A[np.diag_indices_from(A)] += lam
    b = Z.T @ y
    try:
        wvec = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        wvec = np.linalg.lstsq(A, b, rcond=None)[0]

    # ---- threshold from class means of the fitted score ----
    s = Z @ wvec
    mu_p = s[y > 0].mean() if (y > 0).any() else 0.0
    mu_n = s[y < 0].mean() if (y < 0).any() else 0.0
    tau = 0.5 * (mu_p + mu_n)

    # ---- normalise weights into [-1,1] for the arccos construction ----
    scale = np.max(np.abs(wvec))
    if scale < 1e-12:
        scale = 1.0
    g = np.clip(wvec / scale, -1.0, 1.0)          # per-token cos(theta)

    # ---- two fixed orthonormal, zero-mean directions p, q in R^32 ----
    p = np.zeros(32, dtype=np.float64); p[0] = 1.0; p[1] = -1.0
    p /= np.linalg.norm(p)
    q = np.zeros(32, dtype=np.float64); q[0] = 1.0; q[1] = 1.0; q[2] = -2.0
    q -= (q @ p) * p
    q /= np.linalg.norm(q)

    theta = np.arccos(g)
    emb = np.cos(theta)[:, None] * p[None, :] + np.sin(theta)[:, None] * q[None, :]

    # ---- write weights ----
    w["embed.weight"] = emb.astype(np.float32)
    w["pos.weight"] = np.zeros((64, 32), dtype=np.float32)

    # keep all blocks as exact identities
    for blk in ("block0", "block1"):
        w[f"{blk}.qkv.weight"] = np.zeros((96, 32), dtype=np.float32)
        w[f"{blk}.qkv.bias"]   = np.zeros(96, dtype=np.float32)
        w[f"{blk}.proj.weight"] = np.zeros((32, 32), dtype=np.float32)
        w[f"{blk}.proj.bias"]   = np.zeros(32, dtype=np.float32)
        w[f"{blk}.ff1.weight"]  = np.zeros((64, 32), dtype=np.float32)
        w[f"{blk}.ff1.bias"]    = np.zeros(64, dtype=np.float32)
        w[f"{blk}.ff2.weight"]  = np.zeros((32, 64), dtype=np.float32)
        w[f"{blk}.ff2.bias"]    = np.zeros(32, dtype=np.float32)
        w[f"{blk}.norm1.weight"] = np.ones(32, dtype=np.float32)
        w[f"{blk}.norm1.bias"]   = np.zeros(32, dtype=np.float32)
        w[f"{blk}.norm2.weight"] = np.ones(32, dtype=np.float32)
        w[f"{blk}.norm2.bias"]   = np.zeros(32, dtype=np.float32)

    w["norm.weight"] = np.ones(32, dtype=np.float32)
    w["norm.bias"]   = np.zeros(32, dtype=np.float32)

    # ---- head: logit[1]-logit[0] = scale * (p @ pooled) ; threshold via bias ----
    # LN multiplies by ~sqrt(32); fold that constant into the head scale.
    s32 = np.sqrt(32.0)
    d = (scale / s32) * p
    head = np.zeros((2, 32), dtype=np.float64)
    head[1] = d
    head[0] = 0.0
    w["head.weight"] = head.astype(np.float32)
    # bias[1]-bias[0] = -tau  (score = (1/L) sum w(token) ; predict 1 if score > tau)
    bias = np.zeros(2, dtype=np.float64)
    bias[1] = -tau
    bias[0] = 0.0
    w["head.bias"] = bias.astype(np.float32)

    # ---- diagnostics ----
    pred = (s > tau).astype(np.int64)
    acc = float((pred == Y).mean())
    diag = {
        "train_accuracy": acc,
        "w_scale": float(scale),
        "tau": float(tau),
        "w_mean": float(wvec.mean()),
        "w_std": float(wvec.std()),
        "n_nonzero_w": int((np.abs(wvec) > 1e-6).sum()),
        "head_norm": float(np.linalg.norm(head)),
    }
    return {"weights": w, "diagnostics": diag}
