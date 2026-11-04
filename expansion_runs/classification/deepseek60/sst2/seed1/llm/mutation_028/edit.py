import numpy as np

def edit(w, f):
    X = np.asarray(f["x"], dtype=np.int64)      # [N,64]
    y = np.asarray(f["y"], dtype=np.int64)      # [N]
    N, L = X.shape
    V = w["embed.weight"].shape[0]              # 2000
    D = w["embed.weight"].shape[1]              # 32

    # ---- 1. Build normalized bag-of-words matrix (pad id 0 excluded) ----
    mask = (X != 0).astype(np.float64)          # [N,64]
    lens = np.maximum(mask.sum(axis=1), 1.0)    # [N]
    B = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), L)
    cols = X.reshape(-1)
    vals = mask.reshape(-1) / np.repeat(lens, L)
    np.add.at(B, (rows, cols), vals)
    B[:, 0] = 0.0                               # pad never contributes

    # ---- 2. Closed-form ridge fit of a linear BOW classifier ----
    t = np.where(y == 1, 1.0, -1.0)
    # center features & target so the bias is handled exactly
    Bm = B.mean(axis=0, keepdims=True)
    tm = t.mean()
    Bc = B - Bm
    tc = t - tm
    # Gram matrix with ridge; solve (Bc'Bc + lam I) w = Bc'tc
    G = Bc.T @ Bc
    lam = 1.0
    # scale ridge relative to feature energy for stability
    lam = lam * (np.trace(G) / max(V, 1)) * 0.05
    G[np.diag_indices_from(G)] += lam
    rhs = Bc.T @ tc
    try:
        wv = np.linalg.solve(G, rhs)
    except np.linalg.LinAlgError:
        wv = np.linalg.lstsq(G, rhs, rcond=None)[0]
    bias = tm - float(Bm @ wv)

    # shrink extreme weights slightly for generalization
    wv = np.clip(wv, -60.0, 60.0)
    wv[0] = 0.0

    # ---- 3. Realize wv in the network via a 2-D embedding subspace ----
    # t1, t2 orthonormal, both orthogonal to the all-ones vector
    t1 = np.zeros(D); t1[0] = 1.0; t1[1] = -1.0; t1 /= np.linalg.norm(t1)
    t2 = np.zeros(D); t2[0] = 1.0; t2[1] = 1.0; t2[2] = -2.0
    t2 /= np.linalg.norm(t2)

    maxw = float(np.max(np.abs(wv)))
    if maxw < 1e-8:
        maxw = 1e-8
    R = maxw / np.sqrt(D) * 1.0001            # head direction magnitude
    # head direction h = R * t1  (phi = 0)
    h = R * t1

    # per-token angle: sqrt(D)*R*cos(theta_v) = w_v
    arg = np.clip(wv / (np.sqrt(D) * R), -1.0, 1.0)
    theta = np.arccos(arg)                    # in [0, pi]
    E = np.outer(np.cos(theta), t1) + np.outer(np.sin(theta), t2)
    E[0] = t1                                 # pad row: finite, unused

    # ---- 4. Write weights ----
    w["embed.weight"] = E.astype(np.float32)
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)

    for blk in ("block0", "block1"):
        for name in ("norm1.weight", "norm1.bias", "qkv.weight", "qkv.bias",
                     "proj.weight", "proj.bias", "norm2.weight", "norm2.bias",
                     "ff1.weight", "ff1.bias", "ff2.weight", "ff2.bias"):
            w[blk + "." + name] = np.zeros_like(w[blk + "." + name], dtype=np.float32)
        # keep LayerNorm weights at 1 for norm1/norm2 so blocks stay identity-free
        w[blk + ".norm1.weight"] = np.ones_like(w[blk + ".norm1.weight"], dtype=np.float32)
        w[blk + ".norm2.weight"] = np.ones_like(w[blk + ".norm2.weight"], dtype=np.float32)

    w["norm.weight"] = np.ones_like(w["norm.weight"], dtype=np.float32)
    w["norm.bias"] = np.zeros_like(w["norm.bias"], dtype=np.float32)

    hw = np.zeros((2, D), dtype=np.float32)
    hw[1] = h.astype(np.float32)
    hb = np.zeros(2, dtype=np.float32)
    hb[1] = float(bias)
    w["head.weight"] = hw
    w["head.bias"] = hb

    # ---- 5. Diagnostics ----
    # verify realization on a few tokens
    def ln(v):
        v = v.astype(np.float64)
        m = v.mean()
        s = v.std()
        if s < 1e-12:
            return np.zeros_like(v)
        return (v - m) / s
    idx = np.array([1, 2, 3, 4, 5, 10, 20, 100])
    realized = np.array([float(h @ ln(E[i])) for i in idx])
    target = wv[idx]
    err = float(np.max(np.abs(realized - target)))

    # training accuracy of the fitted linear model
    scores = B @ wv + bias
    acc = float(((scores > 0).astype(np.int64) == y).mean())
    ce = float(np.mean(np.log1p(np.exp(-np.where(y == 1, 1.0, -1.0) * scores))))

    diag = {
        "n_samples": int(N),
        "vocab": int(V),
        "ridge_lambda": float(lam),
        "w_norm": float(np.linalg.norm(wv)),
        "w_max": float(maxw),
        "bias": float(bias),
        "realization_max_err": err,
        "train_accuracy": acc,
        "train_ce": ce,
        "nonzero_token_weights": int(np.sum(np.abs(wv) > 1e-6)),
    }
    return {"weights": w, "diagnostics": diag}
