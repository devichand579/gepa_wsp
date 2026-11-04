import numpy as np

def edit(w, f):
    x = f["x"].astype(np.int64)          # [N,64]
    y = f["y"].astype(np.int64)          # [N]
    N, L = x.shape
    V = w["embed.weight"].shape[0]       # 2000
    d = w["embed.weight"].shape[1]       # 32

    # ---- 1. Build normalized bag-of-words counts (exclude pad id 0) ----
    mask = (x != 0).astype(np.float64)   # [N,64]
    n_d = mask.sum(axis=1)               # [N]
    n_d = np.maximum(n_d, 1.0)
    X = np.zeros((N, V), dtype=np.float64)
    for i in range(N):
        ids = x[i][mask[i] > 0]
        np.add.at(X[i], ids, 1.0)
    X /= n_d[:, None]                    # normalized counts

    # ---- 2. Ridge classifier on normalized counts ----
    Y = np.zeros((N, 2), dtype=np.float64)
    Y[np.arange(N), y] = 1.0
    Xm = X.mean(axis=0, keepdims=True)
    Ym = Y.mean(axis=0, keepdims=True)
    Xc = X - Xm
    Yc = Y - Ym
    lam = 1.0
    A = Xc.T @ Xc + lam * np.eye(V)
    M = np.linalg.solve(A, Xc.T @ Yc)    # [V,2]  (centered weights)
    b = (Ym - Xm @ M).ravel()            # [2]

    # ---- 3. Scale M so each row satisfies a^2+b^2+(a+b)^2/30 <= 1 ----
    a = M[:, 0]; bb = M[:, 1]
    need = a * a + bb * bb + (a + bb) ** 2 / (d - 2)
    scale = 1.0
    mx = need.max()
    if mx > 0.9:
        scale = np.sqrt(0.9 / mx)
    M = M * scale
    b = b * scale

    # ---- 4. Build D [V,32] with zero-mean, unit-norm rows, D[:,0:2]=M ----
    D = np.zeros((V, d), dtype=np.float64)
    D[:, 0] = M[:, 0]
    D[:, 1] = M[:, 1]
    a = M[:, 0]; bb = M[:, 1]
    c = -(a + bb) / (d - 2)              # base value for free coords
    r2 = 1.0 - a * a - bb * bb - (d - 2) * c * c
    r2 = np.maximum(r2, 0.0)
    r = np.sqrt(r2)
    # zero-sum perturbation with norm r per row
    rng = np.random.RandomState(0)
    U = rng.randn(V, d - 2)
    U -= U.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(U, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    P = U / nrm * r[:, None]
    D[:, 2:] = c[:, None] + P

    # ---- 5. Write embed, zero out pos & blocks, set head = [I2 | 0] ----
    w["embed.weight"] = D.astype(np.float32)
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)
    for k in ["block0", "block1"]:
        for sub in ["qkv", "proj", "ff1", "ff2"]:
            w[f"{k}.{sub}.weight"] = np.zeros_like(w[f"{k}.{sub}.weight"], dtype=np.float32)
            w[f"{k}.{sub}.bias"] = np.zeros_like(w[f"{k}.{sub}.bias"], dtype=np.float32)
    for k in ["block0", "block1"]:
        for sub in ["norm1", "norm2"]:
            w[f"{k}.{sub}.weight"] = np.ones_like(w[f"{k}.{sub}.weight"], dtype=np.float32)
            w[f"{k}.{sub}.bias"] = np.zeros_like(w[f"{k}.{sub}.bias"], dtype=np.float32)
    w["norm.weight"] = np.ones_like(w["norm.weight"], dtype=np.float32)
    w["norm.bias"] = np.zeros_like(w["norm.bias"], dtype=np.float32)

    head = np.zeros((2, d), dtype=np.float32)
    head[0, 0] = 1.0
    head[1, 1] = 1.0
    w["head.weight"] = head
    w["head.bias"] = b.astype(np.float32)

    # ---- 6. Compute ACTUAL pooled features with LN, refit head in closed form ----
    E = D.astype(np.float64)
    emb = E[x]                            # [N,64,32]
    mu = emb.mean(axis=2, keepdims=True)
    var = emb.var(axis=2, keepdims=True)
    ln = (emb - mu) / np.sqrt(var + 1e-5)
    ln = ln * mask[:, :, None]            # zero out pad positions
    pooled = ln.sum(axis=1) / n_d[:, None]  # [N,32]

    # refit head: minimize ||pooled @ H^T + hb - Y||^2 + lam2*||H||^2
    Pm = pooled.mean(axis=0, keepdims=True)
    Pc = pooled - Pm
    lam2 = 1.0
    Ap = Pc.T @ Pc + lam2 * np.eye(d)
    H = np.linalg.solve(Ap, Pc.T @ Yc)   # [32,2]
    hb = (Ym - Pm @ H).ravel()           # [2]
    w["head.weight"] = H.T.astype(np.float32)   # [2,32]
    w["head.bias"] = hb.astype(np.float32)

    # ---- diagnostics ----
    logits = pooled @ H + hb
    pred = logits.argmax(axis=1)
    acc = float((pred == y).mean())
    diag = {
        "train_acc": acc,
        "M_scale": float(scale),
        "M_absmax": float(np.abs(M).max()),
        "D_row_norm_min": float(np.linalg.norm(D, axis=1).min()),
        "D_row_norm_max": float(np.linalg.norm(D, axis=1).max()),
        "D_row_mean_absmax": float(np.abs(D.mean(axis=1)).max()),
        "head_norm": float(np.linalg.norm(H)),
        "head_bias": hb.tolist(),
        "pooled_std_mean": float(pooled.std(axis=0).mean()),
    }
    return {"weights": w, "diagnostics": diag}
