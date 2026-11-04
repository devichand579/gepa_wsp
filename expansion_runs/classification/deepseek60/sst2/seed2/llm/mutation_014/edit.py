import numpy as np

def edit(w, f):
    ids = f["x"]                      # [4000,64] int64
    y   = f["y"]                      # [4000]
    N, L = ids.shape
    V = 2000

    # ---- 1. bag-of-words counts --------------------------------------
    BOW = np.zeros((N, V), dtype=np.float64)
    valid = (ids != 0)
    rows = np.repeat(np.arange(N), L)
    cols = ids.reshape(-1)
    m = valid.reshape(-1)
    np.add.at(BOW, (rows[m], cols[m]), 1.0)

    n = np.maximum(valid.sum(axis=1).astype(np.float64), 1.0)
    Xn = BOW / n[:, None]             # normalized BOW = mean of per-token indicators

    # ---- 2. closed-form ridge on normalized BOW ----------------------
    yt = 2.0 * y.astype(np.float64) - 1.0
    Xa = np.concatenate([Xn, np.ones((N, 1))], axis=1)   # [N, V+1]
    lam = 1.0
    D = np.eye(V + 1)
    D[V, V] = 0.0                     # do not regularize intercept
    A = Xa.T @ Xa + lam * D
    b = Xa.T @ yt
    sol = np.linalg.solve(A, b)
    s_fit = sol[:V]
    c = float(sol[V])

    # ---- 3. scale so |s[v]| <= 4 (feasibility of LN construction) ----
    M = float(np.max(np.abs(s_fit)))
    r = (4.0 / M) if M > 4.0 else 1.0
    s = s_fit * r
    g = np.zeros(32, dtype=np.float64)
    g[0] = 1.0 / r                    # g @ z_v = s_fit[v]

    # ---- 4. build zero-mean / unit-variance embedding rows -----------
    a = s                             # desired column-0 value
    target = np.maximum(32.0 - a * a * (32.0 / 31.0), 1e-6)
    p = np.sqrt(target / 2.0)
    base = -a / 31.0
    z = np.empty((V, 32), dtype=np.float64)
    z[:, 0] = a
    z[:, 1] = base + p
    z[:, 2] = base - p
    z[:, 3:] = base[:, None]
    # PAD row (excluded from the masked mean, but keep it well-formed)
    z[0] = 0.0
    z[0, 0] = np.sqrt(16.0)
    z[0, 1] = -np.sqrt(16.0)

    w["embed.weight"] = z.astype(np.float32)

    # ---- 5. head reads column 0 --------------------------------------
    hw = np.zeros((2, 32), dtype=np.float64)
    hw[1] = g
    hb = np.zeros(2, dtype=np.float64)
    hb[1] = c
    w["head.weight"] = hw.astype(np.float32)
    w["head.bias"]   = hb.astype(np.float32)

    # keep pos and all block weights at zero (identity blocks)
    for k in list(w.keys()):
        if k.startswith("block"):
            w[k] = np.zeros_like(w[k], dtype=np.float32)
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)

    # ---- 6. diagnostics ----------------------------------------------
    # quick closed-form check of the realized score on the training pool
    score = Xn @ s_fit + c
    pred = (score > 0).astype(np.int64)
    acc = float((pred == y).mean())
    diag = {
        "method": "BOW ridge + LayerNorm-identity embedding",
        "bow_shape": [int(N), int(V)],
        "lambda": lam,
        "s_fit_norm": float(np.linalg.norm(s_fit)),
        "scale_r": float(r),
        "intercept": c,
        "train_acc_estimate": acc,
        "embed_row_mean_maxabs": float(np.max(np.abs(z.mean(axis=1)))),
        "embed_row_var_maxdev": float(np.max(np.abs(z.var(axis=1) - 1.0))),
    }
    return {"weights": w, "diagnostics": diag}
