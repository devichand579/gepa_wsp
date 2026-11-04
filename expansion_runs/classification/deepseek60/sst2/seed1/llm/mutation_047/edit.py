import numpy as np

def edit(w, f):
    X = np.asarray(f["x"])          # [N,64] int64
    Y = np.asarray(f["y"])          # [N]
    N, T = X.shape
    V = w["embed.weight"].shape[0]  # 2000
    D = w["embed.weight"].shape[1]  # 32

    # ---- 1. Build normalized count matrix P[i,j] = count_j / L_i ----
    mask = (X != 0)
    lengths = np.maximum(mask.sum(axis=1).astype(np.float64), 1.0)
    C = np.zeros((N, V), dtype=np.float64)
    for i in range(N):
        row = X[i][mask[i]]
        if row.size:
            C[i] = np.bincount(row, minlength=V)
    P = C / lengths[:, None]

    # ---- 2. Closed-form ridge fit of u (per-token weights) and b ----
    yp = 2.0 * Y.astype(np.float64) - 1.0          # -1 / +1
    PtP = P.T @ P                                   # [V,V]
    Pts = P.T @ yp                                  # [V]
    s   = P.sum(axis=0)                             # [V]
    sy  = yp.sum()
    lam = 1.0
    # Solve [ PtP + lam I , s ; s^T , N ] [u;b] = [Pts ; sy]
    A = np.empty((V + 1, V + 1), dtype=np.float64)
    A[:V, :V] = PtP + lam * np.eye(V)
    A[:V, V]  = s
    A[V, :V]  = s
    A[V, V]   = float(N)
    rhs = np.empty(V + 1, dtype=np.float64)
    rhs[:V] = Pts
    rhs[V]  = sy
    sol = np.linalg.solve(A, rhs)
    u = sol[:V]
    b = float(sol[V])

    # ---- 3. Choose H so that |t_j| = |u_j/H| <= sqrt(31) ----
    umax = float(np.max(np.abs(u))) if V else 0.0
    H = max(1.0, umax / np.sqrt(31.0) * 1.0001)
    # keep head entries bounded
    H = min(H, 900.0)

    # ---- 4. Build embed rows: e_j = sigma * v_j with v_j zero-mean, ||v_j||^2 = 32 ----
    # v_j[0] = t_j, v_j[1] = s_j, v_j[k>=2] = c_j = -(t_j+s_j)/30
    t = u / H
    # clamp for numerical safety
    t = np.clip(t, -np.sqrt(31.0) * 0.9999, np.sqrt(31.0) * 0.9999)
    disc = 29760.0 - 960.0 * t * t
    disc = np.maximum(disc, 0.0)
    s = (-t + np.sqrt(disc)) / 31.0
    c = -(t + s) / 30.0

    Vmat = np.empty((V, D), dtype=np.float64)
    Vmat[:, 0] = t
    Vmat[:, 1] = s
    Vmat[:, 2:] = c[:, None]

    sigma = 100.0
    E = sigma * Vmat
    # safety clamp
    E = np.clip(E, -1000.0, 1000.0)

    # ---- 5. Set weights ----
    w["embed.weight"] = E.astype(np.float32)

    # head: logits[0]=0, logits[1] = H * pooled[0] + b
    head = np.zeros((2, D), dtype=np.float32)
    head[1, 0] = np.float32(H)
    w["head.weight"] = head
    hb = np.zeros((2,), dtype=np.float32)
    hb[0] = 0.0
    hb[1] = np.float32(b)
    w["head.bias"] = hb

    # norm: identity LayerNorm
    w["norm.weight"] = np.ones((D,), dtype=np.float32)
    w["norm.bias"]   = np.zeros((D,), dtype=np.float32)

    # pos: zero (identity)
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)

    # blocks: keep identity (all zeros) — already zero, but enforce for safety
    for blk in ("block0", "block1"):
        for name in ("norm1.weight", "norm1.bias", "qkv.weight", "qkv.bias",
                     "proj.weight", "proj.bias", "norm2.weight", "norm2.bias",
                     "ff1.weight", "ff1.bias", "ff2.weight", "ff2.bias"):
            key = f"{blk}.{name}"
            if key in w:
                w[key] = np.zeros_like(w[key], dtype=np.float32)

    # ---- 6. Diagnostics ----
    # Evaluate training accuracy of the linear model
    scores = P @ u + b
    preds = (scores > 0).astype(np.int64)
    acc = float((preds == Y).mean())

    diag = {
        "method": "closed_form_ridge_on_normalized_counts_via_LN_direction_embed",
        "train_accuracy_linear": acc,
        "lam": lam,
        "H": float(H),
        "bias": float(b),
        "u_absmax": float(np.max(np.abs(u))),
        "u_norm": float(np.linalg.norm(u)),
        "embed_absmax": float(np.max(np.abs(E))),
        "mean_length": float(lengths.mean()),
        "n_tokens": int(V),
    }
    return {"weights": w, "diagnostics": diag}
