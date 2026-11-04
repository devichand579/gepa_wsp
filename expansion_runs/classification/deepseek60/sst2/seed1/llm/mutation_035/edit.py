import numpy as np

def edit(w, f):
    X = np.asarray(f["x"])            # [N,64] int64
    y = np.asarray(f["y"]).astype(np.float64)   # [N]
    N, T = X.shape
    V = w["embed.weight"].shape[0]    # 2000
    D = w["embed.weight"].shape[1]    # 32

    # ---------- 1. length-normalised bag-of-words matrix ----------
    mask = (X != 0)
    L = np.maximum(mask.sum(axis=1), 1).astype(np.float64)   # [N]
    Z = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), T)
    cols = X.reshape(-1)
    valid = mask.reshape(-1)
    np.add.at(Z, (rows[valid], cols[valid]), 1.0)
    Z /= L[:, None]

    # ---------- 2. closed-form ridge least squares on centred targets ----------
    t = 2.0 * y - 1.0                       # {-1,+1}
    lam = 2.0
    # augment with bias column
    Za = np.concatenate([Z, np.ones((N, 1))], axis=1)      # [N,V+1]
    A = Za.T @ Za
    A[np.arange(V + 1), np.arange(V + 1)] += lam
    A[V, V] -= lam                          # don't penalise bias
    rhs = Za.T @ t
    sol = np.linalg.solve(A, rhs)
    wv = sol[:V]                            # per-token score
    bv = sol[V]                             # bias

    # ---------- 3. choose global scale c to calibrate CE ----------
    raw = Z @ wv + bv                       # [N] decision values
    best_c, best_ce = 1.0, np.inf
    for c in np.linspace(0.25, 8.0, 64):
        z = c * raw
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        ce = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if ce < best_ce:
            best_ce, best_c = ce, c
    c = best_c

    s = c * wv                              # realised per-token score
    b = c * bv                              # realised bias

    # ---------- 4. build orthonormal zero-mean directions u, w ----------
    u = np.ones(D, dtype=np.float64)
    u[0] = -(D - 1)
    u /= np.linalg.norm(u)                  # zero-mean unit vector
    # fixed unit vector orthogonal to u and to 1
    wv2 = np.zeros(D, dtype=np.float64)
    wv2[0] = 1.0
    wv2[1] = -1.0
    wv2 -= (wv2 @ u) * u
    wv2 -= (wv2.sum() / D) * np.ones(D)     # remove any all-ones component
    wv2 /= np.linalg.norm(wv2)

    # ---------- 5. realise scores in embed + head ----------
    head1 = c * u                           # head[1] = c*u  -> head[1]·LN = s
    head0 = -head1

    E = np.zeros((V, D), dtype=np.float64)
    # scale s so that |s[v]| <= c*sqrt(32) with margin
    smax = np.max(np.abs(s)) if np.max(np.abs(s)) > 0 else 1.0
    cap = 0.95 * np.sqrt(D)                 # in units of c
    scale = min(1.0, cap * c / smax) if smax > 0 else 1.0
    s_scaled = s * scale
    b_scaled = b * scale

    for v in range(V):
        a = s_scaled[v] / c                 # component along u
        a = np.clip(a, -np.sqrt(D) + 1e-6, np.sqrt(D) - 1e-6)
        rem = np.sqrt(max(D - a * a, 0.0))
        E[v] = a * u + rem * wv2

    w["embed.weight"] = E.astype(np.float32)
    w["pos.weight"] = np.zeros((T, D), dtype=np.float32)

    hw = np.zeros((2, D), dtype=np.float64)
    hw[1] = head1
    hw[0] = head0
    w["head.weight"] = hw.astype(np.float32)

    hb = np.zeros(2, dtype=np.float64)
    hb[1] = b_scaled / 2.0
    hb[0] = -b_scaled / 2.0
    w["head.bias"] = hb.astype(np.float32)

    # ---------- 6. diagnostics ----------
    z = s_scaled @ Z.T + b_scaled           # [N]
    pred = (z > 0).astype(np.float64)
    acc = float(np.mean(pred == y))
    p = 1.0 / (1.0 + np.exp(-z))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    ce = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    diag = {
        "ridge_lambda": lam,
        "temperature_c": float(c),
        "realised_scale": float(scale),
        "s_norm": float(np.linalg.norm(s_scaled)),
        "s_max": float(np.max(np.abs(s_scaled))),
        "bias": float(b_scaled),
        "train_accuracy": acc,
        "train_ce": ce,
        "nonzero_token_weights": int(np.sum(np.abs(s_scaled) > 1e-8)),
        "head1_norm": float(np.linalg.norm(head1)),
        "embed_row_norm_mean": float(np.mean(np.linalg.norm(E, axis=1))),
    }
    return {"weights": w, "diagnostics": diag}
