import numpy as np

def edit(w, f):
    X = f["x"].astype(np.int64)          # [N,64]
    y = f["y"].astype(np.int64)          # [N]
    N, T = X.shape
    V = w["embed.weight"].shape[0]       # 2000
    D = w["embed.weight"].shape[1]       # 32

    # ---- build length-normalized bag-of-words design matrix ----
    mask = (X != 0)
    L = mask.sum(axis=1).astype(np.float64)
    L[L == 0] = 1.0
    # counts per token (length-normalized)
    F = np.zeros((N, V), dtype=np.float64)
    for i in range(N):
        ids = X[i][mask[i]]
        if ids.size:
            np.add.at(F[i], ids, 1.0)
    F /= L[:, None]

    # ---- closed-form ridge least squares on (F, y) ----
    # target: +1 for class1, -1 for class0
    t = np.where(y == 1, 1.0, -1.0)
    lam = 1.0
    # solve (F^T F + lam I) s = F^T t   (skip pad column 0)
    A = F[:, 1:].T @ F[:, 1:]
    b = F[:, 1:].T @ t
    A[np.diag_indices_from(A)] += lam
    s = np.zeros(V, dtype=np.float64)
    s[1:] = np.linalg.solve(A, b)
    # bias
    b0 = float(np.mean(t - F @ s))

    # ---- choose a scale c so realization is feasible ----
    # head_diff = c * u, u unit, zero-mean. Need |s_v| <= sqrt(32)*c
    smax = float(np.max(np.abs(s))) if s.size else 1.0
    c = max(smax / np.sqrt(32.0), 1e-6) * 1.0001

    # u : unit, zero-mean direction in R^D
    u = np.ones(D, dtype=np.float64)
    u[0::2] = 1.0
    u[1::2] = -1.0
    u -= u.mean()
    u /= np.linalg.norm(u)
    # u_perp : unit, zero-mean, orthogonal to u
    up = np.zeros(D, dtype=np.float64)
    up[0::2] = 1.0
    up[1::2] = -1.0
    up -= up.mean()
    up -= (up @ u) * u
    up /= np.linalg.norm(up)

    # ---- realize embed[v] = s_v*u + sqrt(32 c^2 - s_v^2)*up ----
    E = np.zeros((V, D), dtype=np.float64)
    for v in range(V):
        sv = s[v]
        val = 32.0 * c * c - sv * sv
        if val < 0:
            val = 0.0
        E[v] = sv * u + np.sqrt(val) * up
    # pad row: zero embedding (LN(0)=0, contributes nothing)
    E[0] = 0.0

    # ---- head: head[1]-head[0] = c*u, symmetric split ----
    hd = c * u
    head = np.zeros((2, D), dtype=np.float64)
    head[1] = 0.5 * hd
    head[0] = -0.5 * hd
    bias = np.array([-0.5 * b0, 0.5 * b0], dtype=np.float64)

    # ---- temperature calibration on the pool ----
    # compute realized logit-diff = F @ s + b0  (exact by construction)
    logit_diff = F @ s + b0
    # fit temperature a and shift so that softmax matches labels (min CE)
    # logits: [ -a*ld/2 - a*b0/2 , +a*ld/2 + a*b0/2 ]
    best_a, best_ce = 1.0, 1e18
    for a in np.linspace(0.2, 6.0, 60):
        z = a * logit_diff
        p1 = 1.0 / (1.0 + np.exp(-z))
        p1 = np.clip(p1, 1e-9, 1 - 1e-9)
        ce = -np.mean(np.where(y == 1, np.log(p1), np.log(1 - p1)))
        if ce < best_ce:
            best_ce, best_a = ce, a
    a = best_a

    # apply temperature to head & bias
    head *= a
    bias *= a

    w["embed.weight"] = E.astype(np.float32)
    w["head.weight"] = head.astype(np.float32)
    w["head.bias"] = bias.astype(np.float32)

    # clip to safe range
    for k in ["embed.weight", "head.weight", "head.bias"]:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)

    train_pred = (logit_diff > 0).astype(np.int64)
    train_acc = float(np.mean(train_pred == y))
    diag = {
        "train_accuracy": train_acc,
        "train_ce": float(best_ce),
        "temperature": float(a),
        "ridge_lambda": lam,
        "scale_c": float(c),
        "s_norm": float(np.linalg.norm(s)),
        "s_max": float(smax),
        "bias": float(b0),
        "nonzero_token_weights": int(np.sum(s != 0)),
    }
    return {"weights": w, "diagnostics": diag}
