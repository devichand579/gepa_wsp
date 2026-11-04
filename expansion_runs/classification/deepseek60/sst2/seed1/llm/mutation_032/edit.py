import numpy as np

def edit(w, f):
    X = f["x"].astype(np.int64)          # [N,64]
    y = f["y"].astype(np.int64)          # [N]
    N, T = X.shape
    V = w["embed.weight"].shape[0]       # 2000
    D = w["embed.weight"].shape[1]       # 32

    # ---- build length-normalized bag-of-words ----
    mask = (X != 0).astype(np.float64)   # [N,64]
    L = mask.sum(axis=1)                 # [N]
    L[L == 0] = 1.0
    Z = np.zeros((N, V), dtype=np.float64)
    for t in range(T):
        ids = X[:, t]
        m = mask[:, t]
        np.add.at(Z, (np.arange(N), ids), m)
    Z = Z / L[:, None]                   # length-normalized counts

    # ---- ridge-regularized least-squares fit of s, b ----
    # target +1 / -1
    tgt = np.where(y == 1, 1.0, -1.0)
    # center features to fit intercept cleanly
    mu = Z.mean(axis=0)
    Zc = Z - mu
    lam = 0.5
    A = Zc.T @ Zc + lam * np.eye(V)
    rhs = Zc.T @ tgt
    s = np.linalg.solve(A, rhs)          # [V]
    b = tgt.mean() - mu @ s              # intercept

    # ---- calibrate temperature for low cross-entropy ----
    # logit = scale * (s . z + b); choose scale by 1-D search on CE
    raw = Z @ s + b                      # [N]
    best_scale, best_ce = 1.0, 1e9
    for sc in np.linspace(0.2, 6.0, 59):
        lg = sc * raw
        p1 = 1.0 / (1.0 + np.exp(-lg))
        p1 = np.clip(p1, 1e-7, 1 - 1e-7)
        ce = -np.mean(y * np.log(p1) + (1 - y) * np.log(1 - p1))
        if ce < best_ce:
            best_ce, best_scale = ce, sc
    scale = best_scale
    s = s * scale
    b = b * scale

    # ---- realize s_v = head[1] . LN(embed[v]) ----
    # choose head[1] = c * e_0 ; then f_v[0] = s_v / c must satisfy |f_v[0]| <= sqrt(31)
    smax = np.max(np.abs(s)) + 1e-12
    c = smax / (np.sqrt(31.0) * 0.98)    # margin
    head1 = np.zeros(D, dtype=np.float64)
    head1[0] = c

    alpha = s / c                        # f_v[0]
    # solve 31*beta^2 + 2*alpha*beta + 31*alpha^2 - 960 = 0 for beta
    disc = 119040.0 - 3840.0 * alpha ** 2
    disc = np.maximum(disc, 0.0)
    beta = (-2.0 * alpha + np.sqrt(disc)) / 62.0
    gamma = -(alpha + beta) / 30.0

    F = np.empty((V, D), dtype=np.float64)
    F[:, 0] = alpha
    F[:, 1] = beta
    F[:, 2:] = gamma[:, None]

    # verify mean 0, var 1
    # (construction guarantees it)

    # ---- write weights ----
    embed = F.astype(np.float32)
    w["embed.weight"] = embed

    # zero out everything else that is degenerate (keep as-is, they are zero)
    for k in list(w.keys()):
        if k in ("embed.weight", "head.weight", "head.bias"):
            continue
        arr = w[k]
        if k == "norm.weight":
            w[k] = np.ones_like(arr, dtype=np.float32)
        elif k == "norm.bias":
            w[k] = np.zeros_like(arr, dtype=np.float32)
        else:
            w[k] = np.zeros_like(arr, dtype=np.float32)

    head = np.zeros((2, D), dtype=np.float64)
    head[1] = head1
    w["head.weight"] = head.astype(np.float32)
    w["head.bias"] = np.array([0.0, b], dtype=np.float32)

    # clamp to finite, |.|<=1000
    for k in w:
        w[k] = np.clip(np.nan_to_num(w[k].astype(np.float32), nan=0.0,
                                     posinf=0.0, neginf=0.0), -1000.0, 1000.0)

    diag = {
        "ridge_lambda": lam,
        "scale": float(scale),
        "c": float(c),
        "s_norm": float(np.linalg.norm(s)),
        "s_max": float(np.max(np.abs(s))),
        "bias": float(b),
        "train_ce": float(best_ce),
        "train_acc": float(np.mean((raw * scale > 0).astype(np.int64) == y)),
    }
    return {"weights": w, "diagnostics": diag}
