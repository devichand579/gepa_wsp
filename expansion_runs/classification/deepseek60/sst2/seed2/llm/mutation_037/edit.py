import numpy as np

def edit(w, f):
    X = np.asarray(f["x"])            # [N,64] int64
    y = np.asarray(f["y"]).astype(np.float64)   # [N]
    N, T = X.shape
    V = w["embed.weight"].shape[0]    # 2000
    D = w["embed.weight"].shape[1]    # 32

    # ---- normalized count features p [N,V] (exclude pad id 0) ----
    mask = (X != 0)
    L = mask.sum(axis=1).astype(np.float64)
    L[L == 0] = 1.0
    P = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), T)
    cols = X.reshape(-1)
    valid = mask.reshape(-1)
    np.add.at(P, (rows[valid], cols[valid]), 1.0)
    P /= L[:, None]

    # ---- closed-form ridge linear classifier on p ----
    yc = y - y.mean()
    # solve (P^T P + lam I) w = P^T yc  via normal equations (V=2000, N=4000)
    PtP = P.T @ P
    Pty = P.T @ yc
    lam = 1.0
    A = PtP + lam * np.eye(V)
    wvec = np.linalg.solve(A, Pty)          # [V]
    b = y.mean() - P.mean(axis=0) @ wvec    # intercept

    # ---- realize wvec in the model ----
    # We want d @ u_v = w_v with d = c * e_1  =>  u_v[0] = w_v / c
    # u_v must be zero-mean, unit-variance (norm^2 = D).
    # Choose c so that max|w_v/c| <= sqrt(D-1).
    maxabs = np.max(np.abs(wvec))
    if maxabs < 1e-12:
        maxabs = 1e-12
    c = maxabs / np.sqrt(D - 1.0) * 1.0001
    a = wvec / c                            # [V], |a| <= sqrt(D-1)

    # t: fixed zero-sum unit vector in R^{D-1}
    t = np.zeros(D - 1)
    t[0] = 1.0 / np.sqrt(2.0)
    t[1] = -1.0 / np.sqrt(2.0)

    # u_v[0] = a_v ; u_v[1:] = -a_v/(D-1) + b_v * t
    # zero mean: sum_{k>=1} = -a_v  (since t sums to 0)
    # unit variance: sum_{k>=1} u^2 = a_v^2/(D-1) + b_v^2 = D - a_v^2
    b_v = np.sqrt(np.maximum(D - a_v**2 * (1.0 + 1.0 / (D - 1.0)), 0.0))
    U = np.zeros((V, D), dtype=np.float64)
    U[:, 0] = a
    U[:, 1:] = -a[:, None] / (D - 1.0) + b_v[:, None] * t[None, :]

    # embed[v] = U[v] (already zero-mean unit-variance => LN is identity)
    w["embed.weight"] = U.astype(np.float32)

    # head: d = head[0]-head[1] = c * e_1 ; put all on row 0, zero row 1
    H = np.zeros((2, D), dtype=np.float64)
    H[0, 0] = c
    H[1, 0] = 0.0
    w["head.weight"] = H.astype(np.float32)

    # bias: logit_diff = d@pooled + (bias0-bias1) = w·p + (bias0-bias1)
    # we want w·p + b  =>  bias0 - bias1 = b
    hb = np.zeros(2, dtype=np.float64)
    hb[0] = b / 2.0
    hb[1] = -b / 2.0
    w["head.bias"] = hb.astype(np.float32)

    # keep blocks identity / pos zero / norms at (1,0)
    for k in list(w.keys()):
        if k in ("embed.weight", "head.weight", "head.bias"):
            continue
        arr = np.asarray(w[k])
        if k.endswith(".weight") and ("norm" in k):
            w[k] = np.ones_like(arr, dtype=np.float32)
        elif k.endswith(".bias") and ("norm" in k):
            w[k] = np.zeros_like(arr, dtype=np.float32)
        else:
            w[k] = np.zeros_like(arr, dtype=np.float32)

    # ---- diagnostics: closed-form training accuracy ----
    logit_diff = P @ wvec + b
    pred = (logit_diff > 0).astype(np.int64)
    acc = float((pred == y.astype(np.int64)).mean())
    diag = {
        "train_acc_linear_bow": acc,
        "lam": lam,
        "c": float(c),
        "maxabs_w": float(maxabs),
        "embed_absmax": float(np.max(np.abs(U))),
        "head_absmax": float(np.max(np.abs(H))),
    }
    return {"weights": w, "diagnostics": diag}
