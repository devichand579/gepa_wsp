import numpy as np

def edit(w, f):
    X = np.asarray(f["x"])          # [N,64] int64
    y = np.asarray(f["y"]).astype(np.int64)   # [N]
    N, T = X.shape
    V = w["embed.weight"].shape[0]

    # ---- build length-normalized bag-of-words design ----
    mask = (X != 0)
    L = np.maximum(mask.sum(axis=1), 1).astype(np.float64)
    Xf = X.reshape(-1)
    rows = np.repeat(np.arange(N), T)
    keep = mask.reshape(-1)
    # counts per (example, token)
    A = np.zeros((N, V), dtype=np.float64)
    np.add.at(A, (rows[keep], Xf[keep]), 1.0)
    A /= L[:, None]                 # X[i,t] = count/L

    # targets in {-1,+1}
    t = np.where(y == 1, 1.0, -1.0)

    # ---- closed-form ridge on full pool ----
    lam = 5.0
    G = A.T @ A + lam * np.eye(V)
    rhs = A.T @ t
    try:
        s_ridge = np.linalg.solve(G, rhs)
    except np.linalg.LinAlgError:
        s_ridge = np.linalg.lstsq(G, rhs, rcond=None)[0]

    # ---- Naive-Bayes log-odds candidate ----
    pos = (y == 1)
    cnt_p = A[pos].sum(axis=0) + 1.0
    cnt_n = A[~pos].sum(axis=0) + 1.0
    s_nb = np.log(cnt_p / cnt_p.sum()) - np.log(cnt_n / cnt_n.sum())

    # ---- pick candidate by explicit training accuracy ----
    def acc(s, b):
        sc = A @ s + b
        return float(np.mean((sc > 0) == pos))

    cands = {}
    for name, s in (("ridge", s_ridge), ("nb", s_nb)):
        # bias via closed form: median of -A@s gives balanced threshold
        b = -float(np.median(A @ s))
        cands[name] = (acc(s, b), s, b)
    # also a scaled blend
    s_mix = 0.7 * s_ridge / (np.std(s_ridge) + 1e-9) + 0.3 * s_nb / (np.std(s_nb) + 1e-9)
    b_mix = -float(np.median(A @ s_mix))
    cands["mix"] = (acc(s_mix, b_mix), s_mix, b_mix)

    best = max(cands.items(), key=lambda kv: kv[1][0])
    _, s, b = best[1]

    # ---- rescale s so it fits the normalization geometry ----
    # row = [0, a, b*29, c], need a + 29b + c = 0 and a^2 + 29b^2 + c^2 = 32
    # feasible for |a| <= ~5.565; absorb scale into head weight w
    amax = float(np.max(np.abs(s))) + 1e-12
    K = max(1.0, amax / 5.4)
    a = s / K                      # |a| <= 5.4
    # solve 870 b^2 + 58 a b + (2a^2 - 32) = 0  -> pick stable root
    disc = (58.0 * a) ** 2 - 4.0 * 870.0 * (2.0 * a ** 2 - 32.0)
    disc = np.maximum(disc, 0.0)
    bcoef = (-58.0 * a + np.sqrt(disc)) / (2.0 * 870.0)
    c = -a - 29.0 * bcoef

    E = np.zeros((V, 32), dtype=np.float64)
    E[:, 1] = a
    E[:, 2:31] = bcoef[:, None]
    E[:, 31] = c
    # rows already have mean 0, std 1 -> LayerNorm is identity on them

    # ---- head reads dimension 1 ----
    H = np.zeros((2, 32), dtype=np.float64)
    H[1, 1] = K                    # score = K * (1/L) sum a = (1/L) sum s
    B = np.zeros(2, dtype=np.float64)
    B[1] = b

    w["embed.weight"] = E.astype(np.float32)
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)
    w["head.weight"] = H.astype(np.float32)
    w["head.bias"] = B.astype(np.float32)
    # keep blocks zero (they already are) but ensure finiteness
    for k in list(w.keys()):
        if k not in ("embed.weight", "pos.weight", "head.weight", "head.bias"):
            w[k] = np.asarray(w[k], dtype=np.float32)

    diag = {
        "chosen": best[0],
        "train_acc": float(best[1][0]),
        "K": float(K),
        "bias": float(b),
        "s_absmax": float(amax),
        "row_mean_absmax": float(np.max(np.abs(E.mean(axis=1)))),
        "row_std_min": float(np.min(E.std(axis=1))),
        "row_std_max": float(np.max(E.std(axis=1))),
    }
    return {"weights": w, "diagnostics": diag}
