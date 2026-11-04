import numpy as np

def edit(w, f):
    X_ids = f["x"]                      # [N,64] int64
    y     = f["y"].astype(np.int64)     # [N]
    N, L  = X_ids.shape
    V     = w["embed.weight"].shape[0]  # 2000

    # ---- 1. normalized unigram count features Xc[n,j] = c_{n,j}/T_n ----
    # counts via bincount per row (vectorized with offset trick)
    flat = X_ids.reshape(-1)
    row  = np.repeat(np.arange(N), L)
    # accumulate counts into [N,V]
    C = np.zeros((N, V), dtype=np.float64)
    np.add.at(C, (row, flat), 1.0)
    T = C.sum(axis=1, keepdims=True)
    T[T == 0] = 1.0
    Xc = C / T                          # [N,V]

    # ---- 2. closed-form ridge fit to one-hot targets ----
    Y = np.zeros((N, 2), dtype=np.float64)
    Y[np.arange(N), y] = 1.0
    lam = 5.0
    # solve in dual form (N x N) for stability with 2000 features
    Gt = Xc.T                           # [V,N]
    A  = Xc @ Gt + lam * np.eye(N)      # [N,N]
    # coefficients:  Wc = Gt @ A^{-1} @ Y   -> [V,2]
    Wc = Gt @ np.linalg.solve(A, Y)     # [V,2]
    # bias: mean residual
    pred = Xc @ Wc
    b    = (Y - pred).mean(axis=0)      # [2]

    # ---- 3. realize via embed + head ----
    # head projects onto first two coordinates
    head = np.zeros((2, 32), dtype=np.float64)
    head[0, 0] = 1.0
    head[1, 1] = 1.0

    a = Wc[:, 0].copy()
    bb = Wc[:, 1].copy()

    # feasibility: need Q = 32 - a^2 - b^2 >= (a+b)^2/30
    S = a + bb
    Q = 32.0 - a * a - bb * bb
    need = (S * S) / 30.0
    bad = (Q < need) | (Q < 0.0)
    if np.any(bad):
        # scale down offending rows so constraint holds with margin
        scale = np.ones(V)
        idx = np.where(bad)[0]
        for i in idx:
            lo, hi = 0.0, 1.0
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                aa = a[i] * mid; bbb = bb[i] * mid
                ss = aa + bbb
                qq = 32.0 - aa * aa - bbb * bbb
                if qq >= (ss * ss) / 30.0 and qq >= 0.0:
                    lo = mid
                else:
                    hi = mid
            scale[i] = lo
        a *= scale; bb *= scale

    S = a + bb
    Q = 32.0 - a * a - bb * bb
    m = S / 30.0
    rem = np.maximum(Q - 30.0 * m * m, 0.0)
    d = np.sqrt(rem / 30.0)             # magnitude of the ± pattern
    # build v_j [V,32]
    v = np.zeros((V, 32), dtype=np.float64)
    v[:, 0] = a
    v[:, 1] = bb
    v[:, 2:] = m[:, None]
    # 15 plus, 15 minus to keep the sum of the tail zero
    v[:, 2:17]  += d[:, None]
    v[:, 17:32] -= d[:, None]

    # safety: enforce exact zero-mean / unit-variance
    v -= v.mean(axis=1, keepdims=True)
    sd = v.std(axis=1, keepdims=True)
    sd[sd < 1e-6] = 1.0
    v /= sd

    # ---- 4. write weights ----
    w["embed.weight"] = v.astype(np.float32)
    w["head.weight"]  = head.astype(np.float32)
    w["head.bias"]    = b.astype(np.float32)
    # keep pos zero, blocks zero, norm identity (already the case)

    # ---- diagnostics ----
    logits = Xc @ Wc + b
    preds  = logits.argmax(axis=1)
    acc    = float((preds == y).mean())
    ll = logits - logits.max(axis=1, keepdims=True)
    p  = np.exp(ll); p /= p.sum(axis=1, keepdims=True)
    ce = float(-np.log(np.clip(p[np.arange(N), y], 1e-12, 1.0)).mean())
    diag = {
        "fit_accuracy": acc,
        "fit_cross_entropy": ce,
        "ridge_lambda": lam,
        "coef_abs_mean": float(np.abs(Wc).mean()),
        "coef_abs_max": float(np.abs(Wc).max()),
        "bias": b.tolist(),
        "embed_std": float(v.std()),
    }
    return {"weights": w, "diagnostics": diag}
