import numpy as np

def edit(w, f):
    X = np.asarray(f["features"], dtype=np.float64)   # (N, 1568)
    y = np.asarray(f["y"], dtype=np.int64)            # (N,)
    N = X.shape[0]

    W = np.asarray(w["head.weight"], dtype=np.float64).copy()   # (10, 1568)
    b = np.asarray(w["head.bias"], dtype=np.float64).copy()     # (10,)

    L = X @ W.T + b                                    # (N, 10)
    pred = np.argmax(L, axis=1)
    mis = np.where(pred != y)[0]

    diag = {"initial_misclassified": [int(i) for i in mis], "updates": []}

    for i in mis:
        xi = X[i]
        t = int(y[i])
        p = int(pred[i])
        g = X @ xi                                     # (N,) dot products with xi
        gi = g[i]
        if gi <= 0:
            continue

        # margins M[j,k] = L[j,y_j] - L[j,k]
        Ly = L[np.arange(N), y]
        M = Ly[:, None] - L                            # (N, 10)

        # coeff[j,k] = [yj==t] - [k==t] - [yj==p] + [k==p]
        yj_is_t = (y == t).astype(np.float64)
        yj_is_p = (y == p).astype(np.float64)
        k_is_t = np.zeros(10, dtype=np.float64); k_is_t[t] = 1.0
        k_is_p = np.zeros(10, dtype=np.float64); k_is_p[p] = 1.0
        coeff = yj_is_t[:, None] - k_is_t[None, :] - yj_is_p[:, None] + k_is_p[None, :]

        valid = np.ones((N, 10), dtype=bool)
        valid[np.arange(N), y] = False

        # Lower bound: from the misclassified sample i itself (coeff>0, M<0)
        alpha_lo = 0.0
        for k in range(10):
            if k == t:
                continue
            c = coeff[i, k]
            if c > 0 and M[i, k] < 0:
                alpha_lo = max(alpha_lo, -M[i, k] / (c * gi))

        # Upper bound: from all other samples j != i, where constraint currently holds
        alpha_hi = np.inf
        mask = valid & (coeff < 0) & (M > 0)
        mask[i, :] = False
        denom = (-coeff) * g[:, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            bounds = np.where(mask & (denom > 0), M / denom, np.inf)
        if np.any(np.isfinite(bounds)):
            alpha_hi = min(alpha_hi, float(np.min(bounds)))

        # Weight-limit bound (|w| <= 1000)
        xi_pos = xi > 0
        if np.any(xi_pos):
            b1 = (1000.0 - W[t][xi_pos]) / xi[xi_pos]
            b2 = (W[p][xi_pos] + 1000.0) / xi[xi_pos]
            alpha_hi = min(alpha_hi, float(np.min(b1)), float(np.min(b2)))

        if not np.isfinite(alpha_hi) or alpha_lo >= alpha_hi:
            continue

        # Pick alpha: comfortably above the required lower bound, safely below the upper bound
        alpha = alpha_lo * 1.5 + 1e-9
        alpha = min(alpha, alpha_hi * 0.9)
        if alpha <= 0:
            continue

        W[t] += alpha * xi
        W[p] -= alpha * xi
        L[:, t] += alpha * g
        L[:, p] -= alpha * g

        diag["updates"].append({
            "index": int(i), "true": t, "pred": p,
            "alpha_lo": float(alpha_lo), "alpha_hi": float(alpha_hi),
            "alpha_used": float(alpha),
        })

    W = np.clip(W, -1000.0, 1000.0)

    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

    Lf = X @ W.T + b
    pf = np.argmax(Lf, axis=1)
    diag["final_accuracy"] = float(np.mean(pf == y))
    diag["final_misclassified"] = [int(x) for x in np.where(pf != y)[0]]
    diag["max_abs_head_weight"] = float(np.max(np.abs(W)))
    diag["max_abs_head_bias"] = float(np.max(np.abs(b)))

    return {"weights": w, "diagnostics": diag}
