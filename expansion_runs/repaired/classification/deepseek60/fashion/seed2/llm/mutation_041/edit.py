import numpy as np

def edit(w, f):
    # Extract feedback arrays
    H = f["features"].astype(np.float64)          # (N, D)
    y = f["y"].astype(np.int64)                   # (N,)
    N, D = H.shape

    # Current head parameters
    W = w["head.weight"].astype(np.float64).copy()  # (10, D)
    b = w["head.bias"].astype(np.float64).copy()    # (10,)

    # Current logits
    L = H @ W.T + b                                 # (N, 10)

    target_margin = 1.0
    max_iter = 10

    for _ in range(max_iter):
        pred = np.argmax(L, axis=1)
        wrong = np.where(pred != y)[0]
        if len(wrong) == 0:
            break

        i = wrong[0]
        t = y[i]
        p = pred[i]
        h_i = H[i]                                  # (D,)
        s = H @ h_i                                 # (N,)

        # Margin matrix M[j,c] = L[j, y[j]] - L[j, c]
        M = L[np.arange(N), y][:, None] - L         # (N, 10)

        # Coefficient C[j,c] = (y[j]==t) - (y[j]==p) - (c==t) + (c==p)
        Yt = (y == t).astype(np.float64)
        Yp = (y == p).astype(np.float64)
        C = Yt[:, None] - Yp[:, None]               # (N, 1)
        C = C - (np.arange(10) == t)[None, :] + (np.arange(10) == p)[None, :]

        A = s[:, None] * C                          # (N, 10)
        R = target_margin - M                       # (N, 10)

        # Ignore c == y[j] (margin always 0 there)
        valid = (np.arange(10)[None, :] != y[:, None])

        alpha_lo = -np.inf
        alpha_hi = np.inf

        # Margin constraints: M + alpha * A >= target  =>  alpha * A >= R
        pos = valid & (A > 1e-12)
        neg = valid & (A < -1e-12)

        if np.any(pos):
            alpha_lo = max(alpha_lo, np.max(R[pos] / A[pos]))
        if np.any(neg):
            alpha_hi = min(alpha_hi, np.min(R[neg] / A[neg]))

        # Weight bounds for row t: W[t] + alpha * h_i in [-1000, 1000]
        Wt = W[t]
        Wp = W[p]
        h = h_i

        pos_h = h > 1e-12
        neg_h = h < -1e-12

        if np.any(pos_h):
            lo = (-1000.0 - Wt[pos_h]) / h[pos_h]
            hi = (1000.0 - Wt[pos_h]) / h[pos_h]
            alpha_lo = max(alpha_lo, np.max(lo))
            alpha_hi = min(alpha_hi, np.min(hi))
        if np.any(neg_h):
            hi = (-1000.0 - Wt[neg_h]) / h[neg_h]
            lo = (1000.0 - Wt[neg_h]) / h[neg_h]
            alpha_lo = max(alpha_lo, np.max(lo))
            alpha_hi = min(alpha_hi, np.min(hi))

        # Weight bounds for row p: W[p] - alpha * h_i in [-1000, 1000]
        if np.any(pos_h):
            lo = (Wp[pos_h] - 1000.0) / h[pos_h]
            hi = (Wp[pos_h] + 1000.0) / h[pos_h]
            alpha_lo = max(alpha_lo, np.max(lo))
            alpha_hi = min(alpha_hi, np.min(hi))
        if np.any(neg_h):
            hi = (Wp[neg_h] - 1000.0) / h[neg_h]
            lo = (Wp[neg_h] + 1000.0) / h[neg_h]
            alpha_lo = max(alpha_lo, np.max(lo))
            alpha_hi = min(alpha_hi, np.min(hi))

        # If infeasible, take the largest feasible step (may not fully fix)
        if alpha_lo > alpha_hi:
            alpha = alpha_hi if alpha_hi > 0 else alpha_lo
        else:
            # Pick a step inside the feasible interval, biased toward more margin
            alpha = alpha_lo + 0.5 * (alpha_hi - alpha_lo)

        # Apply rank-1 update
        W[t] += alpha * h_i
        W[p] -= alpha * h_i

        # Recompute logits for next iteration
        L = H @ W.T + b

    # Final safety clip (should already be within bounds)
    W = np.clip(W, -1000.0, 1000.0)

    # Write back as float32
    w["head.weight"] = W.astype(np.float32)

    # Diagnostics
    pred_final = np.argmax(L, axis=1)
    acc = float(np.mean(pred_final == y))
    diag = {
        "accuracy_after": acc,
        "n_wrong_after": int(np.sum(pred_final != y)),
        "max_abs_head_weight": float(np.max(np.abs(W))),
        "min_abs_head_weight": float(np.min(np.abs(W))),
    }
    return {"weights": w, "diagnostics": diag}
