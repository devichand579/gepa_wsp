import numpy as np

def edit(w, f):
    # Extract feedback arrays
    F = f["features"].astype(np.float64)          # (2000, 1568)
    y = f["y"].astype(np.int64)                   # (2000,)
    W = w["head.weight"].astype(np.float64)       # (10, 1568)
    b = w["head.bias"].astype(np.float64)         # (10,)

    # Current logits
    L = F @ W.T + b                               # (2000, 10)
    pred = np.argmax(L, axis=1)
    mis = np.where(pred != y)[0]

    if len(mis) == 0:
        return w

    # Fix the first misclassified sample (there is likely only one)
    i = mis[0]
    c = y[i]
    k = pred[i]
    x = F[i]
    norm2 = x @ x
    margin = L[i, c] - L[i, k]
    if margin >= 0:
        return w

    alpha_req = -margin / (2.0 * norm2)

    # Compute the maximum allowed alpha so that no currently correct sample flips
    x_dot = F @ x                                 # (2000,)
    max_alpha = np.inf
    for i2 in range(len(y)):
        yi = y[i2]
        for j in range(10):
            if j == yi:
                continue
            m = L[i2, yi] - L[i2, j]
            if m <= 0:
                continue  # already misclassified, ignore
            dot = x_dot[i2]
            if yi == c:
                if j == k:
                    delta = 2.0 * dot
                else:
                    delta = dot
            elif yi == k:
                if j == c:
                    delta = -2.0 * dot
                else:
                    delta = -dot
            else:
                if j == c:
                    delta = -dot
                elif j == k:
                    delta = dot
                else:
                    delta = 0.0
            if delta < 0:
                bound = m / (-delta)
                if bound < max_alpha:
                    max_alpha = bound

    # Choose alpha
    if max_alpha <= alpha_req:
        # Cannot fix without breaking others; take a safe step that doesn't hurt
        alpha = max_alpha * 0.99 if max_alpha > 0 else 0.0
    else:
        # Slightly overshoot the required alpha but stay safely below max_alpha
        alpha = min(alpha_req * 1.1, max_alpha * 0.99)
        if alpha <= alpha_req:
            alpha = (alpha_req + max_alpha) / 2.0

    # Apply rank-1 update
    W[c] += alpha * x
    W[k] -= alpha * x

    # Clip to allowed range
    W = np.clip(W, -1000.0, 1000.0)

    w["head.weight"] = W.astype(np.float32)
    # bias unchanged

    diagnostics = {
        "misclassified": mis.tolist(),
        "fixed_index": int(i),
        "true_class": int(c),
        "predicted_class": int(k),
        "alpha_req": float(alpha_req),
        "alpha_max": float(max_alpha),
        "alpha_used": float(alpha),
    }
    return {"weights": w, "diagnostics": diagnostics}
