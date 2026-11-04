import numpy as np

def edit(w, f):
    # Extract current head parameters and feedback arrays
    W = w["head.weight"].astype(np.float64)  # shape (10, 1568)
    b = w["head.bias"].astype(np.float64)    # shape (10,)
    features = f["features"].astype(np.float64)  # (2000, 1568)
    y = f["y"]                               # (2000,)
    logits = f["logits"].astype(np.float64)  # (2000, 10)

    # Identify misclassified samples
    pred = np.argmax(logits, axis=1)
    errors = np.where(pred != y)[0]
    if len(errors) == 0:
        return w  # nothing to fix

    # Fix the first error (there should be exactly one)
    i = int(errors[0])
    t = int(y[i])
    p = int(pred[i])
    x = features[i]  # (1568,)
    norm2 = float(np.dot(x, x))
    if norm2 == 0.0:
        return w  # degenerate feature, cannot fix

    # Current margin for the misclassified sample
    margin_i = float(logits[i, t] - logits[i, p])
    # Required alpha to make margin positive: margin_i + 2*alpha*norm2 > 0
    alpha_lo = (-margin_i) / (2.0 * norm2) + 1e-6

    # Compute s = features @ x for all samples
    s = features @ x  # (2000,)

    # Compute margins and coefficients for all j != i, k != y[j]
    # margins[j, k] = logits[j, y[j]] - logits[j, k]
    margins = logits[np.arange(len(y)), y][:, None] - logits  # (2000, 10)
    # valid mask: k != y[j]
    valid = np.ones((len(y), 10), dtype=bool)
    valid[np.arange(len(y)), y] = False

    # coeff[j, k] = s[j] * ( (y[j]==t) - (y[j]==p) - (k==t) + (k==p) )
    y_t = (y == t).astype(np.float64)
    y_p = (y == p).astype(np.float64)
    k_t = (np.arange(10) == t).astype(np.float64)
    k_p = (np.arange(10) == p).astype(np.float64)
    coeff = s[:, None] * (y_t[:, None] - y_p[:, None] - k_t[None, :] + k_p[None, :])

    # alpha_hi from margin constraints: for coeff < 0, alpha < margin / (-coeff)
    allowed = np.full_like(margins, np.inf)
    mask = (coeff < 0) & valid
    allowed[mask] = margins[mask] / (-coeff[mask])
    allowed[i, :] = np.inf  # exclude the misclassified sample itself
    alpha_hi = float(np.min(allowed))

    # alpha_hi from weight bounds: |W_t + alpha*x| <= 1000, |W_p - alpha*x| <= 1000
    Wt = W[t]
    Wp = W[p]
    lo_w = -np.inf
    hi_w = np.inf
    for j in range(len(x)):
        xj = x[j]
        if xj > 0:
            lo_w = max(lo_w, (-1000.0 - Wt[j]) / xj)
            hi_w = min(hi_w, (1000.0 - Wt[j]) / xj)
            lo_w = max(lo_w, (Wp[j] - 1000.0) / xj)
            hi_w = min(hi_w, (1000.0 + Wp[j]) / xj)
        elif xj < 0:
            lo_w = max(lo_w, (1000.0 - Wt[j]) / xj)
            hi_w = min(hi_w, (-1000.0 - Wt[j]) / xj)
            lo_w = max(lo_w, (1000.0 + Wp[j]) / xj)
            hi_w = min(hi_w, (Wp[j] - 1000.0) / xj)
    alpha_hi = min(alpha_hi, hi_w)

    if alpha_hi <= alpha_lo:
        # Cannot fix without violating constraints; return unchanged
        return w

    # Choose a safe step size in the middle of the feasible interval
    alpha = alpha_lo + 0.5 * (alpha_hi - alpha_lo)

    # Apply the rank-1 update
    W[t] += alpha * x
    W[p] -= alpha * x

    # Clip to ±1000 and cast back to float32
    W = np.clip(W, -1000.0, 1000.0).astype(np.float32)
    w["head.weight"] = W

    # Diagnostics
    diagnostics = {
        "fixed_index": i,
        "true_class": t,
        "pred_class": p,
        "alpha": float(alpha),
        "alpha_lo": float(alpha_lo),
        "alpha_hi": float(alpha_hi),
        "norm2": float(norm2),
        "margin_before": float(margin_i),
        "margin_after": float(margin_i + 2.0 * alpha * norm2),
        "max_abs_head_weight": float(np.max(np.abs(W))),
    }
    return {"weights": w, "diagnostics": diagnostics}
