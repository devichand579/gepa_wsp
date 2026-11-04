import numpy as np

def edit(w, f):
    W = w["head.weight"]          # (10, 1568)
    b = w["head.bias"]            # (10,)
    X = f["features"]             # (2000, 1568)
    y = f["y"]                    # (2000,)
    logits = f["logits"]          # (2000, 10)

    pred = np.argmax(logits, axis=1)
    mis = np.where(pred != y)[0]
    if len(mis) == 0:
        return w

    # Fix the first misclassified sample with a rank-1 update.
    i = mis[0]
    x = X[i].astype(np.float64)   # (1568,)
    true = int(y[i])
    wrong = int(pred[i])

    m = float(logits[i, true] - logits[i, wrong])
    norm2 = float(np.dot(x, x))
    if norm2 == 0:
        return w

    # Lower bound for alpha to achieve a small positive margin.
    margin_target = 0.1
    alpha_lo = (margin_target - m) / (2.0 * norm2)
    if alpha_lo < 0:
        alpha_lo = 0.0

    # Upper bound from other samples: keep all other margins positive.
    N = X.shape[0]
    logits_comp = logits.copy()
    logits_comp[np.arange(N), y] = -np.inf
    best_comp = np.max(logits_comp, axis=1)
    margins = logits[np.arange(N), y] - best_comp
    best_comp_idx = np.argmax(logits_comp, axis=1)

    dot_products = X @ x          # (N,)
    C = ((y == true).astype(np.float64) - (y == wrong).astype(np.float64)) - \
        ((best_comp_idx == true).astype(np.float64) - (best_comp_idx == wrong).astype(np.float64))

    alpha_hi = np.inf
    for j in range(N):
        if j == i:
            continue
        coeff = dot_products[j] * C[j]
        if coeff < 0:
            bound = -margins[j] / coeff
            if bound < alpha_hi:
                alpha_hi = bound

    # Upper bound from weight limits (±1000).
    alpha_hi_weight = np.inf
    for d in range(len(x)):
        if x[d] > 0:
            bound1 = (1000.0 - W[true, d]) / x[d]
            bound2 = (W[wrong, d] + 1000.0) / x[d]
            if bound1 > 0 and bound1 < alpha_hi_weight: alpha_hi_weight = bound1
            if bound2 > 0 and bound2 < alpha_hi_weight: alpha_hi_weight = bound2
        elif x[d] < 0:
            bound1 = (-1000.0 - W[true, d]) / x[d]
            bound2 = (W[wrong, d] - 1000.0) / x[d]
            if bound1 > 0 and bound1 < alpha_hi_weight: alpha_hi_weight = bound1
            if bound2 > 0 and bound2 < alpha_hi_weight: alpha_hi_weight = bound2
    alpha_hi = min(alpha_hi, alpha_hi_weight)

    # Choose a safe alpha.
    if alpha_lo >= alpha_hi:
        alpha = alpha_hi * 0.99
    else:
        alpha = alpha_lo * 1.1
        if alpha > alpha_hi:
            alpha = (alpha_lo + alpha_hi) / 2.0

    # Apply the rank-1 update.
    W[true] += alpha * x
    W[wrong] -= alpha * x

    diag = {
        "misclassified_index": int(i),
        "true_class": true,
        "pred_class": wrong,
        "alpha": float(alpha),
        "alpha_lo": float(alpha_lo),
        "alpha_hi": float(alpha_hi),
        "norm2": float(norm2),
        "margin_before": float(m),
        "margin_after": float(m + 2.0 * alpha * norm2),
        "max_abs_head_weight": float(np.max(np.abs(W)))
    }
    return {"weights": w, "diagnostics": diag}
