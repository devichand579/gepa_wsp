import numpy as np

def edit(w, f):
    # Extract head weights and features
    W = w["head.weight"].copy()  # (10, 1568)
    b = w["head.bias"].copy()    # (10,)
    H = f["features"]            # (2000, 1568)
    y = f["y"]                   # (2000,)
    logits = f["logits"]         # (2000, 10)

    pred = np.argmax(logits, axis=1)
    wrong = np.where(pred != y)[0]

    diagnostics = {
        "n_wrong_before": int(len(wrong)),
        "max_abs_W_before": float(np.max(np.abs(W))),
    }

    if len(wrong) == 0:
        diagnostics["message"] = "No misclassified samples; returning unchanged."
        return {"weights": w, "diagnostics": diagnostics}

    # Pick the misclassified sample with the most negative margin
    margins = logits[np.arange(len(y)), y] - logits[np.arange(len(y)), pred]
    i = wrong[np.argmin(margins[wrong])]
    t = int(y[i])
    p = int(pred[i])

    h = H[i].astype(np.float64)
    norm2 = float(np.dot(h, h))
    m0 = float(logits[i, t] - logits[i, p])

    alpha_lo = -np.inf
    alpha_hi = np.inf

    # Precompute dot products of h with all H
    dot = H @ h  # (2000,)

    # Build alpha bounds from all margin constraints
    for j in range(len(y)):
        yj = int(y[j])
        m_j = logits[j, yj]
        for k in range(10):
            if k == yj:
                continue
            m_jk = float(m_j - logits[j, k])
            s = (1 if yj == t else 0) - (1 if yj == p else 0) - (1 if k == t else 0) + (1 if k == p else 0)
            coeff = s * float(dot[j])
            if coeff > 0:
                alpha_lo = max(alpha_lo, -m_jk / coeff)
            elif coeff < 0:
                alpha_hi = min(alpha_hi, m_jk / (-coeff))

    # Weight bounds to keep within [-1000, 1000]
    for idx in range(len(h)):
        hk = float(h[idx])
        if hk > 0:
            alpha_hi = min(alpha_hi, (1000.0 - W[t, idx]) / hk, (W[p, idx] + 1000.0) / hk)
            alpha_lo = max(alpha_lo, (-1000.0 - W[t, idx]) / hk, (W[p, idx] - 1000.0) / hk)
        elif hk < 0:
            alpha_hi = min(alpha_hi, (-1000.0 - W[t, idx]) / hk, (W[p, idx] - 1000.0) / hk)
            alpha_lo = max(alpha_lo, (1000.0 - W[t, idx]) / hk, (W[p, idx] + 1000.0) / hk)

    if alpha_lo < alpha_hi:
        alpha = (alpha_lo + alpha_hi) / 2.0
    else:
        alpha = alpha_lo + 1e-4 if np.isfinite(alpha_lo) else 0.0

    # Apply rank-1 update
    W[t] += alpha * h
    W[p] -= alpha * h

    # Clip to be safe
    W = np.clip(W, -1000.0, 1000.0)

    # Compute new margin for the fixed sample
    new_logit_t = float(W[t] @ H[i] + b[t])
    new_logit_p = float(W[p] @ H[i] + b[p])
    margin_after = new_logit_t - new_logit_p

    w["head.weight"] = W.astype(np.float32)

    diagnostics.update({
        "fixed_index": int(i),
        "true_class": t,
        "pred_class": p,
        "alpha": float(alpha),
        "alpha_lo": float(alpha_lo),
        "alpha_hi": float(alpha_hi),
        "norm2": norm2,
        "margin_before": m0,
        "margin_after": margin_after,
        "max_abs_W_after": float(np.max(np.abs(W))),
    })

    return {"weights": w, "diagnostics": diagnostics}
