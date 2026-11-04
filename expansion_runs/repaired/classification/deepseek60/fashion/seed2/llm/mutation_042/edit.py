import numpy as np

def edit(w, f):
    # Extract feedback arrays
    logits = f["logits"]          # (2000, 10)
    y = f["y"]                    # (2000,)
    
    # Current predictions
    preds = np.argmax(logits, axis=1)
    wrong = np.where(preds != y)[0]
    
    if len(wrong) == 0:
        return w  # already perfect
    
    # Fix the first misclassified sample (there is exactly one)
    i = wrong[0]
    t = int(y[i])
    p = int(preds[i])
    
    # Current margin for the misclassified sample
    m0 = logits[i, t] - logits[i, p]
    target_margin = 1.0
    delta_target = (target_margin - m0) / 2.0
    
    # Compute max delta allowed by all other samples
    N = logits.shape[0]
    # Margins for all j, k: logits[j, y_j] - logits[j, k]
    margins = logits[np.arange(N), y][:, None] - logits  # (N, 10)
    # Ignore k == y_j
    margins[np.arange(N), y] = np.inf
    
    # Coefficient of delta in margin change for each (j, k)
    coeff = np.zeros((N, 10))
    mask_yj_t = (y == t)
    mask_yj_p = (y == p)
    mask_other = ~(mask_yj_t | mask_yj_p)
    
    # yj == t
    coeff[mask_yj_t, :] = 1.0
    coeff[mask_yj_t, p] = 2.0
    # yj == p
    coeff[mask_yj_p, :] = -1.0
    coeff[mask_yj_p, t] = -2.0
    # other yj
    coeff[mask_other, t] = -1.0
    coeff[mask_other, p] = 1.0
    
    # Only j != i matter
    coeff[i, :] = 0.0
    margins[i, :] = np.inf
    
    # For coeff < 0, need delta <= margin / (-coeff)
    neg_mask = coeff < 0
    if np.any(neg_mask):
        bounds = margins[neg_mask] / (-coeff[neg_mask])
        max_delta = np.min(bounds)
    else:
        max_delta = np.inf
    
    # Bias bounds to keep within [-1000, 1000]
    b_t = w["head.bias"][t]
    b_p = w["head.bias"][p]
    max_delta = min(max_delta, 1000.0 - b_t, b_p + 1000.0)
    
    if max_delta <= 0:
        return w  # no feasible positive step
    
    # Choose a safe step
    delta = min(delta_target, max_delta * 0.99)
    if delta <= 0:
        delta = delta_target * 0.5
    
    # Apply bias update
    w["head.bias"][t] += delta
    w["head.bias"][p] -= delta
    
    # Clip and cast
    w["head.bias"] = np.clip(w["head.bias"], -1000.0, 1000.0).astype(np.float32)
    for key in w:
        if isinstance(w[key], np.ndarray):
            w[key] = np.clip(w[key], -1000.0, 1000.0).astype(np.float32)
    
    # Diagnostics
    new_logits = logits.copy()
    new_logits[:, t] += delta
    new_logits[:, p] -= delta
    new_preds = np.argmax(new_logits, axis=1)
    new_acc = float(np.mean(new_preds == y))
    new_wrong = int(np.sum(new_preds != y))
    
    diagnostics = {
        "fixed_index": int(i),
        "true_class": t,
        "pred_class": p,
        "margin_before": float(m0),
        "delta": float(delta),
        "margin_after": float(m0 + 2 * delta),
        "new_accuracy": new_acc,
        "n_wrong_after": new_wrong,
        "max_abs_bias": float(np.max(np.abs(w["head.bias"])))
    }
    
    return {"weights": w, "diagnostics": diagnostics}
