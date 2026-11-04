import numpy as np

def edit(w, f):
    # Extract head parameters
    W = w["head.weight"].copy()   # shape (10, 1568)
    b = w["head.bias"].copy()     # shape (10,)
    
    # Feedback data
    features = f["features"]      # (2000, 1568)
    y = f["y"]                    # (2000,)
    logits = f["logits"]          # (2000, 10)
    
    # Current predictions
    preds = np.argmax(logits, axis=1)
    mis = np.where(preds != y)[0]
    
    if len(mis) == 0:
        return w
    
    # Process each misclassified sample (usually just one)
    for i in mis:
        t = int(y[i])
        p = int(preds[i])
        x = features[i].astype(np.float64)  # (1568,)
        
        # Current margin for this sample
        m = float(logits[i, t] - logits[i, p])  # negative
        
        # Required alpha to fix this sample: margin increases by 2*alpha*(x·x)
        # because we add alpha*x to w[t] and subtract alpha*x from w[p].
        # logit_t increases by alpha*(x·x), logit_p decreases by alpha*(x·x)
        # so margin_t_p increases by 2*alpha*(x·x).
        xx = float(np.dot(x, x))
        if xx <= 0:
            continue
        alpha_lo = -m / (2.0 * xx)
        
        # Compute upper bound from all other samples
        alpha_hi = np.inf
        
        # Precompute x·x_j for all j
        dots = features @ x  # (2000,)
        
        # For each sample j, determine the maximum allowed alpha
        for j in range(len(y)):
            if j == i:
                continue
            yj = int(y[j])
            dot = float(dots[j])
            if dot <= 0:
                continue  # update increases margin, no constraint
            
            # Determine the tightest margin that could be reduced
            if yj == p:
                # logit_p decreases, so margin to all other classes decreases
                # margin_jk = logits[j, p] - logits[j, k] for k != p
                # We need alpha * dot < min_k margin_jk
                margins = logits[j, p] - logits[j]
                margins[p] = np.inf  # ignore self
                min_margin = float(np.min(margins))
                if min_margin <= 0:
                    alpha_hi = 0.0
                    break
                alpha_hi = min(alpha_hi, min_margin / dot)
            elif yj != t:
                # logit_t increases, so margin to class t decreases
                # margin_jt = logits[j, yj] - logits[j, t]
                margin_jt = float(logits[j, yj] - logits[j, t])
                if margin_jt <= 0:
                    alpha_hi = 0.0
                    break
                alpha_hi = min(alpha_hi, margin_jt / dot)
            # if yj == t, margin increases, no constraint
        
        # Also respect weight limits ±1000
        # We add alpha * x to W[t] and subtract alpha * x from W[p]
        # Since x >= 0, W[t] increases, W[p] decreases.
        # Need W[t] + alpha*x <= 1000 and W[p] - alpha*x >= -1000
        # => alpha <= (1000 - W[t]) / x  and alpha <= (W[p] + 1000) / x
        # Take elementwise minimum over all dimensions where x > 0.
        mask = x > 0
        if np.any(mask):
            lim_t = (1000.0 - W[t][mask]) / x[mask]
            lim_p = (W[p][mask] + 1000.0) / x[mask]
            alpha_hi = min(alpha_hi, float(np.min(lim_t)), float(np.min(lim_p)))
        
        # Choose alpha safely between bounds
        if alpha_hi <= alpha_lo:
            # Cannot fix without violating constraints; skip or use minimal
            alpha = alpha_lo * 1.01
        else:
            # Use a value comfortably above the lower bound but below the upper bound
            alpha = alpha_lo + 0.5 * (alpha_hi - alpha_lo)
            # Ensure we don't exceed the upper bound
            alpha = min(alpha, alpha_hi * 0.99)
        
        # Apply rank-1 update to weights
        W[t] += alpha * x
        W[p] -= alpha * x
        
        # Update logits for subsequent iterations (if multiple misclassified)
        # For simplicity, recompute logits from updated weights and bias
        # (only needed if more than one misclassified sample)
        # Since usually only one, we can just update the current logits array.
        # But to be safe, we can recompute for all samples:
        # logits = features @ W.T + b
        # However, we only need to update the affected logits for the next iteration.
        # Since we process sequentially, we can just recompute logits.
        # But note: f["logits"] is read-only, so we make a copy.
        # We'll recompute logits after each update to keep constraints accurate.
        # This is not an iterative optimizer; it's just applying closed-form updates.
        logits = features @ W.T + b
    
    # Clip to ensure within ±1000 (safety)
    W = np.clip(W, -1000.0, 1000.0)
    b = np.clip(b, -1000.0, 1000.0)
    
    # Write back
    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)
    
    # Diagnostics
    new_logits = features @ W.T + b
    new_preds = np.argmax(new_logits, axis=1)
    new_acc = float(np.mean(new_preds == y))
    new_mis = np.where(new_preds != y)[0].tolist()
    
    diagnostics = {
        "initial_misclassified": mis.tolist(),
        "final_accuracy": new_acc,
        "final_misclassified": new_mis,
        "max_abs_head_weight": float(np.max(np.abs(W))),
        "max_abs_head_bias": float(np.max(np.abs(b)))
    }
    
    return {"weights": w, "diagnostics": diagnostics}
