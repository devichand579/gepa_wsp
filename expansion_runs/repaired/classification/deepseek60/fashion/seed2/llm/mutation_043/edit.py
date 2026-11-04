import numpy as np

def edit(w, f):
    # Extract arrays
    X = f["features"].astype(np.float64)  # (N, 1568)
    y = f["y"].astype(np.int64)           # (N,)
    W = w["head.weight"].astype(np.float64)  # (10, 1568)
    b = w["head.bias"].astype(np.float64)    # (10,)
    
    N = X.shape[0]
    num_classes = W.shape[0]
    
    # Compute current logits
    L = X @ W.T + b  # (N, 10)
    preds = np.argmax(L, axis=1)
    wrong = np.where(preds != y)[0]
    n_wrong_before = len(wrong)
    
    # We'll update W and b
    W_new = W.copy()
    b_new = b.copy()
    
    # Process each misclassified sample (usually only one)
    max_iter = 10
    for _ in range(max_iter):
        L = X @ W_new.T + b_new
        preds = np.argmax(L, axis=1)
        wrong = np.where(preds != y)[0]
        if len(wrong) == 0:
            break
        i = wrong[0]
        true = y[i]
        pred = preds[i]
        f_i = X[i]
        norm2 = np.dot(f_i, f_i)
        if norm2 == 0:
            continue
        
        # Current margin for this sample
        margin_i = L[i, true] - L[i, pred]
        # Target margin
        target_margin = 1.0
        alpha_needed = (target_margin - margin_i) / (2.0 * norm2)
        
        # Compute bounds from other samples
        alpha_lo = max(0.0, alpha_needed)
        alpha_hi = np.inf
        
        # Precompute dot products of f_i with all X
        dots = X @ f_i  # (N,)
        
        # Check all j != i and all k != y[j]
        for j in range(N):
            if j == i:
                continue
            yj = y[j]
            Lj = L[j]
            for k in range(num_classes):
                if k == yj:
                    continue
                margin_jk = Lj[yj] - Lj[k]
                coeff = dots[j] * ( (1 if yj == true else 0) - (1 if yj == pred else 0) - (1 if k == true else 0) + (1 if k == pred else 0) )
                if coeff > 0:
                    alpha_lo = max(alpha_lo, -margin_jk / coeff)
                elif coeff < 0:
                    alpha_hi = min(alpha_hi, margin_jk / (-coeff))
        
        # Bounds from weight limits [-1000, 1000]
        mask = f_i > 0
        if np.any(mask):
            max_alpha_true = np.min((1000.0 - W_new[true, mask]) / f_i[mask])
            max_alpha_pred = np.min((W_new[pred, mask] + 1000.0) / f_i[mask])
            alpha_hi = min(alpha_hi, max_alpha_true, max_alpha_pred)
        
        # Choose alpha
        if alpha_hi > alpha_lo:
            alpha = (alpha_lo + alpha_hi) / 2.0
            alpha = min(alpha, alpha_hi * 0.99)
        else:
            # Try a smaller target margin
            target_margin = 0.1
            alpha_needed = (target_margin - margin_i) / (2.0 * norm2)
            alpha_lo = max(0.0, alpha_needed)
            if alpha_hi > alpha_lo:
                alpha = (alpha_lo + alpha_hi) / 2.0
            else:
                alpha = max(0.0, alpha_hi * 0.5) if alpha_hi > 0 else 0.0
        
        if alpha > 0:
            W_new[true] += alpha * f_i
            W_new[pred] -= alpha * f_i
        else:
            break
    
    # Clip to [-1000, 1000] just in case
    W_new = np.clip(W_new, -1000.0, 1000.0)
    b_new = np.clip(b_new, -1000.0, 1000.0)
    
    # Cast back to float32
    w["head.weight"] = W_new.astype(np.float32)
    w["head.bias"] = b_new.astype(np.float32)
    
    # Final diagnostics
    L_final = X @ W_new.T + b_new
    preds_final = np.argmax(L_final, axis=1)
    acc = np.mean(preds_final == y)
    n_wrong_after = int(np.sum(preds_final != y))
    
    diagnostics = {
        "n_wrong_before": int(n_wrong_before),
        "n_wrong_after": n_wrong_after,
        "accuracy_after": float(acc),
        "max_abs_head_weight": float(np.max(np.abs(W_new))),
        "max_abs_head_bias": float(np.max(np.abs(b_new))),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
