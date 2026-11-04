import numpy as np

def edit(w, f):
    # Extract arrays
    F = f["features"].astype(np.float64)          # (N, D)
    y = f["y"]                                    # (N,)
    logits = f["logits"].astype(np.float64).copy()  # (N, 10)
    N, D = F.shape
    num_classes = logits.shape[1]
    
    # Current head weights and bias
    W = w["head.weight"].astype(np.float64).copy()  # (10, D)
    b = w["head.bias"].astype(np.float64).copy()    # (10,)
    
    diagnostics = {}
    preds = np.argmax(logits, axis=1)
    mis = np.where(preds != y)[0]
    diagnostics["initial_misclassified"] = mis.tolist()
    diagnostics["initial_accuracy"] = float(np.mean(preds == y))
    
    max_iter = 10
    for it in range(max_iter):
        preds = np.argmax(logits, axis=1)
        mis = np.where(preds != y)[0]
        if len(mis) == 0:
            break
        i = mis[0]                     # take first misclassified
        true = y[i]
        pred = preds[i]
        margin_i = logits[i, true] - logits[i, pred]
        f_i = F[i]
        norm_sq = np.dot(f_i, f_i)
        if norm_sq == 0:
            continue
        alpha_req = -margin_i / (2.0 * norm_sq)
        
        # Compute alpha_max from constraints of all other samples
        alpha_max = np.inf
        dot_vals = F @ f_i             # (N,)
        for j in range(N):
            if j == i:
                continue
            yj = y[j]
            margins = logits[j, yj] - logits[j, :]   # (10,)
            for c in range(num_classes):
                if c == yj:
                    continue
                margin = margins[c]
                if margin <= 0:
                    continue
                s = (1 if yj == true else 0) - (1 if c == true else 0) \
                    - (1 if yj == pred else 0) + (1 if c == pred else 0)
                delta = s * dot_vals[j]
                if delta < 0:
                    candidate = margin / (-delta)
                    if candidate < alpha_max:
                        alpha_max = candidate
        
        # Also respect the ±1000 weight bound
        # For true class: W[true] + alpha * f_i <= 1000 and >= -1000
        # For pred class: W[pred] - alpha * f_i <= 1000 and >= -1000
        # We only need upper bounds on alpha from these.
        for d in range(D):
            if f_i[d] > 0:
                # true class upper bound
                bound = (1000.0 - W[true, d]) / f_i[d]
                if bound < alpha_max:
                    alpha_max = bound
                # pred class lower bound? Actually for pred: W[pred] - alpha*f_i >= -1000 => alpha <= (W[pred] + 1000)/f_i
                bound = (W[pred, d] + 1000.0) / f_i[d]
                if bound < alpha_max:
                    alpha_max = bound
            elif f_i[d] < 0:
                # true class upper bound: W[true] + alpha*f_i <= 1000 => alpha >= (1000 - W[true])/f_i (negative, so no upper bound)
                # but also W[true] + alpha*f_i >= -1000 => alpha <= (-1000 - W[true])/f_i (positive)
                bound = (-1000.0 - W[true, d]) / f_i[d]
                if bound < alpha_max:
                    alpha_max = bound
                # pred class: W[pred] - alpha*f_i <= 1000 => alpha >= (W[pred] - 1000)/f_i (negative, no upper)
                # W[pred] - alpha*f_i >= -1000 => alpha <= (W[pred] + 1000)/f_i (positive)
                bound = (W[pred, d] + 1000.0) / f_i[d]
                if bound < alpha_max:
                    alpha_max = bound
        
        # Choose alpha
        if np.isfinite(alpha_max) and alpha_max > 0:
            alpha = min(alpha_req * 1.1, alpha_max * 0.99)
        else:
            alpha = alpha_req * 1.1
        if alpha <= 0:
            alpha = alpha_req * 1.1
        
        # Update weights
        W[true] += alpha * f_i
        W[pred] -= alpha * f_i
        
        # Update logits for all samples
        logits[:, true] += alpha * dot_vals
        logits[:, pred] -= alpha * dot_vals
        
        diagnostics.setdefault("updates", []).append({
            "index": int(i),
            "true": int(true),
            "pred": int(pred),
            "alpha_req": float(alpha_req),
            "alpha_max": float(alpha_max) if np.isfinite(alpha_max) else None,
            "alpha_used": float(alpha)
        })
    
    # Clip weights to [-1000, 1000] just in case
    W = np.clip(W, -1000.0, 1000.0)
    b = np.clip(b, -1000.0, 1000.0)
    
    # Write back
    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)
    
    # Final diagnostics
    preds_final = np.argmax(logits, axis=1)
    mis_final = np.where(preds_final != y)[0]
    diagnostics["final_misclassified"] = mis_final.tolist()
    diagnostics["final_accuracy"] = float(np.mean(preds_final == y))
    diagnostics["max_abs_head_weight"] = float(np.max(np.abs(W)))
    diagnostics["max_abs_head_bias"] = float(np.max(np.abs(b)))
    
    return {"weights": w, "diagnostics": diagnostics}
