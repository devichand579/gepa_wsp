import numpy as np

def edit(w, f):
    # Copy weights to avoid mutating input directly (though we can edit in place)
    w = {k: v.copy() for k, v in w.items()}
    
    # Extract feedback data
    X = f["features"].astype(np.float64)          # (2000, 1568)
    y = f["y"].astype(np.int64)                   # (2000,)
    logits = f["logits"].astype(np.float64)       # (2000, 10)
    
    # Current predictions
    preds = np.argmax(logits, axis=1)
    mis = np.where(preds != y)[0]
    
    if len(mis) == 0:
        return {"weights": w, "diagnostics": {"status": "no misclassified samples"}}
    
    # We'll try to fix each misclassified sample one by one
    # (usually only one)
    for m in mis:
        c = int(y[m])          # correct class
        k = int(preds[m])      # wrong class
        # Required margin increase: we need z_c - z_k > 0
        # Current margin (wrong - correct) is positive
        M = logits[m, k] - logits[m, c]
        eps = 1e-3
        req = M + eps
        
        # --- Try bias-only update first ---
        beta = req / 2.0
        # Check if any sample not of class c would have its margin to c reduced below 0
        # For samples with y != c, margin to c is logits[i, y[i]] - logits[i, c]
        # We need this > beta (since z_c increases by beta)
        mask = (y != c)
        if np.any(mask):
            margins_to_c = logits[mask, y[mask]] - logits[mask, c]
            if np.min(margins_to_c) > beta:
                # Also check margins to k? For y != k, z_k decreases, so margin to k increases.
                # So only margin to c matters.
                w["head.bias"][c] += beta
                w["head.bias"][k] -= beta
                # Update logits for subsequent iterations (if any)
                logits[:, c] += beta
                logits[:, k] -= beta
                continue  # fixed this sample
        
        # --- Bias-only failed, try weight update ---
        # We want to add alpha * d to W[c] and subtract alpha * d from W[k]
        # such that for sample m, margin increases by 2 * alpha * (d @ X[m])
        X_m = X[m]  # (1568,)
        X_others = np.delete(X, m, axis=0)  # (1999, 1568)
        
        # Precompute X_others.T @ X_others
        XTX = X_others.T @ X_others  # (1568, 1568)
        
        # Try a range of regularization strengths
        lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
        fixed = False
        for lam in lambdas:
            # Solve (XTX + lam * I) d = X_m
            try:
                d = np.linalg.solve(XTX + lam * np.eye(XTX.shape[0]), X_m)
            except np.linalg.LinAlgError:
                continue
            
            dot = X @ d  # (2000,)
            if dot[m] <= 0:
                d = -d
                dot = -dot
            if dot[m] <= 0:
                continue
            
            alpha = req / (2.0 * dot[m])
            
            # Compute new logits
            new_logits = logits.copy()
            new_logits[:, c] += alpha * dot
            new_logits[:, k] -= alpha * dot
            
            # Check if any prediction flips
            new_preds = np.argmax(new_logits, axis=1)
            if np.all(new_preds == y):
                # Apply update
                w["head.weight"][c] += alpha * d
                w["head.weight"][k] -= alpha * d
                # Update logits for subsequent iterations
                logits = new_logits
                fixed = True
                break
        
        if not fixed:
            # Could not fix without flipping others; leave this sample as is
            pass
    
    # Ensure weights are float32 and within bounds
    for key in w:
        w[key] = w[key].astype(np.float32)
        w[key] = np.clip(w[key], -1000.0, 1000.0)
    
    # Diagnostics
    final_preds = np.argmax(logits, axis=1)
    acc = np.mean(final_preds == y)
    # Compute cross-entropy
    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    ce = -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12))
    
    diag = {
        "status": "updated",
        "initial_misclassified": mis.tolist(),
        "final_accuracy": float(acc),
        "final_cross_entropy": float(ce),
        "final_misclassified": np.where(final_preds != y)[0].tolist(),
        "max_abs_head_weight": float(np.max(np.abs(w["head.weight"]))),
        "max_abs_head_bias": float(np.max(np.abs(w["head.bias"]))),
    }
    
    return {"weights": w, "diagnostics": diag}
