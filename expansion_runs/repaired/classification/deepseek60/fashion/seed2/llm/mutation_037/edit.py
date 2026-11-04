import numpy as np

def edit(w, f):
    # Extract arrays
    features = f["features"]  # (2000, 1568)
    logits = f["logits"]      # (2000, 10)
    y = f["y"]                # (2000,)
    probs = f["probs"]        # (2000, 10)
    
    # Find misclassified samples
    preds = np.argmax(probs, axis=1)
    misclassified = np.where(preds != y)[0]
    
    diagnostics = {
        "initial_misclassified": misclassified.tolist(),
        "num_misclassified": int(len(misclassified))
    }
    
    if len(misclassified) == 0:
        diagnostics["message"] = "No misclassified samples, no update."
        return {"weights": w, "diagnostics": diagnostics}
    
    # Fix each misclassified sample sequentially with a closed-form rank-1 update.
    # (In practice there is only one, but this handles multiple without iterative optimization.)
    for i in misclassified:
        y_true = int(y[i])
        y_pred = int(preds[i])
        x_i = features[i]
        
        # Compute dot products for all samples
        d = features @ x_i  # (2000,)
        
        # Compute feasible interval for alpha from all margin constraints
        L_alpha = -np.inf
        U_alpha = np.inf
        
        for j in range(len(y)):
            yj = int(y[j])
            for c in range(10):
                if c == yj:
                    continue
                # K = (1_{yj==y_true} - 1_{yj==y_pred}) - (1_{c==y_true} - 1_{c==y_pred})
                K = (1 if yj == y_true else 0) - (1 if yj == y_pred else 0) \
                    - (1 if c == y_true else 0) + (1 if c == y_pred else 0)
                if K == 0:
                    continue
                d_j = d[j]
                if d_j == 0:
                    continue
                A = d_j * K
                RHS = logits[j, c] - logits[j, yj]
                if A > 0:
                    L_alpha = max(L_alpha, RHS / A)
                else:
                    U_alpha = min(U_alpha, RHS / A)
        
        # Choose alpha
        if L_alpha >= U_alpha:
            # Fallback: just make the misclassified sample correct
            norm2 = float(np.dot(x_i, x_i))
            if norm2 > 0:
                alpha_min = (logits[i, y_pred] - logits[i, y_true]) / (2 * norm2)
                alpha = alpha_min + 0.1
            else:
                alpha = 0.0
        else:
            alpha = (L_alpha + U_alpha) / 2.0
            if alpha <= 0:
                alpha = L_alpha + 0.01
        
        # Enforce weight bounds (±1000)
        max_alpha_weight = np.inf
        w_y = w["head.weight"][y_true]
        w_p = w["head.weight"][y_pred]
        for k in range(len(x_i)):
            if x_i[k] > 0:
                ub1 = (1000.0 - w_y[k]) / x_i[k]
                ub2 = (w_p[k] + 1000.0) / x_i[k]
                max_alpha_weight = min(max_alpha_weight, ub1, ub2)
        if max_alpha_weight < alpha:
            alpha = max_alpha_weight * 0.99
            if alpha <= 0:
                alpha = 0.0
        
        # Apply rank-1 update
        w["head.weight"][y_true] += alpha * x_i
        w["head.weight"][y_pred] -= alpha * x_i
        
        # Clip to ensure bounds
        w["head.weight"] = np.clip(w["head.weight"], -1000.0, 1000.0).astype(np.float32)
        
        diagnostics[f"fixed_{i}"] = {
            "y_true": y_true,
            "y_pred": y_pred,
            "alpha": float(alpha),
            "L_alpha": float(L_alpha) if L_alpha != -np.inf else None,
            "U_alpha": float(U_alpha) if U_alpha != np.inf else None,
            "norm2": float(np.dot(x_i, x_i)),
            "margin_before": float(logits[i, y_true] - logits[i, y_pred]),
            "margin_after": float((logits[i, y_true] + alpha * np.dot(x_i, x_i)) -
                                  (logits[i, y_pred] - alpha * np.dot(x_i, x_i)))
        }
    
    # Recompute final predictions using updated head
    new_logits = features @ w["head.weight"].T + w["head.bias"]
    new_preds = np.argmax(new_logits, axis=1)
    final_misclassified = np.where(new_preds != y)[0]
    diagnostics["final_misclassified"] = final_misclassified.tolist()
    diagnostics["final_accuracy"] = float(np.mean(new_preds == y))
    diagnostics["max_abs_head_weight"] = float(np.max(np.abs(w["head.weight"])))
    diagnostics["max_abs_head_bias"] = float(np.max(np.abs(w["head.bias"])))
    
    return {"weights": w, "diagnostics": diagnostics}
