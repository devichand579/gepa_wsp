import numpy as np

def edit(w, f):
    # Extract arrays
    features = f["features"]  # (2000, 1568)
    y = f["y"]                # (2000,)
    logits = f["logits"]      # (2000, 10)
    
    # Current predictions
    pred = np.argmax(logits, axis=1)
    mis = np.where(pred != y)[0]
    
    diagnostics = {
        "initial_misclassified": mis.tolist(),
        "initial_accuracy": float(np.mean(pred == y)),
        "initial_cross_entropy": float(-np.mean(np.log(f["probs"][np.arange(len(y)), y] + 1e-12))),
    }
    
    if len(mis) == 0:
        diagnostics["action"] = "none"
        return {"weights": w, "diagnostics": diagnostics}
    
    # Process each misclassified sample
    for i in mis:
        correct = int(y[i])
        wrong = int(pred[i])
        x = features[i]  # (1568,)
        
        # Current margin between correct and wrong
        margin = logits[i, correct] - logits[i, wrong]
        safety = 1.0
        if margin >= safety:
            continue
        
        xx = np.dot(x, x) + 1.0
        alpha_req = (safety - margin) / (2.0 * xx)
        
        # f_j = features @ x + 1.0
        f_j = features @ x + 1.0  # (2000,)
        
        alpha_max = np.inf
        
        # Case 1: t != correct, k == correct
        mask1 = (y != correct) & (np.arange(len(y)) != i)
        if np.any(mask1):
            bound1 = (logits[mask1, y[mask1]] - logits[mask1, correct]) / f_j[mask1]
            if np.any(bound1 > 0):
                alpha_max = min(alpha_max, np.min(bound1[bound1 > 0]))
        
        # Case 2: t == wrong, k != wrong
        mask2 = (y == wrong) & (np.arange(len(y)) != i)
        if np.any(mask2):
            logits_temp = logits[mask2].copy()
            logits_temp[np.arange(len(logits_temp)), wrong] = -np.inf
            max_other = np.max(logits_temp, axis=1)
            bound2 = (logits[mask2, wrong] - max_other) / f_j[mask2]
            if np.any(bound2 > 0):
                alpha_max = min(alpha_max, np.min(bound2[bound2 > 0]))
        
        # Case 3: t == wrong, k == correct
        mask3 = (y == wrong) & (np.arange(len(y)) != i)
        if np.any(mask3):
            bound3 = (logits[mask3, wrong] - logits[mask3, correct]) / (2.0 * f_j[mask3])
            if np.any(bound3 > 0):
                alpha_max = min(alpha_max, np.min(bound3[bound3 > 0]))
        
        if np.isinf(alpha_max):
            alpha = alpha_req * 1.1
        else:
            alpha = min(alpha_max * 0.99, alpha_req * 1.1)
        
        if alpha <= 0:
            continue
        
        # Update head weights and bias
        w["head.weight"][correct] += alpha * x
        w["head.weight"][wrong] -= alpha * x
        w["head.bias"][correct] += alpha
        w["head.bias"][wrong] -= alpha
        
        # Clip to [-1000, 1000]
        w["head.weight"] = np.clip(w["head.weight"], -1000.0, 1000.0)
        w["head.bias"] = np.clip(w["head.bias"], -1000.0, 1000.0)
        
        diagnostics[f"sample_{i}"] = {
            "correct": correct,
            "wrong": wrong,
            "margin": float(margin),
            "alpha_req": float(alpha_req),
            "alpha_max": float(alpha_max) if not np.isinf(alpha_max) else None,
            "chosen_alpha": float(alpha),
        }
    
    # Recompute logits with updated head
    new_logits = features @ w["head.weight"].T + w["head.bias"]
    new_pred = np.argmax(new_logits, axis=1)
    new_acc = float(np.mean(new_pred == y))
    # Compute cross-entropy
    max_logits = np.max(new_logits, axis=1, keepdims=True)
    exp_logits = np.exp(new_logits - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    new_ce = float(-np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12)))
    
    diagnostics["final_accuracy"] = new_acc
    diagnostics["final_cross_entropy"] = new_ce
    diagnostics["final_misclassified"] = np.where(new_pred != y)[0].tolist()
    
    return {"weights": w, "diagnostics": diagnostics}
