import numpy as np

def edit(w, f):
    features = f["features"]  # (N, D)
    logits = f["logits"]      # (N, C)
    y = f["y"]                # (N,)
    N, D = features.shape
    C = logits.shape[1]
    
    preds = np.argmax(logits, axis=1)
    mis = np.where(preds != y)[0]
    
    # initial metrics
    probs = f["probs"]
    initial_acc = float(np.mean(preds == y))
    initial_ce = float(-np.mean(np.log(np.clip(probs[np.arange(N), y], 1e-12, 1.0))))
    
    diag = {
        "initial_misclassified": mis.tolist(),
        "initial_accuracy": initial_acc,
        "initial_cross_entropy": initial_ce,
    }
    
    if len(mis) == 0:
        return {"weights": w, "diagnostics": diag}
    
    # Fix the first misclassified sample
    i = mis[0]
    x = features[i].astype(np.float64)
    c = int(y[i])
    k = int(preds[i])
    m = float(logits[i, c] - logits[i, k])
    dot = float(np.dot(x, x))
    if dot <= 0:
        return {"weights": w, "diagnostics": diag}
    alpha_req = -m / (2.0 * dot)
    
    # Compute alpha_max: maximum alpha that keeps all other samples correct
    s = features @ x  # (N,)
    alpha_max = np.inf
    for p in range(C):
        mask = (y != p)
        if not np.any(mask):
            continue
        M = logits[mask, y[mask]] - logits[mask, p]  # margins
        yj = y[mask]
        sj = s[mask]
        # coeff = (yj==c) - (yj==k) - (p==c) + (p==k)
        coeff = (yj == c).astype(np.float64) - (yj == k).astype(np.float64) - (p == c) + (p == k)
        denom = -sj * coeff
        valid = denom > 0
        if np.any(valid):
            alpha_limit = M[valid] / denom[valid]
            alpha_max = min(alpha_max, np.min(alpha_limit))
    
    diag["alpha_req"] = float(alpha_req)
    diag["alpha_max"] = float(alpha_max) if np.isfinite(alpha_max) else None
    
    if np.isfinite(alpha_max) and alpha_max > alpha_req:
        alpha = min(alpha_max * 0.99, alpha_req * 1.1)
        if alpha <= alpha_req:
            alpha = alpha_req * 1.001
    else:
        alpha = alpha_req * 1.001
    
    diag["alpha_used"] = float(alpha)
    
    # Update head weights
    head_w = w["head.weight"].astype(np.float64)
    head_w[c] += alpha * x
    head_w[k] -= alpha * x
    head_w = np.clip(head_w, -1000.0, 1000.0)
    w["head.weight"] = head_w.astype(np.float32)
    
    # Compute new metrics
    new_logits = features @ w["head.weight"].T + w["head.bias"]
    new_preds = np.argmax(new_logits, axis=1)
    new_acc = float(np.mean(new_preds == y))
    # softmax CE
    max_logits = np.max(new_logits, axis=1, keepdims=True)
    exp_logits = np.exp(new_logits - max_logits)
    new_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    new_ce = float(-np.mean(np.log(np.clip(new_probs[np.arange(N), y], 1e-12, 1.0))))
    
    diag["final_accuracy"] = new_acc
    diag["final_cross_entropy"] = new_ce
    diag["final_misclassified"] = np.where(new_preds != y)[0].tolist()
    
    return {"weights": w, "diagnostics": diag}
