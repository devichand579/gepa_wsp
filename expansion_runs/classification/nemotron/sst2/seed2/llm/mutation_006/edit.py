def edit(w, f):
    import numpy as np
    
    X = f["features"].astype(np.float32)          # [4000, 32]
    y = f["y"].astype(np.int64)                   # [4000]
    
    lam = np.float32(0.001)
    
    new_weight = np.zeros((2, 32), dtype=np.float32)
    new_bias   = np.zeros(2, dtype=np.float32)
    
    for c in range(2):
        y_c = (y == c).astype(np.float32)         # [4000] binary indicator
        
        # Closed‑form ridge normal equation: (XᵀX + λI) w = Xᵀ y_c
        XtX = X.T @ X + lam * np.eye(32, dtype=np.float32)
        Xty = X.T @ y_c
        w_c = np.linalg.solve(XtX, Xty)           # [32]
        new_weight[c] = w_c
        
        # Bias as mean residual for this class
        residual = y_c - X @ w_c
        b_c = float(np.mean(residual))
        new_bias[c] = b_c
    
    # Edit head parameters in‑place (shapes and names preserved)
    w["head.weight"] = new_weight
    w["head.bias"]   = new_bias
    
    # Diagnostics: compute new predictions on the feedback pool
    logits_new = X @ w["head.weight"].T + w["head.bias"]   # [4000, 2]
    logits_max = logits_new.max(axis=1, keepdims=True)
    probs_new  = np.exp(logits_new - logits_max) / np.exp(logits_new - logits_max).sum(axis=1, keepdims=True)
    acc_new    = float((probs_new.argmax(axis=1) == y).mean())
    
    diag = {
        "ridge_lambda": float(lam),
        "head_weight_norm": float(np.linalg.norm(new_weight)),
        "head_bias": new_bias.tolist(),
        "feedback_accuracy": acc_new,
        "feedback_n": int(X.shape[0])
    }
    
    return {"weights": w, "diagnostics": diag}
