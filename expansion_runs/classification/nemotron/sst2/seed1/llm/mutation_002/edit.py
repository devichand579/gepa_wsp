def edit(w, f):
    # Extract full feedback pool
    X = f["features"]                     # [4000, 32], float32
    y = f["y"].astype(np.int64)           # [4000], int64 {0,1}
    
    N = X.shape[0]
    # Build one-hot labels
    Y = np.zeros((N, 2), dtype=np.float32)
    Y[np.arange(N), y] = 1.0
    
    # Ridge regularization coefficient (small constant)
    lam = 1e-4
    
    # Augment features with a bias term (ones column)
    X_aug = np.hstack([X, np.ones((N, 1), dtype=np.float32)], axis=1)  # [N, 33]
    
    # Closed-form ridge regression: (X_aug^T X_aug + λ I) W_aug^T = X_aug^T Y
    # W_aug maps [features + bias] -> logits, shape [2, 33]
    A = X_aug.T @ X_aug + lam * np.eye(33, dtype=np.float32)          # [33, 33]
    B = X_aug.T @ Y                                                    # [33, 2]
    W_aug = np.linalg.solve(A, B).T                                   # [2, 33]
    
    new_weight = W_aug[:, :32].astype(np.float32)                     # [2, 32]
    new_bias   = W_aug[:, 32].astype(np.float32)                      # [2]
    
    # Preserve finiteness and absolute-value constraint
    new_weight = np.clip(new_weight, -1000.0, 1000.0)
    new_bias   = np.clip(new_bias,   -1000.0, 1000.0)
    
    # Edit only the head parameters; all other weights stay untouched
    w["head.weight"] = new_weight
    w["head.bias"]   = new_bias
    
    # Diagnostics on the feedback pool
    logits_new = X @ new_weight.T + new_bias           # [4000, 2]
    probs_new  = np.exp(logits_new) / np.exp(logits_new).sum(axis=1, keepdims=True)
    pred_new   = np.argmax(probs_new, axis=1)
    accuracy_new = float(np.mean(pred_new == y))
    
    diag = {
        "head_weight_norm": float(np.linalg.norm(new_weight)),
        "head_bias_norm":   float(np.linalg.norm(new_bias)),
        "new_accuracy":     accuracy_new,
    }
    
    return {"weights": w, "diagnostics": diag}
