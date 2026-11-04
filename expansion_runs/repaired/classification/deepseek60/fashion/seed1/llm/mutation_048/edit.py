def edit(w, f):
    import numpy as np
    
    # Extract features and labels
    X = f["features"].astype(np.float64)  # (2000, 1568)
    y = f["y"].astype(np.int64)           # (2000,)
    N, D = X.shape
    num_classes = 10
    
    # One-hot targets
    Y = np.zeros((N, num_classes), dtype=np.float64)
    Y[np.arange(N), y] = 1.0
    
    # Augment features with bias column
    X_aug = np.hstack([X, np.ones((N, 1), dtype=np.float64)])  # (N, D+1)
    
    # Ridge regression: solve (X_aug^T X_aug + λ I) W = X_aug^T Y
    lam = 1e-6
    A = X_aug.T @ X_aug + lam * np.eye(D+1)
    B = X_aug.T @ Y
    W_aug = np.linalg.solve(A, B)  # (D+1, 10)
    
    # Extract weights and bias
    W = W_aug[:-1, :].T  # (10, D)
    b = W_aug[-1, :]     # (10,)
    
    # Scale to max absolute value 1000
    max_abs = max(np.max(np.abs(W)), np.max(np.abs(b)))
    if max_abs > 0:
        scale = 1000.0 / max_abs
        W *= scale
        b *= scale
    else:
        scale = 1.0
    
    # Update weights
    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)
    
    # Diagnostics: full pool
    logits = X @ W.T + b
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == y)
    logits_max = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    ce = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))
    
    # Diagnostics: paired batch
    X_batch = f["layer.head.input0"].astype(np.float64)
    y_batch = f["batch.y"].astype(np.int64)
    logits_batch = X_batch @ W.T + b
    preds_batch = np.argmax(logits_batch, axis=1)
    acc_batch = np.mean(preds_batch == y_batch)
    logits_max_b = np.max(logits_batch, axis=1, keepdims=True)
    exp_logits_b = np.exp(logits_batch - logits_max_b)
    probs_b = exp_logits_b / np.sum(exp_logits_b, axis=1, keepdims=True)
    ce_batch = -np.mean(np.log(probs_b[np.arange(len(y_batch)), y_batch] + 1e-12))
    
    diagnostics = {
        "full_pool_accuracy": float(acc),
        "full_pool_cross_entropy": float(ce),
        "paired_batch_accuracy": float(acc_batch),
        "paired_batch_cross_entropy": float(ce_batch),
        "max_abs_weight": float(np.max(np.abs(W))),
        "max_abs_bias": float(np.max(np.abs(b))),
        "scale": float(scale)
    }
    
    return {"weights": w, "diagnostics": diagnostics}
