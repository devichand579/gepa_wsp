import numpy as np

def edit(w, f):
    # Extract full-pool features and labels
    X = f["features"].astype(np.float64)          # (2000, 1568)
    y = f["y"].astype(np.int64)                   # (2000,)
    
    # Current predictions to identify misclassified examples
    logits = f["logits"].astype(np.float64)       # (2000, 10)
    pred = np.argmax(logits, axis=1)
    mis = (pred != y).astype(np.float64)
    
    # Upweight misclassified examples heavily
    weights = 1.0 + 1000.0 * mis
    
    # One-hot targets
    Y = np.zeros((len(y), 10), dtype=np.float64)
    Y[np.arange(len(y)), y] = 1.0
    
    # Augment features with a bias column
    X_aug = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float64)])
    
    # Weighted ridge regression (closed-form normal equations)
    lambda_reg = 1e-5
    Xw = X_aug * weights[:, None]
    A = X_aug.T @ Xw + lambda_reg * np.eye(X_aug.shape[1])
    b = X_aug.T @ (weights[:, None] * Y)
    B = np.linalg.solve(A, b)                     # (1569, 10)
    
    new_weight = B[:1568, :].T                    # (10, 1568)
    new_bias = B[1568, :]                         # (10,)
    
    # Scale so max absolute value is 1000 (preserves argmax)
    max_abs = max(np.max(np.abs(new_weight)), np.max(np.abs(new_bias)))
    if max_abs > 0:
        scale = 1000.0 / max_abs
        new_weight *= scale
        new_bias *= scale
    
    # Clip to be safe and cast to float32
    new_weight = np.clip(new_weight, -1000, 1000).astype(np.float32)
    new_bias = np.clip(new_bias, -1000, 1000).astype(np.float32)
    
    # Update the head weights
    w["head.weight"] = new_weight
    w["head.bias"] = new_bias
    
    # Diagnostics
    new_logits = X @ new_weight.T + new_bias
    new_pred = np.argmax(new_logits, axis=1)
    acc = np.mean(new_pred == y)
    
    X_batch = f["layer.head.input0"].astype(np.float64)
    y_batch = f["batch.y"].astype(np.int64)
    batch_logits = X_batch @ new_weight.T + new_bias
    batch_pred = np.argmax(batch_logits, axis=1)
    batch_acc = np.mean(batch_pred == y_batch)
    
    diagnostics = {
        "full_pool_accuracy": float(acc),
        "paired_batch_accuracy": float(batch_acc),
        "num_errors_before": int(np.sum(mis)),
        "num_errors_after": int(np.sum(new_pred != y)),
        "max_abs_weight": float(np.max(np.abs(new_weight))),
        "max_abs_bias": float(np.max(np.abs(new_bias))),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
