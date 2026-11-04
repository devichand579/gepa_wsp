import numpy as np

def edit(w, f):
    # Extract features and labels
    X = f["features"].astype(np.float64)  # (2000, 1568)
    y = f["y"].astype(np.int64)           # (2000,)
    N, D = X.shape
    C = 10
    
    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0
    
    # Current predictions from logits
    logits = f["logits"].astype(np.float64)
    pred = np.argmax(logits, axis=1)
    errors = (pred != y)
    
    # Upweight misclassified examples
    weights = np.ones(N, dtype=np.float64)
    weights[errors] = 100.0
    
    # Weighted means
    W_sum = weights.sum()
    mean_X = (weights[:, None] * X).sum(axis=0) / W_sum
    mean_Y = (weights[:, None] * Y).sum(axis=0) / W_sum
    
    # Centered data
    Xc = X - mean_X
    Yc = Y - mean_Y
    
    # Weighted least-squares (closed-form)
    Xw = Xc * np.sqrt(weights)[:, None]
    Yw = Yc * np.sqrt(weights)[:, None]
    W_T, residuals, rank, s = np.linalg.lstsq(Xw, Yw, rcond=None)
    W = W_T.T  # (10, 1568)
    b = mean_Y - mean_X @ W.T  # (10,)
    
    # Scale to max absolute value 1000
    max_val = max(np.max(np.abs(W)), np.max(np.abs(b)))
    if max_val > 0:
        scale = 1000.0 / max_val
        W *= scale
        b *= scale
    
    # Update only the head weights
    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)
    
    # Diagnostics
    new_logits = X @ W.T + b
    new_pred = np.argmax(new_logits, axis=1)
    acc = np.mean(new_pred == y)
    
    batch_ids = f["batch.ids"]
    batch_y = f["batch.y"]
    batch_X = X[batch_ids]
    batch_logits = batch_X @ W.T + b
    batch_pred = np.argmax(batch_logits, axis=1)
    batch_acc = np.mean(batch_pred == batch_y)
    
    diagnostics = {
        "full_pool_accuracy": float(acc),
        "paired_batch_accuracy": float(batch_acc),
        "num_errors_before": int(errors.sum()),
        "num_errors_after": int((new_pred != y).sum()),
        "max_abs_weight": float(np.max(np.abs(W))),
        "max_abs_bias": float(np.max(np.abs(b))),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
