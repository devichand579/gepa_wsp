import numpy as np

def edit(w, f):
    # Extract features and labels
    X = f["features"].astype(np.float64)
    y = f["y"].astype(np.int64)
    N, D = X.shape
    C = 10
    
    # One-hot encode labels
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0
    
    # Augment features with a bias column
    X_aug = np.hstack([X, np.ones((N, 1), dtype=np.float64)])
    
    # Ridge regression (closed-form normal equations)
    lam = 1e-4
    A = X_aug.T @ X_aug + lam * np.eye(D + 1, dtype=np.float64)
    B = X_aug.T @ Y
    W_aug = np.linalg.solve(A, B).T  # shape (C, D+1)
    
    W = W_aug[:, :D]
    b = W_aug[:, D]
    
    # Scale to maximum allowed magnitude
    max_abs = max(np.abs(W).max(), np.abs(b).max())
    if max_abs > 0:
        scale = 1000.0 / max_abs
    else:
        scale = 1.0
    W *= scale
    b *= scale
    
    # Update the head weights
    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)
    
    # Diagnostics
    logits = X @ W.T + b
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == y)
    
    logits_max = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    ce = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))
    
    diagnostics = {
        "scale": float(scale),
        "max_abs_weight": float(np.abs(W).max()),
        "max_abs_bias": float(np.abs(b).max()),
        "full_accuracy": float(acc),
        "cross_entropy": float(ce)
    }
    
    return {"weights": w, "diagnostics": diagnostics}
