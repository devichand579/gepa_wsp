import numpy as np

def edit(w, f):
    # Read current head weights and bias
    W = w["head.weight"].astype(np.float64)  # [10, 1568]
    b = w["head.bias"].astype(np.float64)    # [10]
    
    # Full feedback features, labels, and current logits
    X = f["features"].astype(np.float64)     # [2000, 1568]
    y = f["y"].astype(np.int64)              # [2000]
    Z = f["logits"].astype(np.float64)       # [2000, 10]
    
    N, D = X.shape
    C = Z.shape[1]
    
    # Current predictions and errors
    pred = np.argmax(Z, axis=1)
    errors = np.where(pred != y)[0]
    
    # If no errors, return unchanged
    if len(errors) == 0:
        return w
    
    # Build target logits: start from current logits
    Y = Z.copy()
    # For each error, increase true class logit and decrease predicted class logit
    margin = 20.0
    for i in errors:
        t = y[i]
        p = pred[i]
        # Increase true class
        Y[i, t] = Z[i, t] + margin
        # Decrease predicted class
        Y[i, p] = Z[i, p] - margin
        # Optionally, push other classes slightly down? Keep as is.
    
    # Weights: high weight on errors, normal on correct
    weights = np.ones(N)
    weights[errors] = 1000.0
    
    # Augment features with bias column
    X_aug = np.hstack([X, np.ones((N, 1))])  # [2000, 1569]
    
    # Weighted least squares with small ridge on weights (not bias)
    sqrt_w = np.sqrt(weights)[:, None]
    Xw = X_aug * sqrt_w
    Yw = Y * sqrt_w
    
    # Regularization: lambda on weights, 0 on bias
    lambda_reg = 1e-6
    reg = np.eye(D + 1) * lambda_reg
    reg[-1, -1] = 0.0
    
    # Solve normal equations: (Xw^T Xw + reg) W_aug^T = Xw^T Yw
    A = Xw.T @ Xw + reg
    B = Xw.T @ Yw
    W_aug_T = np.linalg.solve(A, B)  # [1569, 10]
    W_aug = W_aug_T.T                # [10, 1569]
    
    W_new = W_aug[:, :D]             # [10, 1568]
    b_new = W_aug[:, D]              # [10]
    
    # Scale to ensure max absolute value <= 1000
    max_abs = max(np.max(np.abs(W_new)), np.max(np.abs(b_new)))
    if max_abs > 1000.0:
        scale = 1000.0 / max_abs
        W_new *= scale
        b_new *= scale
    
    # Cast back to float32 and update weights
    w["head.weight"] = W_new.astype(np.float32)
    w["head.bias"] = b_new.astype(np.float32)
    
    # Diagnostics
    new_logits = X @ W_new.T + b_new
    new_pred = np.argmax(new_logits, axis=1)
    new_acc = np.mean(new_pred == y)
    diagnostics = {
        "num_errors_before": int(len(errors)),
        "num_errors_after": int(np.sum(new_pred != y)),
        "full_pool_accuracy_after": float(new_acc),
        "max_abs_weight": float(np.max(np.abs(W_new))),
        "max_abs_bias": float(np.max(np.abs(b_new))),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
