import numpy as np

def edit(w, f):
    # Extract features and labels from the full feedback pool
    X = f["features"].astype(np.float64)          # (2000, 1568)
    y = f["y"].astype(np.int64)                   # (2000,)
    n, d = X.shape
    n_classes = 10

    # One-hot targets
    Y = np.zeros((n, n_classes), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    # Ridge regression with tiny regularization to allow perfect separation
    lam = 1e-6
    # Add bias by augmenting X with a column of ones
    X_aug = np.hstack([X, np.ones((n, 1))])       # (2000, 1569)
    # Solve (X_aug^T X_aug + lam I) W_aug = X_aug^T Y
    A = X_aug.T @ X_aug + lam * np.eye(d + 1)
    B = X_aug.T @ Y
    W_aug = np.linalg.solve(A, B)                 # (1569, 10)

    # Extract weights and biases
    new_head_weight = W_aug[:d, :].T.astype(np.float32)   # (10, 1568)
    new_head_bias = W_aug[d, :].astype(np.float32)        # (10,)

    # Verify training accuracy of the new head
    logits_new = X @ new_head_weight.T + new_head_bias
    preds_new = np.argmax(logits_new, axis=1)
    acc_new = np.mean(preds_new == y)

    # If the new head is not perfect, fall back to the current head
    if acc_new < 1.0:
        # Keep original weights
        new_head_weight = w["head.weight"].copy()
        new_head_bias = w["head.bias"].copy()

    # Update the weights dictionary
    w["head.weight"] = new_head_weight
    w["head.bias"] = new_head_bias

    # Diagnostics
    diagnostics = {
        "new_head_weight_max_abs": float(np.max(np.abs(new_head_weight))),
        "new_head_bias_max_abs": float(np.max(np.abs(new_head_bias))),
        "training_accuracy_new": float(acc_new),
        "fallback_used": bool(acc_new < 1.0)
    }

    return {"weights": w, "diagnostics": diagnostics}
