import numpy as np

def edit(w, f):
    # Extract frozen features and labels from the full feedback pool
    X = f["features"].astype(np.float64)      # (2000, 1568)
    y = f["y"].astype(np.int64)               # (2000,)
    N, D = X.shape
    num_classes = 10

    # One-hot encode targets
    Y = np.zeros((N, num_classes), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Augment features with a bias column
    X_aug = np.hstack([X, np.ones((N, 1), dtype=np.float64)])  # (N, D+1)
    D_aug = D + 1

    # Compute normal equation components
    XtX = X_aug.T @ X_aug
    XtY = X_aug.T @ Y

    # Ridge penalty (do not penalize the bias term)
    alpha = 1.0
    penalty = np.eye(D_aug, dtype=np.float64) * alpha
    penalty[-1, -1] = 0.0

    # Closed-form ridge solution
    W = np.linalg.solve(XtX + penalty, XtY)   # (D+1, num_classes)

    # Split into weight and bias
    head_weight = W[:D, :].T                  # (10, 1568)
    head_bias = W[D, :]                       # (10,)

    # Clip to safe bounds (should not be needed, but ensures validity)
    head_weight = np.clip(head_weight, -1000.0, 1000.0)
    head_bias = np.clip(head_bias, -1000.0, 1000.0)

    # Update the parameter dictionary
    w["head.weight"] = head_weight.astype(np.float32)
    w["head.bias"] = head_bias.astype(np.float32)

    # Diagnostics
    logits = X_aug @ W
    preds = np.argmax(logits, axis=1)
    train_acc = float(np.mean(preds == y))
    diagnostics = {
        "alpha": alpha,
        "train_accuracy": train_acc,
        "max_abs_head_weight": float(np.max(np.abs(head_weight))),
        "max_abs_head_bias": float(np.max(np.abs(head_bias))),
        "mean_abs_head_weight": float(np.mean(np.abs(head_weight))),
    }

    return {"weights": w, "diagnostics": diagnostics}
