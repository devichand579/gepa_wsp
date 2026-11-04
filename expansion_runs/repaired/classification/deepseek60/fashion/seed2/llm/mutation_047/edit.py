import numpy as np

def edit(w, f):
    # Extract features and labels from the feedback pool
    X = f["features"].astype(np.float64)          # (2000, 1568)
    y = f["y"]                                    # (2000,)
    N, D = X.shape

    # Augment features with a bias column
    X_aug = np.hstack([X, np.ones((N, 1))])       # (2000, 1569)

    # One-hot encode labels
    Y = np.zeros((N, 10), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Minimum-norm least-squares solution with tiny ridge for numerical stability:
    # W_aug = X_aug^T (X_aug X_aug^T + λ I)^{-1} Y
    lambda_reg = 1e-6
    K = X_aug @ X_aug.T                           # (2000, 2000)
    K_reg = K + lambda_reg * np.eye(N)
    A = np.linalg.solve(K_reg, Y)                 # (2000, 10)
    W_aug = X_aug.T @ A                           # (1569, 10)

    # Split into head weight and bias
    head_weight = W_aug[:D, :].T                  # (10, 1568)
    head_bias   = W_aug[D, :]                     # (10,)

    # Update the weights dictionary
    w["head.weight"] = head_weight.astype(np.float32)
    w["head.bias"]   = head_bias.astype(np.float32)

    # Diagnostics
    logits = X_aug @ W_aug
    preds = np.argmax(logits, axis=1)
    acc = float(np.mean(preds == y))
    diag = {
        "train_accuracy": acc,
        "max_abs_head_weight": float(np.max(np.abs(head_weight))),
        "max_abs_head_bias": float(np.max(np.abs(head_bias))),
        "mean_abs_head_weight": float(np.mean(np.abs(head_weight))),
    }

    return {"weights": w, "diagnostics": diag}
