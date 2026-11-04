import numpy as np

def edit(w, f):
    # Extract feedback features and labels
    X = f["features"].astype(np.float64)   # (2000, 128)
    y = f["y"].astype(np.int64)            # (2000,)
    n, d = X.shape
    num_classes = 10

    # One-hot targets
    Y = np.zeros((n, num_classes), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    # Add bias column
    X_aug = np.hstack([X, np.ones((n, 1), dtype=np.float64)])  # (2000, 129)

    # Ridge regression: solve (X_aug^T X_aug + lambda I) W = X_aug^T Y
    # Use a small ridge for numerical stability and mild regularisation.
    lambda_reg = 1e-3
    A = X_aug.T @ X_aug
    A += lambda_reg * np.eye(d + 1, dtype=np.float64)
    B = X_aug.T @ Y
    W_aug = np.linalg.solve(A, B)  # (129, 10)

    # Extract weights and biases for fc2
    new_fc2_weight = W_aug[:d, :].T.astype(np.float32)   # (10, 128)
    new_fc2_bias = W_aug[d, :].astype(np.float32)        # (10,)

    # Update weights dictionary
    w["fc2.weight"] = new_fc2_weight
    w["fc2.bias"] = new_fc2_bias

    # Diagnostics
    diagnostics = {
        "lambda_reg": float(lambda_reg),
        "fc2_weight_mean": float(np.mean(new_fc2_weight)),
        "fc2_weight_std": float(np.std(new_fc2_weight)),
        "fc2_bias_mean": float(np.mean(new_fc2_bias)),
        "fc2_bias_std": float(np.std(new_fc2_bias)),
        "train_accuracy": float(np.mean(np.argmax(X_aug @ W_aug, axis=1) == y)),
    }

    return {"weights": w, "diagnostics": diagnostics}
