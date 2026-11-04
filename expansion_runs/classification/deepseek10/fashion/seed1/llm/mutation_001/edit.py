import numpy as np

def edit(w, f):
    # Extract features and labels from the full feedback pool
    X = f["features"]          # (2000, 1568)
    y = f["y"]                 # (2000,)
    N, D = X.shape
    num_classes = 10

    # One-hot encode labels
    Y = np.zeros((N, num_classes), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    # Standardize features for numerical stability
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-6
    X_norm = (X - X_mean) / X_std

    # Augment with bias column
    X_aug = np.hstack([X_norm, np.ones((N, 1), dtype=np.float32)])

    # Ridge regression: solve (X^T X + λI) W = X^T Y
    lam = 1e-2
    A = X_aug.T @ X_aug + lam * np.eye(D + 1, dtype=np.float32)
    B = X_aug.T @ Y
    W_aug = np.linalg.solve(A, B)   # (D+1, 10)

    W_norm = W_aug[:D, :].T         # (10, D)
    b_norm = W_aug[D, :]            # (10,)

    # Convert back to original feature space
    new_head_weight = W_norm / X_std[None, :]
    new_head_bias = b_norm - np.sum(W_norm * X_mean[None, :] / X_std[None, :], axis=1)

    # Update the head parameters
    w["head.weight"] = new_head_weight.astype(np.float32)
    w["head.bias"] = new_head_bias.astype(np.float32)

    # Diagnostics: training accuracy and weight statistics
    logits = X @ new_head_weight.T + new_head_bias
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == y)

    diagnostics = {
        "train_accuracy": float(acc),
        "lambda": lam,
        "head_weight_norm": float(np.linalg.norm(new_head_weight)),
        "head_bias_norm": float(np.linalg.norm(new_head_bias)),
        "max_abs_weight": float(np.max(np.abs(new_head_weight))),
        "max_abs_bias": float(np.max(np.abs(new_head_bias)))
    }

    return {"weights": w, "diagnostics": diagnostics}
