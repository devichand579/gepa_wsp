import numpy as np

def edit(w, f):
    # Extract current final hidden features and labels
    X = f["features"].astype(np.float32)  # (2000, 1568)
    y = f["y"].astype(np.int64)           # (2000,)
    n, d = X.shape
    n_classes = 10

    # Standardize features (zero mean, unit variance)
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True)
    sigma[sigma < 1e-6] = 1.0
    Xs = (X - mu) / sigma

    # Add bias term
    X_aug = np.concatenate([Xs, np.ones((n, 1), dtype=np.float32)], axis=1)  # (n, d+1)

    # One-hot targets
    Y = np.zeros((n, n_classes), dtype=np.float32)
    Y[np.arange(n), y] = 1.0

    # Ridge regression with a small regularization strength.
    # Choose lambda relative to the scale of X_aug^T X_aug.
    # Since features are standardized, diagonal of X^T X is ~n.
    lam = 1.0  # small ridge; can be tuned, but 1.0 is a safe default for standardized data.
    A = X_aug.T @ X_aug + lam * np.eye(d + 1, dtype=np.float32)
    B = X_aug.T @ Y
    W_aug = np.linalg.solve(A, B)  # (d+1, n_classes)

    # Extract weights and bias
    W = W_aug[:d, :].T.astype(np.float32)      # (10, 1568)
    b = W_aug[d, :].astype(np.float32)         # (10,)

    # The new head operates on the original (unstandardized) features.
    # We must fold the standardization into the weights:
    # logits = (X - mu)/sigma @ W^T + b = X @ (W/sigma)^T + (b - mu/sigma @ W^T)
    W_orig = W / sigma  # broadcasting: (10, 1568) / (1, 1568)
    b_orig = b - (mu / sigma) @ W.T  # (1, 1568) @ (1568, 10) -> (1, 10)
    b_orig = b_orig.ravel().astype(np.float32)

    # Update weights
    w["head.weight"] = W_orig
    w["head.bias"] = b_orig

    # Diagnostics
    diagnostics = {
        "head_weight_mean": float(np.mean(W_orig)),
        "head_weight_std": float(np.std(W_orig)),
        "head_weight_max_abs": float(np.max(np.abs(W_orig))),
        "head_bias_mean": float(np.mean(b_orig)),
        "head_bias_std": float(np.std(b_orig)),
        "lambda": lam,
    }

    return {"weights": w, "diagnostics": diagnostics}
