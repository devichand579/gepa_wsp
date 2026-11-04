import numpy as np

def edit(w, f):
    # Extract full feedback features and labels
    X = f["features"].astype(np.float64)
    y = f["y"].astype(np.int64)
    N, D = X.shape

    # One-hot encode labels
    Y = np.zeros((N, 10), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Augment features with a bias column
    X_aug = np.hstack([X, np.ones((N, 1), dtype=np.float64)])

    # Closed-form ridge regression (normal equations)
    lam = 1e-4
    A = X_aug.T @ X_aug + lam * np.eye(D + 1)
    B = X_aug.T @ Y
    W_aug = np.linalg.solve(A, B)

    # Split weights and bias
    W = W_aug[:-1, :].T   # shape (10, D)
    b = W_aug[-1, :]      # shape (10,)

    # Scale both weights and bias so max absolute weight becomes 1000
    max_abs = np.max(np.abs(W))
    if max_abs > 0:
        scale = 1000.0 / max_abs
        W *= scale
        b *= scale
    else:
        scale = 1.0

    # Ensure all values are within the allowed bounds
    W = np.clip(W, -1000.0, 1000.0)
    b = np.clip(b, -1000.0, 1000.0)

    # Assign back to the network weights
    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

    diagnostics = {
        "lambda": lam,
        "scale": float(scale),
        "max_abs_weight": float(np.max(np.abs(w["head.weight"]))),
        "max_abs_bias": float(np.max(np.abs(w["head.bias"]))),
    }
    return {"weights": w, "diagnostics": diagnostics}
