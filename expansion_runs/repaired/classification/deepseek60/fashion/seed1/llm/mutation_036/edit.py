import numpy as np

def edit(w, f):
    # Extract full feedback features and labels
    X = f["features"].astype(np.float64)          # (2000, 1568)
    y = f["y"].astype(np.int64)                   # (2000,)
    N, D = X.shape
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Augment with bias column
    X_aug = np.hstack([X, np.ones((N, 1), dtype=np.float64)])  # (2000, 1569)

    # Ridge regression: solve (X_aug^T X_aug + lambda I) W_aug = X_aug^T Y
    # Use a small lambda to fit training data well while avoiding numerical issues.
    lam = 1e-6
    A = X_aug.T @ X_aug + lam * np.eye(D + 1, dtype=np.float64)
    B = X_aug.T @ Y
    W_aug = np.linalg.solve(A, B).T               # (10, 1569)

    W_new = W_aug[:, :D].astype(np.float32)       # (10, 1568)
    b_new = W_aug[:, D].astype(np.float32)        # (10,)

    # Rescale to maximum allowed magnitude 1000
    max_abs = max(np.max(np.abs(W_new)), np.max(np.abs(b_new)))
    if max_abs > 0:
        scale = 1000.0 / max_abs
        W_new *= scale
        b_new *= scale

    # Clip to be safe (should already be within limits)
    W_new = np.clip(W_new, -1000.0, 1000.0)
    b_new = np.clip(b_new, -1000.0, 1000.0)

    # Update the head only; keep convolutional layers unchanged
    w["head.weight"] = W_new
    w["head.bias"] = b_new

    # Diagnostics
    diagnostics = {
        "max_abs_head_weight": float(np.max(np.abs(W_new))),
        "max_abs_head_bias": float(np.max(np.abs(b_new))),
        "scale": float(scale) if max_abs > 0 else 1.0,
        "lambda": lam,
    }

    return {"weights": w, "diagnostics": diagnostics}
