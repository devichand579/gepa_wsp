import numpy as np

def edit(w, f):
    # Read full feedback features and labels
    X = f["features"].astype(np.float64)  # [2000, 1568]
    y = f["y"].astype(np.int64)           # [2000]
    n, d = X.shape
    K = 10

    # One-hot targets
    Y = np.zeros((n, K), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    # Add bias by augmenting features
    Xb = np.concatenate([X, np.ones((n, 1), dtype=np.float64)], axis=1)  # [2000, 1569]

    # Ridge regression with a very small regularization to avoid numerical issues.
    # This is a closed-form least-squares solution, not a gradient step.
    lam = 1e-6
    A = Xb.T @ Xb + lam * np.eye(d + 1, dtype=np.float64)
    B = Xb.T @ Y
    Wb = np.linalg.solve(A, B)  # [1569, 10]

    # Extract weights and bias (transpose to [10, 1568] and [10])
    W_new = Wb[:-1, :].T.astype(np.float32)  # [10, 1568]
    b_new = Wb[-1, :].astype(np.float32)     # [10]

    # Rescale weights and bias to maximize confidence while keeping |weight| <= 1000.
    # The scale factor is chosen so that the maximum absolute weight is exactly 1000.
    max_abs_w = np.max(np.abs(W_new))
    if max_abs_w > 0:
        scale = 1000.0 / max_abs_w
    else:
        scale = 1.0
    W_new = (W_new * scale).astype(np.float32)
    b_new = (b_new * scale).astype(np.float32)

    # Ensure finite and within bounds
    W_new = np.clip(W_new, -1000.0, 1000.0).astype(np.float32)
    b_new = np.clip(b_new, -1000.0, 1000.0).astype(np.float32)

    # Update the head weights and bias in the parameter dictionary
    w["head.weight"] = W_new
    w["head.bias"] = b_new

    # Diagnostics
    diagnostics = {
        "max_abs_head_weight": float(np.max(np.abs(W_new))),
        "max_abs_head_bias": float(np.max(np.abs(b_new))),
        "scale": float(scale),
        "lambda": lam,
    }

    return {"weights": w, "diagnostics": diagnostics}
