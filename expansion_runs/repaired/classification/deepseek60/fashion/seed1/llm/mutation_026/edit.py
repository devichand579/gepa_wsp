import numpy as np

def edit(w, f):
    # Extract full feedback features and labels
    X = f["features"].astype(np.float64)          # (2000, 1568)
    y = f["y"].astype(np.int64)                   # (2000,)
    N, D = X.shape

    # One-hot encode labels
    Y = np.zeros((N, 10), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Augment features with a bias column
    X_aug = np.concatenate([X, np.ones((N, 1), dtype=np.float64)], axis=1)  # (2000, 1569)

    # Ridge regression closed-form solution:
    # minimize ||X_aug @ W_aug.T - Y||^2 + lam * ||W_aug||^2
    lam = 1e-4
    A = X_aug.T @ X_aug + lam * np.eye(D + 1)
    B = X_aug.T @ Y
    W_aug = np.linalg.solve(A, B).T               # (10, 1569)

    W = W_aug[:, :D]                              # (10, 1568)
    b = W_aug[:, D]                               # (10,)

    # Scale weights and bias to max absolute value 999
    max_abs = max(np.abs(W).max(), np.abs(b).max())
    scale = 999.0 / max_abs if max_abs > 0 else 1.0
    W *= scale
    b *= scale

    # Update the head parameters
    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

    diagnostics = {
        "lambda": lam,
        "scale": float(scale),
        "max_abs_head_weight": float(np.abs(w["head.weight"]).max()),
        "max_abs_head_bias": float(np.abs(w["head.bias"]).max()),
    }
    return {"weights": w, "diagnostics": diagnostics}
