import numpy as np

def edit(w, f):
    # Extract features and labels from the full feedback pool
    X = f["features"].astype(np.float64)          # [2000, 1568]
    y = f["y"].astype(np.int64)                   # [2000]
    n, p = X.shape
    n_classes = 10

    # One-hot targets
    Y = np.zeros((n, n_classes), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    # Standardize features (column-wise)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    X_std = (X - mu) / sigma

    # Add bias column
    X_aug = np.hstack([X_std, np.ones((n, 1))])   # [2000, 1569]

    # Ridge regression closed-form: W = (X^T X + λ I)^{-1} X^T Y
    lam = 316.0
    A = X_aug.T @ X_aug
    A += lam * np.eye(A.shape[0])
    # Solve for all classes at once
    W_aug = np.linalg.solve(A, X_aug.T @ Y)       # [1569, 10]

    # Split back into weights and bias for standardized features
    W_std = W_aug[:-1, :].T                       # [10, 1568]
    b_std = W_aug[-1, :]                          # [10]

    # Unstandardize: logits = X_std @ W_std.T + b_std
    #                 = ((X - mu)/sigma) @ W_std.T + b_std
    #                 = X @ (W_std/sigma).T + (b_std - (mu/sigma) @ W_std.T)
    W_new = W_std / sigma                         # [10, 1568]
    b_new = b_std - (mu / sigma) @ W_std.T        # [10]

    # Clip to safe range (should already be finite and small)
    W_new = np.clip(W_new, -1000.0, 1000.0).astype(np.float32)
    b_new = np.clip(b_new, -1000.0, 1000.0).astype(np.float32)

    # Write back into the head only
    w["head.weight"] = W_new
    w["head.bias"] = b_new

    # Diagnostics
    diagnostics = {
        "head_weight_norm": float(np.linalg.norm(W_new)),
        "head_bias_norm": float(np.linalg.norm(b_new)),
        "head_weight_max_abs": float(np.max(np.abs(W_new))),
        "head_bias_max_abs": float(np.max(np.abs(b_new))),
        "lambda": lam,
    }

    return {"weights": w, "diagnostics": diagnostics}
