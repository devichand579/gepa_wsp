import numpy as np

def edit(w, f):
    # Extract feedback data
    F = f["features"].astype(np.float64)          # (2000, 1568)
    y = f["y"].astype(np.int64)                   # (2000,)
    N, D = F.shape

    # One-hot targets
    Y = np.zeros((N, 10), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Current head parameters
    W0 = w["head.weight"].astype(np.float64)      # (10, 1568)
    b0 = w["head.bias"].astype(np.float64)        # (10,)

    # Standardize features for numerical stability
    mean_F = F.mean(axis=0)
    std_F = F.std(axis=0) + 1e-6
    F_std = (F - mean_F) / std_F

    # Augment with bias column
    X = np.hstack([F_std, np.ones((N, 1), dtype=np.float64)])  # (2000, 1569)

    # Map current head to standardized space
    W0_std = W0 * std_F[None, :]                  # (10, 1568)
    b0_std = b0 + mean_F @ W0.T                   # (10,)
    W0_full = np.hstack([W0_std, b0_std[:, None]])  # (10, 1569)

    # Ridge regression with prior centered at current head
    # Minimize ||X W^T - Y||^2 + lambda ||W - W0_full||^2
    # Solution: W = (X^T X + lambda I)^{-1} (X^T Y + lambda W0_full)
    lambda_reg = 10.0  # moderate regularization toward current head
    XtX = X.T @ X
    XtY = X.T @ Y
    A = XtX + lambda_reg * np.eye(D + 1)
    B = XtY + lambda_reg * W0_full.T
    W_full = np.linalg.solve(A, B).T              # (10, 1569)

    # Split back into weights and bias in standardized space
    W_std = W_full[:, :D]
    b_std = W_full[:, D]

    # Map back to original feature space
    W_new = W_std / std_F[None, :]
    b_new = b_std - (mean_F / std_F) @ W_std.T

    # Update weights
    w["head.weight"] = W_new.astype(np.float32)
    w["head.bias"] = b_new.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "lambda_reg": lambda_reg,
        "max_abs_weight": float(np.max(np.abs(w["head.weight"]))),
        "max_abs_bias": float(np.max(np.abs(w["head.bias"]))),
        "weight_change_norm": float(np.linalg.norm(w["head.weight"] - W0.astype(np.float32))),
        "bias_change_norm": float(np.linalg.norm(w["head.bias"] - b0.astype(np.float32))),
    }

    return {"weights": w, "diagnostics": diagnostics}
