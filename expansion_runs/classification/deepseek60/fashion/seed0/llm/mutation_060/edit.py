import numpy as np

def edit(w, f):
    # Extract features and labels from the full feedback pool
    X = f["features"].astype(np.float64)  # [N, 1568]
    y = f["y"].astype(np.int64)           # [N]
    N, D = X.shape
    C = 10

    # Standardize features (mean/std over the feedback pool)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    Xs = (X - mu) / sigma

    # One-hot labels
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Ridge regression with a fixed moderate lambda.
    # We use the normal equations: (X^T X + lambda I) W = X^T Y
    # Add a small intercept by augmenting a column of ones.
    Xs_aug = np.hstack([Xs, np.ones((N, 1))])
    D_aug = D + 1
    lam = 100.0  # moderate regularization
    A = Xs_aug.T @ Xs_aug + lam * np.eye(D_aug)
    B = Xs_aug.T @ Y
    W_aug = np.linalg.solve(A, B)  # [D+1, C]

    # W_aug[:D] are weights for standardized features, W_aug[D] is bias.
    W_std = W_aug[:D].T   # [C, D]
    b_std = W_aug[D]      # [C]

    # Fold standardization back into the original feature space:
    # logits = W_std @ ((x - mu)/sigma) + b_std
    #        = (W_std / sigma) @ x + (b_std - W_std @ (mu/sigma))
    W_new = W_std / sigma  # [C, D]
    b_new = b_std - W_std @ (mu / sigma)

    # Convert to float32 and clip to safe range
    W_new = np.clip(W_new.astype(np.float32), -1000.0, 1000.0)
    b_new = np.clip(b_new.astype(np.float32), -1000.0, 1000.0)

    # Update the head weights and biases
    w["head.weight"] = W_new
    w["head.bias"] = b_new

    # Diagnostics
    diagnostics = {
        "lambda": lam,
        "head_weight_norm": float(np.linalg.norm(W_new)),
        "head_bias_norm": float(np.linalg.norm(b_new)),
        "head_weight_max_abs": float(np.max(np.abs(W_new))),
        "head_bias_max_abs": float(np.max(np.abs(b_new))),
        "feature_mean_norm": float(np.linalg.norm(mu)),
        "feature_std_mean": float(np.mean(sigma)),
    }

    return {"weights": w, "diagnostics": diagnostics}
