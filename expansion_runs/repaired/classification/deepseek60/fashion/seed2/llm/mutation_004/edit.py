import numpy as np

def edit(w, f):
    # Extract full feedback features and labels
    X = f["features"].astype(np.float64)  # (2000, 1568)
    y = f["y"].astype(np.int64)           # (2000,)
    N, D = X.shape
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Augment features with a bias column
    X_aug = np.hstack([X, np.ones((N, 1), dtype=np.float64)])  # (2000, 1569)

    # Ridge regression: minimize ||X_aug B - Y||^2 + lambda * ||B[:-1]||^2
    lambda_ = 0.1
    A = X_aug.T @ X_aug  # (1569, 1569)
    reg = np.eye(D + 1) * lambda_
    reg[-1, -1] = 0.0    # do not regularize the bias term
    A += reg

    # Closed-form solution
    B = np.linalg.solve(A, X_aug.T @ Y)  # (1569, 10)

    # Extract new head weights and bias
    W_new = B[:D, :].T  # (10, 1568)
    b_new = B[D, :]     # (10,)

    # Update the network weights
    w["head.weight"] = W_new.astype(np.float32)
    w["head.bias"] = b_new.astype(np.float32)

    # Diagnostics
    diag = {
        "head_weight_std": float(np.std(w["head.weight"])),
        "head_bias_mean": float(np.mean(w["head.bias"])),
        "lambda": lambda_,
    }
    return {"weights": w, "diagnostics": diag}
