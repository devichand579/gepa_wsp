import numpy as np

def edit(w, f):
    # Read current weights
    fc1_w = w["fc1.weight"].astype(np.float32)
    fc1_b = w["fc1.bias"].astype(np.float32)
    fc2_w = w["fc2.weight"].astype(np.float32)
    fc2_b = w["fc2.bias"].astype(np.float32)

    # Feedback data
    X = f["features"].astype(np.float32)          # [2000, 128]
    y = f["y"].astype(np.int64)                   # [2000]
    n_samples, n_features = X.shape
    n_classes = 10

    # One-hot targets
    Y = np.zeros((n_samples, n_classes), dtype=np.float32)
    Y[np.arange(n_samples), y] = 1.0

    # Center features to improve conditioning
    X_mean = X.mean(axis=0, keepdims=True)
    Xc = X - X_mean

    # Ridge regression toward current fc2 weights.
    # We solve for a correction dW such that new_W = fc2_w + dW.
    # The objective is ||Xc @ (fc2_w + dW)^T + (fc2_b + db) - Y||^2 + lambda * (||dW||^2 + ||db||^2)
    # This is equivalent to ridge regression on the residuals.
    # Compute current predictions (logits) on centered features:
    # logits = Xc @ fc2_w.T + (fc2_b + X_mean @ fc2_w.T)
    # But we can just compute the residual targets:
    # R = Y - (Xc @ fc2_w.T + fc2_b + X_mean @ fc2_w.T)
    # Actually, we want to fit the residual. Let's use the standard ridge solution for the full weights
    # with a prior centered at the current weights.
    # The solution for W (including bias) is:
    # W = (X^T X + lambda I)^-1 (X^T Y + lambda W0)
    # where X is augmented with a column of ones.
    # We'll use a small lambda to keep changes small.
    lambda_reg = 1e-3  # small regularization toward current weights

    # Augment features with bias column
    X_aug = np.hstack([X, np.ones((n_samples, 1), dtype=np.float32)])  # [2000, 129]
    # Current weights augmented: fc2_w is [10, 128], fc2_b is [10]. We need [10, 129]
    W0 = np.hstack([fc2_w, fc2_b.reshape(-1, 1)])  # [10, 129]

    # Compute normal equations
    # We want to minimize ||X_aug @ W.T - Y||^2 + lambda * ||W - W0||^2
    # Solution: W = (X_aug^T X_aug + lambda I)^-1 (X_aug^T Y + lambda W0)
    # Note: Y is [2000, 10], so X_aug^T Y is [129, 10]. W0 is [10, 129], so we need to transpose.
    XtX = X_aug.T @ X_aug  # [129, 129]
    XtY = X_aug.T @ Y      # [129, 10]
    I = np.eye(129, dtype=np.float32)
    # Solve for W.T (which is [129, 10])
    # (XtX + lambda I) W.T = XtY + lambda W0.T
    A = XtX + lambda_reg * I
    B = XtY + lambda_reg * W0.T
    W_new_T = np.linalg.solve(A, B)  # [129, 10]
    W_new = W_new_T.T  # [10, 129]

    # Extract new fc2 weights and bias
    new_fc2_w = W_new[:, :128].astype(np.float32)
    new_fc2_b = W_new[:, 128].astype(np.float32)

    # Update weights
    w["fc2.weight"] = new_fc2_w
    w["fc2.bias"] = new_fc2_b

    # Diagnostics
    diagnostics = {
        "lambda_reg": float(lambda_reg),
        "fc2_weight_norm": float(np.linalg.norm(new_fc2_w)),
        "fc2_bias_norm": float(np.linalg.norm(new_fc2_b)),
        "fc2_weight_mean": float(new_fc2_w.mean()),
        "fc2_weight_std": float(new_fc2_w.std()),
    }

    return {"weights": w, "diagnostics": diagnostics}
