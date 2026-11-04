import numpy as np

def edit(w, f):
    # Extract training data
    X = f["features"].astype(np.float64)          # [2000, 128]
    y = f["y"].astype(np.int64)                   # [2000]
    n, d = X.shape
    K = 10

    # Center features (important for conditioning)
    X_mean = X.mean(axis=0, keepdims=True)
    Xc = X - X_mean

    # One-hot targets
    Y = np.zeros((n, K), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    # Ridge regression: solve for W (K x d) and b (K)
    # We augment with a bias column
    X_aug = np.hstack([Xc, np.ones((n, 1))])      # [n, d+1]
    # Solve (X^T X + lambda I) W^T = X^T Y
    lam = 1e-3
    A = X_aug.T @ X_aug
    A += lam * np.eye(d + 1)
    B = X_aug.T @ Y
    W_aug = np.linalg.solve(A, B)                 # [d+1, K]
    W_new = W_aug[:d, :].T                        # [K, d]
    b_new = W_aug[d, :]                           # [K]

    # Adjust bias to account for centering: logits = W_new @ X + b_new
    # But we centered X, so we need to add W_new @ X_mean to bias
    b_new = b_new + W_new @ X_mean.flatten()

    # Write back to weights
    w["fc2.weight"] = W_new.astype(np.float32)
    w["fc2.bias"] = b_new.astype(np.float32)

    # Diagnostics
    # Compute training accuracy
    logits = X @ W_new.T + b_new
    pred = np.argmax(logits, axis=1)
    acc = np.mean(pred == y)
    diagnostics = {
        "train_accuracy": float(acc),
        "fc2_weight_mean": float(w["fc2.weight"].mean()),
        "fc2_weight_std": float(w["fc2.weight"].std()),
        "fc2_bias_mean": float(w["fc2.bias"].mean()),
        "fc2_bias_std": float(w["fc2.bias"].std()),
        "lambda": lam,
    }
    return {"weights": w, "diagnostics": diagnostics}
