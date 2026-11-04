import numpy as np

def edit(w, f):
    # Read feedback arrays
    X = f["features"].astype(np.float64)          # [2000,128] current hidden features
    y = f["y"].astype(np.int64)                   # [2000]
    n, d = X.shape
    num_classes = 10

    # Standardize features using training statistics
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True)
    sigma[sigma < 1e-6] = 1.0
    Xs = (X - mu) / sigma

    # One-hot targets
    Y = np.zeros((n, num_classes), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    # Ridge regression: solve for W (num_classes, d) and b (num_classes,)
    # Augment with bias column
    X_aug = np.concatenate([Xs, np.ones((n, 1), dtype=np.float64)], axis=1)  # [n, d+1]
    lam = 1.0  # ridge penalty
    # Closed-form: (X^T X + lam I) W^T = X^T Y
    A = X_aug.T @ X_aug
    A[np.arange(d+1), np.arange(d+1)] += lam
    B = X_aug.T @ Y
    W_aug = np.linalg.solve(A, B)  # [d+1, num_classes]

    W_fc2 = W_aug[:d, :].T          # [num_classes, d]
    b_fc2 = W_aug[d, :]             # [num_classes]

    # Convert back to original feature space:
    # logits = W_fc2 @ ((X - mu)/sigma)^T + b_fc2
    #        = (W_fc2 / sigma) @ X^T + (b_fc2 - W_fc2 @ (mu/sigma)^T)
    W_fc2_orig = W_fc2 / sigma      # broadcasting: [10,128] / [1,128]
    b_fc2_orig = b_fc2 - (W_fc2_orig @ mu.T).ravel()

    # Update weights
    w["fc2.weight"] = W_fc2_orig.astype(np.float32)
    w["fc2.bias"] = b_fc2_orig.astype(np.float32)

    # Keep fc1 unchanged
    # (Optionally, we could also refit fc1, but this is a safe first step.)

    # Diagnostics
    diagnostics = {
        "fc2_weight_mean": float(w["fc2.weight"].mean()),
        "fc2_weight_std": float(w["fc2.weight"].std()),
        "fc2_bias_mean": float(w["fc2.bias"].mean()),
        "fc2_bias_std": float(w["fc2.bias"].std()),
        "lambda": lam,
        "feature_mean_norm": float(np.linalg.norm(mu)),
        "feature_std_mean": float(sigma.mean()),
    }
    return {"weights": w, "diagnostics": diagnostics}
