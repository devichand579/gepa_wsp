import numpy as np

def edit(w, f):
    # Extract full feedback data
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float32)  # (2000, 784)
    y = f["y"].astype(np.int64)                                 # (2000,)
    N, D = X.shape
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    # Add bias column
    X_aug = np.hstack([X, np.ones((N, 1), dtype=np.float32)])   # (2000, 785)

    # Ridge regression: solve (X_aug^T X_aug + lambda I) W_aug = X_aug^T Y
    lam = 1.0
    A = X_aug.T @ X_aug + lam * np.eye(D + 1, dtype=np.float32)
    B = X_aug.T @ Y
    W_aug = np.linalg.solve(A, B)   # (785, 10)

    W = W_aug[:D, :].T              # (10, 784)
    b = W_aug[D, :]                 # (10,)

    # Scale W and b so that min_x (W_c · x) > -C for all classes c,
    # where C is the constant bias we will add in fc1.
    C = 900.0
    # For each class, the minimum of W_c · x over x in [0,1]^784 is sum of negative entries.
    neg_sum = np.sum(np.where(W < 0, W, 0.0), axis=1)  # (10,)
    M = np.max(-neg_sum) if np.any(neg_sum < 0) else 0.0
    if M > C - 100.0:  # leave some margin
        scale = (C - 100.0) / M
        W *= scale
        b *= scale

    # Build new fc1 and fc2
    fc1_weight = np.zeros((128, 784), dtype=np.float32)
    fc1_bias = np.zeros(128, dtype=np.float32)
    fc2_weight = np.zeros((10, 128), dtype=np.float32)
    fc2_bias = np.zeros(10, dtype=np.float32)

    # Use first 10 hidden units for the linear classifier
    fc1_weight[:10, :] = W
    fc1_bias[:10] = C
    fc2_weight[:, :10] = np.eye(10, dtype=np.float32)
    fc2_bias[:] = b - C

    # Remaining units are zero (ReLU(0) = 0), so they don't affect logits.

    # Write back to w
    w["fc1.weight"] = fc1_weight
    w["fc1.bias"] = fc1_bias
    w["fc2.weight"] = fc2_weight
    w["fc2.bias"] = fc2_bias

    # Diagnostics
    diagnostics = {
        "ridge_lambda": lam,
        "scale_factor": float(scale) if 'scale' in locals() else 1.0,
        "max_neg_sum": float(M),
        "fc1_bias_const": C,
        "fc2_bias_min": float(np.min(fc2_bias)),
        "fc2_bias_max": float(np.max(fc2_bias)),
        "fc1_weight_norm": float(np.linalg.norm(fc1_weight)),
        "fc2_weight_norm": float(np.linalg.norm(fc2_weight)),
    }
    return {"weights": w, "diagnostics": diagnostics}
