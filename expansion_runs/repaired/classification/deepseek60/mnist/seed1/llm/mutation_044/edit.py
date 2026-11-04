import numpy as np

def edit(w, f):
    # Extract feedback data
    H = f["features"].astype(np.float32)          # (2000, 128)
    y = f["y"].astype(np.int64)                   # (2000,)
    N, D = H.shape
    num_classes = 10

    # One-hot targets
    Y = np.zeros((N, num_classes), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    # Augment features with bias column
    H_aug = np.hstack([H, np.ones((N, 1), dtype=np.float32)])  # (2000, 129)

    # Ridge regression closed-form: W = (H_aug^T H_aug + lambda I)^-1 H_aug^T Y
    # Choose a moderate lambda to prevent overfitting (2000 samples, 129 params per class)
    lam = 1.0
    A = H_aug.T @ H_aug + lam * np.eye(D + 1, dtype=np.float32)
    B = H_aug.T @ Y
    W_aug = np.linalg.solve(A, B)                 # (129, 10)

    # Extract weights and bias for fc2
    fc2_weight = W_aug[:D, :].T.astype(np.float32)  # (10, 128)
    fc2_bias = W_aug[D, :].astype(np.float32)       # (10,)

    # Clip to safe range
    fc2_weight = np.clip(fc2_weight, -1000.0, 1000.0)
    fc2_bias = np.clip(fc2_bias, -1000.0, 1000.0)

    # Update weights
    w["fc2.weight"] = fc2_weight
    w["fc2.bias"] = fc2_bias

    # Diagnostics
    diagnostics = {
        "fc2_weight_mean": float(fc2_weight.mean()),
        "fc2_weight_std": float(fc2_weight.std()),
        "fc2_bias_mean": float(fc2_bias.mean()),
        "fc2_bias_std": float(fc2_bias.std()),
        "lambda": lam,
    }

    return {"weights": w, "diagnostics": diagnostics}
