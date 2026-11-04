import numpy as np

def edit(w, f):
    # Extract feedback features and labels
    H = f["features"].astype(np.float64)          # [2000, 128]
    y = f["y"].astype(np.int64)                   # [2000]
    n, d = H.shape

    # One-hot targets
    Y = np.zeros((n, 10), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    # Augment features with bias column
    H_aug = np.hstack([H, np.ones((n, 1), dtype=np.float64)])  # [2000, 129]

    # Current fc2 parameters (augmented)
    W_old = w["fc2.weight"].astype(np.float64)    # [10, 128]
    b_old = w["fc2.bias"].astype(np.float64)      # [10]
    W_aug_old = np.hstack([W_old, b_old[:, None]])  # [10, 129]

    # Ridge update with prior centered at current weights
    # Minimize ||H_aug W^T - Y||^2 + lambda ||W - W_old||^2
    lambda_reg = 1000.0
    A = H_aug.T @ H_aug + lambda_reg * np.eye(d + 1)   # [129, 129]
    B = H_aug.T @ (Y - H_aug @ W_aug_old.T)            # [129, 10]
    delta = np.linalg.solve(A, B)                      # [129, 10]
    W_aug_new = W_aug_old + delta.T                    # [10, 129]

    # Split back
    w["fc2.weight"] = W_aug_new[:, :d].astype(np.float32)
    w["fc2.bias"]   = W_aug_new[:, d].astype(np.float32)

    # Diagnostics
    diagnostics = {
        "lambda": lambda_reg,
        "fc2_weight_mean": float(w["fc2.weight"].mean()),
        "fc2_weight_std": float(w["fc2.weight"].std()),
        "fc2_bias_mean": float(w["fc2.bias"].mean()),
        "fc2_bias_std": float(w["fc2.bias"].std()),
        "fc2_weight_change_norm": float(np.linalg.norm(w["fc2.weight"] - W_old.astype(np.float32))),
        "fc2_bias_change_norm": float(np.linalg.norm(w["fc2.bias"] - b_old.astype(np.float32))),
    }
    return {"weights": w, "diagnostics": diagnostics}
