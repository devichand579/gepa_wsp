import numpy as np

def edit(w, f):
    # Extract current parameters
    W1 = w["fc1.weight"].astype(np.float64)  # [128, 784]
    b1 = w["fc1.bias"].astype(np.float64)    # [128]
    W2 = w["fc2.weight"].astype(np.float64)  # [10, 128]
    b2 = w["fc2.bias"].astype(np.float64)    # [10]

    # Full training pool
    X = f["x"].astype(np.float64)            # [2000, 1, 28, 28]
    y = f["y"].astype(np.int64)              # [2000]
    N = X.shape[0]
    X_flat = X.reshape(N, -1)                # [2000, 784]

    # One-hot targets
    Y = np.zeros((N, 10), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Compute desired hidden features H such that H @ W2.T + b2 ≈ Y
    # Minimum-norm solution: H = (Y - b2) @ pinv(W2.T)
    # W2.T is [128, 10], pinv is [10, 128]
    W2_pinv = np.linalg.pinv(W2.T)           # [10, 128]
    H_target = (Y - b2) @ W2_pinv            # [2000, 128]

    # Make targets non-negative to match ReLU output
    H_target = np.maximum(H_target, 0.0)

    # Fit fc1 to map X_flat to H_target with ridge regression
    # Add bias column to X_flat
    X_aug = np.hstack([X_flat, np.ones((N, 1))])  # [2000, 785]
    # Solve for W1_aug [128, 785] such that X_aug @ W1_aug.T ≈ H_target
    # Ridge: W1_aug.T = (X_aug.T @ X_aug + λ I)^-1 @ X_aug.T @ H_target
    # Choose λ by simple heuristic: small value to avoid singularity
    # We can use a fixed small ridge, e.g., 1e-3
    lam = 1e-3
    XtX = X_aug.T @ X_aug
    XtX_reg = XtX + lam * np.eye(XtX.shape[0])
    XtH = X_aug.T @ H_target
    W1_aug_T = np.linalg.solve(XtX_reg, XtH)  # [785, 128]
    W1_aug = W1_aug_T.T                       # [128, 785]

    # Split back into weight and bias
    new_W1 = W1_aug[:, :784]
    new_b1 = W1_aug[:, 784]

    # Update weights
    w["fc1.weight"] = new_W1.astype(np.float32)
    w["fc1.bias"] = new_b1.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "fc1_weight_norm": float(np.linalg.norm(new_W1)),
        "fc1_bias_norm": float(np.linalg.norm(new_b1)),
        "target_hidden_mean": float(np.mean(H_target)),
        "target_hidden_std": float(np.std(H_target)),
        "ridge_lambda": lam
    }

    return {"weights": w, "diagnostics": diagnostics}
