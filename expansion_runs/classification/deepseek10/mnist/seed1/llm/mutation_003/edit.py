import numpy as np

def edit(w, f):
    # Extract current parameters
    W1 = w["fc1.weight"].astype(np.float64)
    b1 = w["fc1.bias"].astype(np.float64)
    W2 = w["fc2.weight"].astype(np.float64)
    b2 = w["fc2.bias"].astype(np.float64)

    # Get training features and labels
    H = f["features"].astype(np.float64)  # (2000, 128)
    y = f["y"].astype(np.int64)           # (2000,)

    # One-hot targets
    Y = np.zeros((H.shape[0], 10), dtype=np.float64)
    Y[np.arange(H.shape[0]), y] = 1.0

    # Add bias column to features
    H_aug = np.hstack([H, np.ones((H.shape[0], 1))])  # (2000, 129)

    # Ridge regression: solve for [W2, b2] with small regularization
    # Use lambda = 1e-3 * trace(H_aug^T H_aug) / 129 to scale with feature variance
    HtH = H_aug.T @ H_aug
    trace_HtH = np.trace(HtH)
    lam = 1e-3 * trace_HtH / H_aug.shape[1]
    # Add small epsilon to ensure positive definite
    lam = max(lam, 1e-6)

    # Normal equations: (H^T H + lam I) X = H^T Y
    A = HtH + lam * np.eye(H_aug.shape[1])
    B = H_aug.T @ Y
    # Solve for X (129, 10)
    X = np.linalg.solve(A, B)

    # Extract new W2 and b2
    W2_new = X[:128, :].T  # (10, 128)
    b2_new = X[128, :]     # (10,)

    # Blend with current weights to avoid drastic change: 0.7 new + 0.3 old
    # This acts as a strong prior toward the existing solution.
    alpha = 0.7
    W2_final = alpha * W2_new + (1 - alpha) * W2
    b2_final = alpha * b2_new + (1 - alpha) * b2

    # Clip to safe range
    W2_final = np.clip(W2_final, -1000, 1000).astype(np.float32)
    b2_final = np.clip(b2_final, -1000, 1000).astype(np.float32)

    # Update weights
    w["fc2.weight"] = W2_final
    w["fc2.bias"] = b2_final

    # Diagnostics
    diagnostics = {
        "ridge_lambda": float(lam),
        "blend_alpha": alpha,
        "W2_norm_old": float(np.linalg.norm(W2)),
        "W2_norm_new": float(np.linalg.norm(W2_final)),
        "b2_mean_old": float(np.mean(b2)),
        "b2_mean_new": float(np.mean(b2_final)),
    }

    return {"weights": w, "diagnostics": diagnostics}
