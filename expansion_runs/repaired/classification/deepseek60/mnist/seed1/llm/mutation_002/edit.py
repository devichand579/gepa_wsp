import numpy as np

def edit(w, f):
    # Extract full-pool hidden features and labels
    H = f["features"].astype(np.float64)   # (2000, 128)
    y = f["y"].astype(np.int64)            # (2000,)
    n, d = H.shape
    n_classes = 10

    # One-hot encode labels
    Y = np.zeros((n, n_classes), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    # Append bias column
    H_aug = np.concatenate([H, np.ones((n, 1), dtype=np.float64)], axis=1)  # (2000, 129)

    # Closed-form ridge regression: W = (H_aug^T H_aug + lambda I)^-1 H_aug^T Y
    lam = 1e-3
    A = H_aug.T @ H_aug + lam * np.eye(d + 1)
    B = H_aug.T @ Y
    W = np.linalg.solve(A, B)  # (129, 10)

    # Extract fc2 parameters
    fc2_weight = W[:d, :].T   # (10, 128)
    fc2_bias = W[d, :]        # (10,)

    # Update weights
    w["fc2.weight"] = fc2_weight.astype(np.float32)
    w["fc2.bias"] = fc2_bias.astype(np.float32)

    # Enforce constraints
    w["fc2.weight"] = np.clip(w["fc2.weight"], -1000, 1000)
    w["fc2.bias"] = np.clip(w["fc2.bias"], -1000, 1000)

    # Diagnostics
    logits = H @ w["fc2.weight"].T + w["fc2.bias"]
    preds = np.argmax(logits, axis=1)
    acc = float(np.mean(preds == y))

    diagnostics = {
        "train_accuracy": acc,
        "fc2_weight_mean": float(np.mean(w["fc2.weight"])),
        "fc2_weight_std": float(np.std(w["fc2.weight"])),
        "fc2_bias_mean": float(np.mean(w["fc2.bias"])),
        "fc2_bias_std": float(np.std(w["fc2.bias"])),
        "lambda": lam,
    }

    return {"weights": w, "diagnostics": diagnostics}
