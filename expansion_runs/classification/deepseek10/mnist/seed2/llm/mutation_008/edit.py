import numpy as np

def edit(w, f):
    # Extract full feedback pool
    X = f["x"].reshape(2000, -1).astype(np.float32)  # (2000, 784)
    y = f["y"].astype(np.int64)                       # (2000,)
    n_samples, n_features = X.shape
    n_classes = 10

    # One-hot targets
    Y = np.zeros((n_samples, n_classes), dtype=np.float32)
    Y[np.arange(n_samples), y] = 1.0

    # ---- 1. Ridge linear classifier on raw pixels ----
    X_aug = np.concatenate([X, np.ones((n_samples, 1), dtype=np.float32)], axis=1)  # (2000, 785)
    lambda_lin = 1e-3
    A = X_aug.T @ X_aug + lambda_lin * np.eye(n_features + 1, dtype=np.float32)
    B = X_aug.T @ Y
    W_lin_aug = np.linalg.solve(A, B).T  # (10, 785)
    W_lin = W_lin_aug[:, :n_features]    # (10, 784)
    b_lin = W_lin_aug[:, n_features]     # (10,)

    # Shift so ReLU preserves the linear classifier outputs
    L = X @ W_lin.T + b_lin              # (2000, 10)
    shift = -L.min() + 1.0

    # ---- 2. Build fc1 ----
    fc1_weight = np.zeros((128, 784), dtype=np.float32)
    fc1_bias = np.zeros(128, dtype=np.float32)

    # First 10 units: shifted linear classifier
    fc1_weight[:10, :] = W_lin
    fc1_bias[:10] = b_lin + shift

    # Remaining 118 units: random ReLU features
    rng = np.random.default_rng(42)
    n_random = 128 - 10
    fc1_weight[10:, :] = rng.normal(0.0, 0.1, size=(n_random, 784)).astype(np.float32)
    fc1_bias[10:] = rng.normal(0.0, 0.5, size=n_random).astype(np.float32)

    # ---- 3. Compute hidden features ----
    H = np.maximum(0, X @ fc1_weight.T + fc1_bias)  # (2000, 128)

    # ---- 4. Ridge fit fc2 on hidden features ----
    H_aug = np.concatenate([H, np.ones((n_samples, 1), dtype=np.float32)], axis=1)  # (2000, 129)
    lambda_fc2 = 1e-3
    A2 = H_aug.T @ H_aug + lambda_fc2 * np.eye(129, dtype=np.float32)
    B2 = H_aug.T @ Y
    W2_aug = np.linalg.solve(A2, B2).T  # (10, 129)
    fc2_weight = W2_aug[:, :128]
    fc2_bias = W2_aug[:, 128]

    # ---- 5. Update weights ----
    w["fc1.weight"] = fc1_weight
    w["fc1.bias"] = fc1_bias
    w["fc2.weight"] = fc2_weight
    w["fc2.bias"] = fc2_bias

    # ---- Diagnostics ----
    logits = H @ fc2_weight.T + fc2_bias
    preds = np.argmax(logits, axis=1)
    acc = float(np.mean(preds == y))

    diagnostics = {
        "train_accuracy": acc,
        "shift": float(shift),
        "fc1_weight_norm": float(np.linalg.norm(fc1_weight)),
        "fc1_bias_norm": float(np.linalg.norm(fc1_bias)),
        "fc2_weight_norm": float(np.linalg.norm(fc2_weight)),
        "fc2_bias_norm": float(np.linalg.norm(fc2_bias)),
        "hidden_mean": float(H.mean()),
        "hidden_std": float(H.std()),
        "hidden_zero_fraction": float((H == 0).mean()),
    }

    return {"weights": w, "diagnostics": diagnostics}
