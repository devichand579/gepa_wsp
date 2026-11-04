import numpy as np

def edit(w, f):
    # Extract training data
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float32)  # (2000, 784)
    y = f["y"].astype(np.int64)                                 # (2000,)

    # Center the data
    mean_img = X.mean(axis=0, keepdims=True)                    # (1, 784)
    Xc = X - mean_img

    # Compute top 128 principal components via SVD
    # Xc is (2000, 784). We need right singular vectors (784, 128).
    # Use economy SVD: Xc = U S Vt, Vt is (min(2000,784), 784) = (2000, 784)
    # We only need the first 128 rows of Vt.
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:128, :].astype(np.float32)                 # (128, 784)

    # Set fc1 weights and bias
    w["fc1.weight"][:] = components
    # bias = - components @ mean_img.T so that pre-activation = components @ (x - mean_img)
    w["fc1.bias"][:] = - (components @ mean_img.T).flatten().astype(np.float32)

    # Compute hidden features for all training samples
    # h = relu(components @ Xc.T + bias) = relu(components @ Xc.T - components @ mean_img.T)
    # Actually, pre-activation = components @ (x - mean_img).T
    pre = Xc @ components.T                                     # (2000, 128)
    H = np.maximum(pre, 0.0).astype(np.float32)                 # (2000, 128)

    # Prepare one-hot targets
    Y = np.zeros((y.size, 10), dtype=np.float32)
    Y[np.arange(y.size), y] = 1.0

    # Ridge regression: solve for W2, b2
    # We augment H with a column of ones for bias
    H_aug = np.concatenate([H, np.ones((H.shape[0], 1), dtype=np.float32)], axis=1)  # (2000, 129)
    # Solve (H_aug^T H_aug + lambda I) W = H_aug^T Y
    # Use a small lambda for regularization. We can select lambda via cross-validation,
    # but here we use a fixed reasonable value.
    # To avoid overfitting, we can use a moderate lambda. Let's use lambda = 1.0.
    lam = 1.0
    A = H_aug.T @ H_aug + lam * np.eye(H_aug.shape[1], dtype=np.float32)
    B = H_aug.T @ Y
    W = np.linalg.solve(A, B)                                   # (129, 10)

    # Extract fc2 weights and bias
    w["fc2.weight"][:] = W[:128, :].T.astype(np.float32)        # (10, 128)
    w["fc2.bias"][:] = W[128, :].astype(np.float32)             # (10,)

    # Diagnostics
    # Compute training accuracy with new weights
    logits = H @ w["fc2.weight"].T + w["fc2.bias"]
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == y)

    diagnostics = {
        "train_accuracy": float(acc),
        "lambda": lam,
        "fc1_weight_mean": float(w["fc1.weight"].mean()),
        "fc1_weight_std": float(w["fc1.weight"].std()),
        "fc2_weight_mean": float(w["fc2.weight"].mean()),
        "fc2_weight_std": float(w["fc2.weight"].std()),
        "fc2_bias_mean": float(w["fc2.bias"].mean()),
        "fc2_bias_std": float(w["fc2.bias"].std()),
    }

    return {"weights": w, "diagnostics": diagnostics}
