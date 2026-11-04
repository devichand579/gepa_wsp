import numpy as np

def edit(w, f):
    # Extract current weights
    W1 = w["fc1.weight"].astype(np.float32)
    b1 = w["fc1.bias"].astype(np.float32)
    W2 = w["fc2.weight"].astype(np.float32)
    b2 = w["fc2.bias"].astype(np.float32)

    # Get full training data and labels
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float32)  # (2000, 784)
    y = f["y"].astype(np.int64)                                  # (2000,)

    # Compute current hidden features for all training samples
    # Z = X @ W1.T + b1
    Z = X @ W1.T + b1
    H = np.maximum(Z, 0.0)  # ReLU

    # One-hot encode labels
    N = X.shape[0]
    num_classes = 10
    Y = np.zeros((N, num_classes), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    # Ridge regression for fc2: minimize || H @ W2.T + b2 - Y ||^2 + lambda * ||W2||^2
    # We solve for [W2, b2] by augmenting H with a column of ones.
    # Let H_aug = [H, 1] of shape (N, 129). Then logits = H_aug @ theta.T, where theta = [W2, b2].
    H_aug = np.hstack([H, np.ones((N, 1), dtype=np.float32)])
    lambda_ = 1e-3  # small ridge penalty

    # Solve for each class independently: (H_aug^T H_aug + lambda I) theta_c = H_aug^T Y_c
    # Use lstsq for numerical stability (handles singular matrices).
    # We can solve all classes at once by solving H_aug @ Theta = Y, where Theta is (129, 10).
    # Regularized least squares: minimize || H_aug @ Theta - Y ||^2 + lambda * ||Theta||^2
    # Equivalent to solving [H_aug; sqrt(lambda)*I] @ Theta = [Y; 0]
    A = np.vstack([H_aug, np.sqrt(lambda_) * np.eye(H_aug.shape[1], dtype=np.float32)])
    B = np.vstack([Y, np.zeros((H_aug.shape[1], num_classes), dtype=np.float32)])
    Theta, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

    # Extract new fc2 weights and bias
    W2_new = Theta[:-1, :].T  # shape (10, 128)
    b2_new = Theta[-1, :]     # shape (10,)

    # Update weights
    w["fc2.weight"] = W2_new.astype(np.float32)
    w["fc2.bias"] = b2_new.astype(np.float32)

    # Diagnostics
    # Compute new logits and accuracy on training data
    logits_new = H @ W2_new.T + b2_new
    preds = np.argmax(logits_new, axis=1)
    acc = np.mean(preds == y)

    diagnostics = {
        "train_accuracy": float(acc),
        "lambda": float(lambda_),
        "fc2_weight_mean": float(np.mean(W2_new)),
        "fc2_weight_std": float(np.std(W2_new)),
        "fc2_bias_mean": float(np.mean(b2_new)),
        "fc2_bias_std": float(np.std(b2_new)),
    }

    return {"weights": w, "diagnostics": diagnostics}
