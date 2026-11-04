import numpy as np

def edit(w, f):
    # Extract full training pool
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)  # (2000, 784)
    y = f["y"].astype(np.int64)                                  # (2000,)
    N, D = X.shape
    K = 10

    # Center the data
    mean = X.mean(axis=0)
    Xc = X - mean

    # PCA via SVD: take top 128 components
    n_components = 128
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    W1 = Vt[:n_components, :]          # (128, 784)
    b1 = -W1 @ mean                    # (128,)

    # Compute hidden activations with ReLU
    H = X @ W1.T + b1                  # (2000, 128)
    H = np.maximum(H, 0)

    # One-hot labels
    Y = np.zeros((N, K), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Ridge regression for the second layer (with bias)
    H_aug = np.hstack([H, np.ones((N, 1))])   # (2000, 129)
    lambda_reg = 1e-3
    A = H_aug.T @ H_aug + lambda_reg * np.eye(H_aug.shape[1])
    B = np.linalg.solve(A, H_aug.T @ Y)       # (129, 10)

    W2 = B[:n_components, :].T                # (10, 128)
    b2 = B[n_components, :]                   # (10,)

    # Update the network weights
    w["fc1.weight"] = W1.astype(np.float32)
    w["fc1.bias"]   = b1.astype(np.float32)
    w["fc2.weight"] = W2.astype(np.float32)
    w["fc2.bias"]   = b2.astype(np.float32)

    # Diagnostics
    logits = H @ W2.T + b2
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == y)

    diagnostics = {
        "train_accuracy": float(acc),
        "fc1_weight_norm": float(np.linalg.norm(w["fc1.weight"])),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "n_components": n_components,
        "lambda_reg": lambda_reg
    }

    return {"weights": w, "diagnostics": diagnostics}
