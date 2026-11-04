import numpy as np

def edit(w, f):
    # Extract full training features and labels
    X = f["features"].astype(np.float32)          # (2000, 1568)
    y = f["y"].astype(np.int64)                   # (2000,)
    N, D = X.shape
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    # Standardize features (zero mean, unit variance) for better conditioning
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-8
    Xs = (X - mu) / sigma

    # Augment with bias column
    Xa = np.hstack([Xs, np.ones((N, 1), dtype=np.float32)])  # (N, D+1)

    # Ridge regression: W = (Xa^T Xa + λI)^-1 Xa^T Y
    lam = 1.0
    A = Xa.T @ Xa + lam * np.eye(D + 1, dtype=np.float32)
    b = Xa.T @ Y
    W_aug = np.linalg.solve(A, b)                 # (D+1, C)

    # Extract head weights and bias, then undo standardization
    W = W_aug[:D, :].T                            # (C, D)
    bias = W_aug[D, :]                            # (C,)
    # Adjust for standardization: logits = (Xs @ W.T) + bias
    # Xs = (X - mu) / sigma  =>  X @ (W / sigma).T + (bias - (mu/sigma) @ W.T)
    W_final = W / sigma                           # (C, D)
    bias_final = bias - (mu / sigma) @ W.T        # (C,)

    # Write back to the network weights
    w["head.weight"] = W_final.astype(np.float32)
    w["head.bias"] = bias_final.astype(np.float32)

    # Diagnostics
    logits = X @ W_final.T + bias_final
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == y)
    diagnostics = {
        "train_accuracy": float(acc),
        "ridge_lambda": lam,
        "feature_mean_abs": float(np.mean(np.abs(mu))),
        "feature_std_mean": float(np.mean(sigma)),
    }
    return {"weights": w, "diagnostics": diagnostics}
