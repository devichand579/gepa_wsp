import numpy as np

def edit(w, f):
    # Extract cached final hidden features and labels
    X = f["features"].astype(np.float64)          # (N, D)
    y = f["y"].astype(np.int64)                   # (N,)
    N, D = X.shape

    # Add intercept column
    X_aug = np.concatenate([X, np.ones((N, 1), dtype=np.float64)], axis=1)  # (N, D+1)

    # One-hot targets
    Y = np.zeros((N, 10), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Initial ridge regression (closed-form normal equations)
    lam = 1e-3
    A = X_aug.T @ X_aug + lam * np.eye(D + 1)
    B = X_aug.T @ Y
    W_aug = np.linalg.solve(A, B)                 # (D+1, 10)

    # Check training accuracy
    logits = X_aug @ W_aug
    pred = np.argmax(logits, axis=1)
    acc = np.mean(pred == y)

    # If not perfect, perform a weighted ridge fit emphasising misclassified samples
    if acc < 1.0:
        weights = np.ones(N, dtype=np.float64)
        mis = pred != y
        weights[mis] = 1000.0
        A_w = X_aug.T @ (weights[:, None] * X_aug) + lam * np.eye(D + 1)
        B_w = X_aug.T @ (weights[:, None] * Y)
        W_aug = np.linalg.solve(A_w, B_w)
        logits = X_aug @ W_aug
        pred = np.argmax(logits, axis=1)
        acc = np.mean(pred == y)

        # If still not perfect, try a much larger weight
        if acc < 1.0:
            weights[mis] = 1e6
            A_w = X_aug.T @ (weights[:, None] * X_aug) + lam * np.eye(D + 1)
            B_w = X_aug.T @ (weights[:, None] * Y)
            W_aug = np.linalg.solve(A_w, B_w)
            logits = X_aug @ W_aug
            pred = np.argmax(logits, axis=1)
            acc = np.mean(pred == y)

    # Extract head parameters
    W = W_aug[:-1].T          # (10, D)
    b = W_aug[-1]             # (10,)

    # Scale to maximum allowed magnitude if we achieved perfect training accuracy
    if acc == 1.0:
        max_abs = max(np.max(np.abs(W)), np.max(np.abs(b)))
        if max_abs > 0:
            scale = 1000.0 / max_abs
            W *= scale
            b *= scale

    # Clamp and cast to float32
    W = np.clip(W, -1000.0, 1000.0).astype(np.float32)
    b = np.clip(b, -1000.0, 1000.0).astype(np.float32)

    # Write back into the parameter dictionary
    w["head.weight"] = W
    w["head.bias"] = b

    diagnostics = {
        "accuracy": float(acc),
        "max_abs_weight": float(np.max(np.abs(W))),
        "max_abs_bias": float(np.max(np.abs(b))),
        "num_misclassified": int(np.sum(pred != y)),
    }
    return {"weights": w, "diagnostics": diagnostics}
