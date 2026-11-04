def edit(w, f):
    import numpy as np

    # Extract full feedback features and labels
    X = f["features"].astype(np.float64)          # (2000, 128)
    y = f["y"].astype(np.int64)                   # (2000,)
    N = X.shape[0]

    # One-hot targets
    Y = np.zeros((N, 10), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Add bias column
    X_aug = np.hstack([X, np.ones((N, 1))])       # (2000, 129)

    # Ridge regression with lambda=1e-3, no regularisation on bias
    lam = 1e-3
    R = np.eye(129) * lam
    R[128, 128] = 0.0

    # Solve (X_aug^T X_aug + R) W = X_aug^T Y
    A = X_aug.T @ X_aug + R
    B = X_aug.T @ Y
    W_aug = np.linalg.solve(A, B)                 # (129, 10)

    fc2_weight = W_aug[:128].T.astype(np.float32) # (10, 128)
    fc2_bias   = W_aug[128].astype(np.float32)    # (10,)

    # Update only the final classifier
    w["fc2.weight"] = fc2_weight
    w["fc2.bias"]   = fc2_bias

    # Diagnostics
    logits_new = X @ fc2_weight.T + fc2_bias
    preds = np.argmax(logits_new, axis=1)
    acc = float(np.mean(preds == y))

    probs = np.exp(logits_new - logits_new.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    ce = float(-np.mean(np.log(probs[np.arange(N), y] + 1e-12)))

    diagnostics = {
        "train_acc": acc,
        "train_ce": ce,
        "lambda": lam,
        "fc2_weight_mean": float(fc2_weight.mean()),
        "fc2_weight_std": float(fc2_weight.std()),
        "fc2_bias_mean": float(fc2_bias.mean()),
        "fc2_bias_std": float(fc2_bias.std()),
    }

    return {"weights": w, "diagnostics": diagnostics}
