def edit(w, f):
    X = f["features"].astype(np.float64)  # (2000, 1568)
    y = f["y"].astype(np.int64)
    N, D = X.shape
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Augment features with a bias column
    X_aug = np.hstack([X, np.ones((N, 1), dtype=np.float64)])

    # Ridge regularization (do not regularize the bias term)
    lam = 1e-4
    reg = np.eye(D + 1, dtype=np.float64) * lam
    reg[-1, -1] = 0.0

    # Closed-form ridge solution: (X^T X + reg) W = X^T Y
    A = X_aug.T @ X_aug + reg
    B = X_aug.T @ Y
    W_aug = np.linalg.solve(A, B)

    W = W_aug[:-1, :].T  # (10, 1568)
    b = W_aug[-1, :]     # (10,)

    # Enforce the absolute value constraint
    W = np.clip(W, -1000.0, 1000.0)
    b = np.clip(b, -1000.0, 1000.0)

    # Update only the head parameters
    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

    # Diagnostics
    logits = X @ W.T + b
    logits_max = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)
    acc = (preds == y).mean()
    ce = -np.log(probs[np.arange(N), y] + 1e-12).mean()

    diag = {
        "train_accuracy": float(acc),
        "train_cross_entropy": float(ce),
        "max_abs_head_weight": float(np.abs(W).max()),
        "max_abs_head_bias": float(np.abs(b).max()),
        "lambda": lam
    }

    return {"weights": w, "diagnostics": diag}
