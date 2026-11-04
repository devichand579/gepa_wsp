import numpy as np

def edit(w, f):
    # Extract frozen features and labels from the full feedback pool
    X = f["features"].astype(np.float64)  # (2000, 1568)
    y = f["y"].astype(np.int64)           # (2000,)
    N, D = X.shape
    C = 10

    # One-hot encode labels
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Augment features with a bias column
    X_aug = np.hstack([X, np.ones((N, 1))])  # (2000, 1569)

    # Ridge regression with lambda = 1e-4 (do not regularize bias)
    lam = 1e-4
    reg = lam * np.eye(D + 1)
    reg[-1, -1] = 0.0
    A = X_aug.T @ X_aug + reg
    B = X_aug.T @ Y
    W_aug = np.linalg.solve(A, B)  # (1569, 10)

    W = W_aug[:-1, :].T  # (10, 1568)
    b = W_aug[-1, :]     # (10,)

    # Clip to allowed range
    W = np.clip(W, -1000.0, 1000.0)
    b = np.clip(b, -1000.0, 1000.0)

    # Update only the head parameters
    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

    # Diagnostics
    logits = X @ W.T + b
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == y)

    logits_max = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits_max)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    ce = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))

    diag = {
        "train_accuracy": float(acc),
        "train_cross_entropy": float(ce),
        "max_abs_head_weight": float(np.max(np.abs(W))),
        "max_abs_head_bias": float(np.max(np.abs(b))),
        "lambda": lam
    }

    return {"weights": w, "diagnostics": diag}
