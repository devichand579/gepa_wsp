def edit(w, f):
    import numpy as np

    # Extract cached features and labels from the full feedback pool
    X = f["features"].astype(np.float64)          # (2000, 1568)
    y = f["y"].astype(np.int64)                   # (2000,)
    n, d = X.shape

    # Augment features with a bias column
    X_aug = np.hstack([X, np.ones((n, 1), dtype=np.float64)])  # (2000, 1569)

    # One-hot encode labels
    Y = np.zeros((n, 10), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    # Closed-form least-squares fit (minimum-norm solution)
    W_aug, _, _, _ = np.linalg.lstsq(X_aug, Y, rcond=None)     # (1569, 10)

    # Split into weights and bias
    W = W_aug[:d, :].T          # (10, 1568)
    b = W_aug[d, :]             # (10,)

    # Scale to the maximum allowed magnitude to minimize cross-entropy
    max_abs = max(np.max(np.abs(W)), np.max(np.abs(b)))
    if max_abs > 0:
        scale = 999.0 / max_abs
        W *= scale
        b *= scale

    # Ensure float32, finite, and within bounds
    W = np.clip(W, -1000.0, 1000.0).astype(np.float32)
    b = np.clip(b, -1000.0, 1000.0).astype(np.float32)

    # Write back into the parameter dictionary
    w["head.weight"] = W
    w["head.bias"] = b

    # Diagnostics on the full feedback pool
    logits = X @ W.T + b
    logits_max = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == y)
    eps = 1e-12
    ce = -np.mean(np.log(probs[np.arange(n), y] + eps))

    diagnostics = {
        "full_accuracy": float(acc),
        "cross_entropy": float(ce),
        "max_abs_weight": float(max_abs * scale if max_abs > 0 else 0.0),
        "scale": float(scale if max_abs > 0 else 1.0),
    }

    return {"weights": w, "diagnostics": diagnostics}
