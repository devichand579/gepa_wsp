def edit(w, f):
    import numpy as np

    # Extract full feedback features and labels
    X = f["features"].astype(np.float64)  # (2000, 1568)
    y = f["y"].astype(np.int64)           # (2000,)
    N, D = X.shape
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Augment features with a bias column
    X_aug = np.hstack([X, np.ones((N, 1), dtype=np.float64)])  # (2000, 1569)

    # Closed-form least squares solution (no autograd, no iterative optimization)
    Theta, _, _, _ = np.linalg.lstsq(X_aug, Y, rcond=None)  # (1569, 10)

    W = Theta[:-1].T  # (10, 1568)
    b = Theta[-1]     # (10,)

    # Scale weights and bias to maximize logit separation while respecting bounds
    max_w = np.max(np.abs(W))
    s = 999.0 / max_w if max_w > 0 else 1.0
    W *= s
    b *= s

    # Center bias (softmax is invariant to adding a constant to all logits)
    b -= np.mean(b)

    # Ensure max absolute bias <= 1000
    max_b = np.max(np.abs(b))
    if max_b > 1000.0:
        s2 = 1000.0 / max_b
        W *= s2
        b *= s2

    # Safety clip to the allowed range
    W = np.clip(W, -1000.0, 1000.0)
    b = np.clip(b, -1000.0, 1000.0)

    # Write back into the network weights
    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

    # Diagnostics on the full feedback pool
    logits = X @ W.T + b
    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == y)
    ce = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))

    diag = {
        "full_accuracy": float(acc),
        "cross_entropy": float(ce),
        "max_abs_weight": float(np.max(np.abs(W))),
        "max_abs_bias": float(np.max(np.abs(b))),
        "scale": float(s)
    }

    return {"weights": w, "diagnostics": diag}
