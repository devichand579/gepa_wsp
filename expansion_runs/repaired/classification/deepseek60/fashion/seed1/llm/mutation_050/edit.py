import numpy as np

def edit(w, f):
    X = f["features"].astype(np.float64)
    y = f["y"].astype(np.int64)
    N = X.shape[0]

    # One-hot encode labels
    Y = np.zeros((N, 10), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Augment features with a bias column
    X_aug = np.hstack([X, np.ones((N, 1))])

    # Closed-form least-squares solution for the linear head
    W_aug, _, _, _ = np.linalg.lstsq(X_aug, Y, rcond=None)
    W = W_aug[:-1].T  # shape (10, 1568)
    b = W_aug[-1]     # shape (10,)

    # Scale to maximum allowed magnitude
    max_abs = max(np.abs(W).max(), np.abs(b).max())
    scale = 1000.0 / max_abs if max_abs > 0 else 1.0
    W = W * scale
    b = b * scale

    # Clip to ensure within bounds
    W = np.clip(W, -1000.0, 1000.0)
    b = np.clip(b, -1000.0, 1000.0)

    # Update the head weights
    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

    # Diagnostics
    logits = X @ W.T + b
    logits_max = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    ce = -np.mean(np.log(probs[np.arange(N), y] + 1e-30))
    acc = np.mean(np.argmax(probs, axis=1) == y)

    diagnostics = {
        "full_pool_accuracy": float(acc),
        "full_pool_cross_entropy": float(ce),
        "max_abs_weight": float(np.abs(w["head.weight"]).max()),
        "max_abs_bias": float(np.abs(w["head.bias"]).max()),
        "scale": float(scale)
    }

    return {"weights": w, "diagnostics": diagnostics}
