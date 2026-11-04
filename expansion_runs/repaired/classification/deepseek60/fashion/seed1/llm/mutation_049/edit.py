import numpy as np

def edit(w, f):
    # Extract cached final hidden features and labels
    X = f["features"].astype(np.float64)  # (2000, 1568)
    y = f["y"].astype(np.int64)           # (2000,)
    N, D = X.shape
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Augment features with a bias column
    X_aug = np.concatenate([X, np.ones((N, 1))], axis=1)  # (2000, 1569)

    # Closed-form least-squares solution: X_aug @ W_aug = Y
    W_aug, residuals, rank, s = np.linalg.lstsq(X_aug, Y, rcond=None)

    W = W_aug[:-1, :].T  # (10, 1568)
    b = W_aug[-1, :]     # (10,)

    # Scale to the maximum allowed magnitude (1000) while keeping both weights and biases <= 1000
    max_w = np.max(np.abs(W))
    max_b = np.max(np.abs(b))
    max_val = max(max_w, max_b)
    if max_val > 0:
        scale = 1000.0 / max_val
        W *= scale
        b *= scale
    else:
        scale = 1.0

    # Write back as float32
    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

    # Diagnostics on the full feedback pool
    logits = X @ W.T + b
    pred = np.argmax(logits, axis=1)
    acc = np.mean(pred == y)

    logits_max = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    ce = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))

    diagnostics = {
        "full_pool_accuracy": float(acc),
        "full_pool_cross_entropy": float(ce),
        "max_abs_weight": float(np.max(np.abs(w["head.weight"]))),
        "max_abs_bias": float(np.max(np.abs(w["head.bias"]))),
        "scale": float(scale)
    }
    return {"weights": w, "diagnostics": diagnostics}
