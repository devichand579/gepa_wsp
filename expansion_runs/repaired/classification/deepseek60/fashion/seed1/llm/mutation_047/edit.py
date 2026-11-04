def edit(w, f):
    import numpy as np

    # Read cached features and labels from the entire feedback pool
    X = f["features"].astype(np.float64)          # (2000, 1568)
    y = f["y"].astype(np.int64)                   # (2000,)
    N, D = X.shape
    num_classes = 10

    # One-vs-rest targets: +1 for correct class, -1 for others
    Y = -np.ones((N, num_classes), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Augment features with a bias column
    Xb = np.concatenate([X, np.ones((N, 1), dtype=np.float64)], axis=1)  # (2000, 1569)

    # Closed-form least-squares solution: min ||Xb W - Y||^2
    W, _, _, _ = np.linalg.lstsq(Xb, Y, rcond=None)   # (1569, 10)

    # Extract head weights and bias
    W_head = W[:D, :].T          # (10, 1568)
    b_head = W[D, :]             # (10,)

    # Scale to maximum allowed magnitude (1000) to minimize cross-entropy
    max_abs = max(np.max(np.abs(W_head)), np.max(np.abs(b_head)))
    if max_abs > 0:
        scale = 1000.0 / max_abs
        W_head *= scale
        b_head *= scale

    # Safety clip (should not be needed, but ensures constraint)
    W_head = np.clip(W_head, -1000.0, 1000.0)
    b_head = np.clip(b_head, -1000.0, 1000.0)

    # Assign updated head parameters
    w["head.weight"] = W_head.astype(np.float32)
    w["head.bias"] = b_head.astype(np.float32)

    # Diagnostics: evaluate the new head on the full feedback pool
    logits = X @ W_head.T + b_head
    logits_max = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == y)

    log_probs = logits - logits_max - np.log(np.sum(np.exp(logits - logits_max), axis=1, keepdims=True))
    ce = -np.mean(log_probs[np.arange(N), y])

    diagnostics = {
        "full_pool_accuracy": float(acc),
        "full_pool_cross_entropy": float(ce),
        "max_abs_weight": float(np.max(np.abs(W_head))),
        "max_abs_bias": float(np.max(np.abs(b_head))),
        "scale": float(scale) if max_abs > 0 else 1.0,
        "num_errors": int(np.sum(preds != y))
    }

    return {"weights": w, "diagnostics": diagnostics}
