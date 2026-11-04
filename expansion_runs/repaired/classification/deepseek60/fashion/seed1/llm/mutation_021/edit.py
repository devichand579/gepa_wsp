def edit(w, f):
    import numpy as np

    # Extract full feedback features and labels
    X = f["features"].astype(np.float64)          # (2000, 1568)
    y = f["y"].astype(np.int64)                   # (2000,)
    n, d = X.shape

    # Add bias column
    Xb = np.concatenate([X, np.ones((n, 1), dtype=np.float64)], axis=1)  # (2000, 1569)

    # One-hot targets
    Y = np.zeros((n, 10), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    # Ridge least-squares solution (closed form)
    lambda_ = 1e-6
    A = Xb.T @ Xb + lambda_ * np.eye(d + 1)
    W = np.linalg.solve(A, Xb.T @ Y)              # (1569, 10)

    head_weight = W[:-1].T                        # (10, 1568)
    head_bias = W[-1]                             # (10,)

    # Scale logits to reduce cross-entropy (best recent scale)
    scale = 10.0
    head_weight *= scale
    head_bias *= scale

    # Write back into the parameter dictionary
    w["head.weight"] = head_weight.astype(np.float32)
    w["head.bias"] = head_bias.astype(np.float32)

    # Diagnostics: full-pool accuracy and cross-entropy with the new head
    logits = Xb @ W * scale
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)
    acc = np.mean(preds == y)
    ce = -np.mean(np.log(probs[np.arange(n), y] + 1e-12))

    diagnostics = {
        "full_accuracy": float(acc),
        "full_cross_entropy": float(ce),
        "scale": float(scale),
        "lambda": float(lambda_),
        "head_weight_std": float(np.std(head_weight)),
        "head_bias_mean": float(np.mean(head_bias)),
    }

    return {"weights": w, "diagnostics": diagnostics}
