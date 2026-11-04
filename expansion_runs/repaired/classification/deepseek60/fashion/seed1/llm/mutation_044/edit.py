import numpy as np

def edit(w, f):
    # Extract full-pool features and labels
    X = f["features"].astype(np.float64)          # (2000, 1568)
    y = f["y"].astype(np.int64)                   # (2000,)
    N, D = X.shape
    num_classes = 10

    # Augment with bias column
    A = np.hstack([X, np.ones((N, 1), dtype=np.float64)])  # (2000, 1569)

    # Small ridge for numerical stability
    lam = 1e-3
    I = np.eye(D + 1, dtype=np.float64)

    # Precompute A^T A + lam I
    AtA = A.T @ A + lam * I

    # Solve for each class
    W = np.zeros((num_classes, D), dtype=np.float64)
    b = np.zeros(num_classes, dtype=np.float64)

    for c in range(num_classes):
        t = np.where(y == c, 1.0, -1.0).astype(np.float64)
        Atb = A.T @ t
        wc = np.linalg.solve(AtA, Atb)
        W[c] = wc[:-1]
        b[c] = wc[-1]

    # Scale to maximum allowed absolute value
    max_abs = max(np.max(np.abs(W)), np.max(np.abs(b)))
    if max_abs > 0:
        scale = 1000.0 / max_abs
        W *= scale
        b *= scale

    # Clip to ensure <= 1000 (should already be exactly 1000 max)
    W = np.clip(W, -1000.0, 1000.0)
    b = np.clip(b, -1000.0, 1000.0)

    # Update weights
    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

    # Optional diagnostics: compute paired-batch CE using provided head input
    try:
        Xb = f["layer.head.input0"].astype(np.float64)   # (128, 1568)
        yb = f["batch.y"].astype(np.int64)               # (128,)
        logits = Xb @ W.T + b
        # stable softmax
        logits_max = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        ce = -np.mean(np.log(probs[np.arange(len(yb)), yb] + 1e-12))
        acc = np.mean(np.argmax(logits, axis=1) == yb)
        diag = {
            "paired_batch_accuracy": float(acc),
            "paired_batch_cross_entropy": float(ce),
            "max_abs_weight": float(np.max(np.abs(W))),
            "max_abs_bias": float(np.max(np.abs(b))),
        }
        return {"weights": w, "diagnostics": diag}
    except Exception:
        return w
