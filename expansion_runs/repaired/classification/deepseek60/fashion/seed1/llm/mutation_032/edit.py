import numpy as np

def edit(w, f):
    X = f["features"].astype(np.float64)          # [2000, 1568]
    y = f["y"].astype(np.int64)                   # [2000]
    N, D = X.shape
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Center features and targets for an intercept term
    X_mean = X.mean(axis=0, keepdims=True)
    Y_mean = Y.mean(axis=0, keepdims=True)
    Xc = X - X_mean
    Yc = Y - Y_mean

    # Ridge regression: W = (Xc^T Xc + lambda I)^-1 Xc^T Yc
    # Use a small lambda to allow near-perfect separation while avoiding
    # numerical issues.  The value is chosen conservatively.
    lam = 1e-3
    A = Xc.T @ Xc + lam * np.eye(D, dtype=np.float64)
    B = Xc.T @ Yc
    W = np.linalg.solve(A, B)                     # [1568, 10]
    b = (Y_mean - X_mean @ W).ravel()             # [10]

    # Transpose to PyTorch linear layout [out, in]
    W = W.T                                       # [10, 1568]

    # Rescale weights and bias so that the maximum absolute weight is 1000.
    # This preserves the decision boundaries but increases confidence.
    max_abs = np.max(np.abs(W))
    if max_abs > 0:
        scale = 1000.0 / max_abs
        W = W * scale
        b = b * scale

    # Clip to the allowed range just in case
    W = np.clip(W, -1000.0, 1000.0).astype(np.float32)
    b = np.clip(b, -1000.0, 1000.0).astype(np.float32)

    w["head.weight"] = W
    w["head.bias"] = b

    # Diagnostics
    logits = X @ W.T + b
    pred = np.argmax(logits, axis=1)
    acc = float(np.mean(pred == y))
    # Cross-entropy
    logits_max = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    ce = float(-np.mean(np.log(probs[np.arange(N), y] + 1e-12)))

    diagnostics = {
        "full_accuracy": acc,
        "cross_entropy": ce,
        "max_abs_weight": float(np.max(np.abs(W))),
        "max_abs_bias": float(np.max(np.abs(b))),
        "scale": float(scale) if max_abs > 0 else 1.0,
    }

    return {"weights": w, "diagnostics": diagnostics}
