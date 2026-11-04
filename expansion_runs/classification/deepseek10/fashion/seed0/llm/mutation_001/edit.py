import numpy as np

def edit(w, f):
    # Extract full feedback features and labels
    X = f["features"].astype(np.float64)   # [2000, 1568]
    y = f["y"]                             # [2000]
    N, d = X.shape
    classes = np.unique(y)
    n_classes = len(classes)

    # Class means and counts
    mu = np.zeros((n_classes, d))
    n_c = np.zeros(n_classes)
    for i, c in enumerate(classes):
        mask = (y == c)
        n_c[i] = np.sum(mask)
        mu[i] = X[mask].mean(axis=0)

    # Total scatter X^T X
    XtX = X.T @ X

    # Within-class scatter S_w = X^T X - sum_c n_c * mu_c mu_c^T
    S_w = XtX.copy()
    for i in range(n_classes):
        S_w -= n_c[i] * np.outer(mu[i], mu[i])

    # Ridge regularization for numerical stability and shrinkage
    trace = np.trace(S_w)
    reg = 1e-3 * trace / d
    S = S_w + reg * np.eye(d)

    # Solve S * W = mu^T  (LDA weights)
    # Use least squares for robustness against near-singular S
    W = np.linalg.lstsq(S, mu.T, rcond=None)[0].T   # [n_classes, d]

    # LDA bias: -0.5 * mu_c^T W_c + log prior
    b = -0.5 * np.sum(mu * W, axis=1) + np.log(n_c / N)

    # Scale down if any absolute value exceeds 1000 (preserves argmax)
    max_val = max(np.max(np.abs(W)), np.max(np.abs(b)))
    if max_val > 1000.0:
        scale = 1000.0 / max_val
        W *= scale
        b *= scale

    # Evaluate training accuracy with the new head
    logits = X @ W.T + b
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == y)

    # Update weights in place
    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

    diagnostics = {
        "train_accuracy": float(acc),
        "max_weight": float(np.max(np.abs(w["head.weight"]))),
        "max_bias": float(np.max(np.abs(w["head.bias"]))),
        "trace_S_w": float(trace),
        "reg": float(reg),
        "n_classes": int(n_classes),
        "n_samples": int(N)
    }
    return {"weights": w, "diagnostics": diagnostics}
