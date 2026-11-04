import numpy as np

def edit(w, f):
    # Extract features and labels for the entire feedback pool
    X = f["features"].astype(np.float64)          # (2000, 1568)
    y = f["y"].astype(np.int64)                   # (2000,)
    n, d = X.shape
    classes = np.unique(y)
    C = len(classes)

    # Compute class means and global mean
    mu = X.mean(axis=0)
    mu_c = np.zeros((C, d), dtype=np.float64)
    for i, c in enumerate(classes):
        mu_c[i] = X[y == c].mean(axis=0)

    # Within-class scatter matrix Sw = sum_i (x_i - mu_{y_i})(x_i - mu_{y_i})^T
    # Compute efficiently by centering each sample by its class mean
    Xc = X - mu_c[y]                              # (2000, 1568)
    Sw = Xc.T @ Xc                                # (1568, 1568)

    # Add a small ridge for numerical stability (equivalent to regularised LDA)
    # The original head was extremely unregularised; a tiny ridge keeps it stable.
    lam = 1e-3
    Sw += lam * np.eye(d, dtype=np.float64)

    # LDA directions: W = Sw^{-1} (mu_c - mu)
    # Solve Sw * W^T = (mu_c - mu)^T for numerical stability
    B = (mu_c - mu).T                             # (1568, 10)
    W = np.linalg.solve(Sw, B).T                  # (10, 1568)

    # Bias: choose to make class logits comparable; use log priors (uniform here)
    # For balanced classes, zero bias is fine, but we can also centre the logits.
    bias = np.zeros(C, dtype=np.float64)

    # Scale W and bias so that max |weight| <= 1000
    max_abs = np.max(np.abs(W))
    if max_abs > 0:
        scale = 1000.0 / max_abs
        W *= scale
        bias *= scale

    # Ensure float32 and within bounds
    W = np.clip(W, -1000.0, 1000.0).astype(np.float32)
    bias = np.clip(bias, -1000.0, 1000.0).astype(np.float32)

    # Write back into the head
    w["head.weight"] = W
    w["head.bias"] = bias

    # Diagnostics
    logits = X @ W.T + bias
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    pred = probs.argmax(axis=1)
    acc = (pred == y).mean()
    ce = -np.log(np.clip(probs[np.arange(n), y], 1e-12, 1.0)).mean()

    diagnostics = {
        "accuracy": float(acc),
        "cross_entropy": float(ce),
        "head_weight_max_abs": float(np.max(np.abs(W))),
        "head_bias_max_abs": float(np.max(np.abs(bias))),
        "ridge_lambda": lam,
    }
    return {"weights": w, "diagnostics": diagnostics}
