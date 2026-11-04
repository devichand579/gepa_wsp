import numpy as np

def edit(w, f):
    # Extract features and labels from the full feedback pool
    X = f["features"]          # (2000, 1568)
    y = f["y"]                 # (2000,)
    N, D = X.shape
    C = 10

    # Compute class means
    mu = np.zeros((C, D), dtype=np.float32)
    for c in range(C):
        mu[c] = X[y == c].mean(axis=0)

    # Compute pooled within-class variance per feature
    # (X - mu[y]) has shape (N, D)
    diff = X - mu[y]
    var = np.mean(diff ** 2, axis=0)   # (D,)
    # Add small epsilon to avoid division by zero
    eps = 1e-8
    var = np.maximum(var, eps)

    # Diagonal LDA weights and biases
    W = mu / var                       # (C, D)
    b = -0.5 * np.sum(mu ** 2 / var, axis=1)  # (C,)

    # Ensure float32 and clip to allowed range
    W = np.clip(W, -1000.0, 1000.0).astype(np.float32)
    b = np.clip(b, -1000.0, 1000.0).astype(np.float32)

    # Update head weights and biases
    w["head.weight"] = W
    w["head.bias"] = b

    # Diagnostics
    diagnostics = {
        "mean_var": float(np.mean(var)),
        "max_abs_W": float(np.max(np.abs(W))),
        "max_abs_b": float(np.max(np.abs(b))),
        "class_means_norm": [float(np.linalg.norm(mu[c])) for c in range(C)],
    }

    return {"weights": w, "diagnostics": diagnostics}
