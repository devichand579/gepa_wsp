import numpy as np

def edit(w, f):
    # Extract full feedback features and labels
    X = f["features"].astype(np.float64)   # (2000, 1568)
    y = f["y"].astype(np.int64)            # (2000,)
    n, d = X.shape
    classes = np.unique(y)
    K = len(classes)

    # Class means and priors
    mu = np.zeros((K, d), dtype=np.float64)
    priors = np.zeros(K, dtype=np.float64)
    for i, k in enumerate(classes):
        mask = (y == k)
        mu[i] = X[mask].mean(axis=0)
        priors[i] = mask.mean()

    # Within-class scatter matrix
    X_centered = X - mu[y]                 # broadcast: (2000, 1568)
    S_w = X_centered.T @ X_centered        # (1568, 1568)

    # Regularized LDA (shrinkage)
    alpha = 1e-3
    S_w_reg = S_w + alpha * np.eye(d, dtype=np.float64)

    # Solve S_w_reg @ W^T = mu^T  =>  W = (S_w_reg^{-1} mu^T)^T
    # Using np.linalg.solve is more stable than explicit inverse.
    W = np.linalg.solve(S_w_reg, mu.T).T   # (K, d)
    b = -0.5 * np.sum(mu * W, axis=1) + np.log(priors)  # (K,)

    # Compute logits on the full pool to determine scaling
    logits = X @ W.T + b                   # (2000, 10)
    logit_std = np.std(logits)

    # Scale logits to a target standard deviation (heuristic to reduce CE)
    target_std = 10.0
    scale = target_std / (logit_std + 1e-12)
    W *= scale
    b *= scale

    # Write back to the head only
    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

    # Diagnostics
    diag = {
        "lda_alpha": alpha,
        "logit_std_before_scale": float(logit_std),
        "scale_factor": float(scale),
        "head_weight_std": float(np.std(W)),
        "head_bias_mean": float(np.mean(b)),
        "head_weight_max_abs": float(np.max(np.abs(W))),
        "head_bias_max_abs": float(np.max(np.abs(b))),
    }
    return {"weights": w, "diagnostics": diag}
