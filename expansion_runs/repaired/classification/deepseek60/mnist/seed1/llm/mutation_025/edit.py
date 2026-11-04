import numpy as np

def edit(w, f):
    # Extract full feedback pool
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)  # [2000, 784]
    y = f["y"].astype(np.int64)                                 # [2000]
    H = f["features"].astype(np.float64)                        # [2000, 128]
    n, d = H.shape
    classes = np.unique(y)
    C = len(classes)

    # Class means and priors
    means = np.zeros((C, d), dtype=np.float64)
    priors = np.zeros(C, dtype=np.float64)
    for i, c in enumerate(classes):
        mask = (y == c)
        means[i] = H[mask].mean(axis=0)
        priors[i] = mask.mean()

    # Pooled within-class covariance
    Sigma = np.zeros((d, d), dtype=np.float64)
    for i, c in enumerate(classes):
        mask = (y == c)
        Hc = H[mask] - means[i]
        Sigma += Hc.T @ Hc
    Sigma /= (n - C)

    # Shrinkage regularization (fixed, robust)
    alpha = 0.1
    Sigma_shrunk = (1.0 - alpha) * Sigma + alpha * np.trace(Sigma) / d * np.eye(d)

    # Solve LDA weights
    # W: [C, d], b: [C]
    W = np.linalg.solve(Sigma_shrunk, means.T).T          # [C, d]
    b = -0.5 * np.sum(means * W, axis=1) + np.log(priors + 1e-12)

    # Convert to float32 and assign to fc2
    w["fc2.weight"] = W.astype(np.float32)
    w["fc2.bias"]   = b.astype(np.float32)

    # Clip to safe range
    w["fc2.weight"] = np.clip(w["fc2.weight"], -1000.0, 1000.0)
    w["fc2.bias"]   = np.clip(w["fc2.bias"], -1000.0, 1000.0)

    # Diagnostics
    logits_new = H @ W.T + b
    pred = np.argmax(logits_new, axis=1)
    acc = (pred == y).mean()
    # Cross-entropy
    logits_new -= logits_new.max(axis=1, keepdims=True)
    probs = np.exp(logits_new)
    probs /= probs.sum(axis=1, keepdims=True)
    ce = -np.log(probs[np.arange(n), y] + 1e-12).mean()

    diagnostics = {
        "method": "lda_shrinkage",
        "alpha": alpha,
        "train_accuracy": float(acc),
        "train_cross_entropy": float(ce),
        "fc2_weight_mean": float(w["fc2.weight"].mean()),
        "fc2_weight_std": float(w["fc2.weight"].std()),
        "fc2_bias_mean": float(w["fc2.bias"].mean()),
        "fc2_bias_std": float(w["fc2.bias"].std()),
    }
    return {"weights": w, "diagnostics": diagnostics}
