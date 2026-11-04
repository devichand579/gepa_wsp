def edit(w, f):
    import numpy as np

    # Extract full feedback features and labels
    H = f["features"].astype(np.float64)   # (2000, 128)
    y = f["y"].astype(np.int64)            # (2000,)

    n_classes = 10
    n_features = H.shape[1]

    # Class means and counts
    mu = np.zeros((n_classes, n_features), dtype=np.float64)
    counts = np.zeros(n_classes, dtype=np.float64)
    for c in range(n_classes):
        mask = (y == c)
        counts[c] = mask.sum()
        if counts[c] > 0:
            mu[c] = H[mask].mean(axis=0)

    # Within-class scatter
    Sw = np.zeros((n_features, n_features), dtype=np.float64)
    for c in range(n_classes):
        mask = (y == c)
        if counts[c] > 0:
            diff = H[mask] - mu[c]
            Sw += diff.T @ diff

    # Regularise (small ridge)
    reg = 1e-3 * np.trace(Sw) / n_features
    Sw_reg = Sw + reg * np.eye(n_features)

    # LDA classifier: W_c = Sw^{-1} mu_c, b_c = -0.5 mu_c^T Sw^{-1} mu_c + log prior_c
    W = np.linalg.solve(Sw_reg, mu.T).T          # (10, 128)
    b = -0.5 * np.sum(mu * W, axis=1) + np.log(counts / counts.sum())

    # Scale to match current logits' standard deviation
    logits_old = f["logits"].astype(np.float64)  # (2000, 10)
    std_old = logits_old.std()
    Z = H @ W.T + b
    std_new = Z.std()
    scale = std_old / std_new if std_new > 1e-8 else 1.0
    W *= scale
    b *= scale

    # Write back to fc2 only
    w["fc2.weight"] = np.clip(W, -1000, 1000).astype(np.float32)
    w["fc2.bias"]   = np.clip(b, -1000, 1000).astype(np.float32)

    # Diagnostics
    train_acc = float((np.argmax(Z, axis=1) == y).mean())
    diagnostics = {
        "scale": float(scale),
        "std_old": float(std_old),
        "std_new": float(std_new),
        "train_accuracy": train_acc,
        "fc2_weight_mean": float(w["fc2.weight"].mean()),
        "fc2_weight_std": float(w["fc2.weight"].std()),
        "fc2_bias_mean": float(w["fc2.bias"].mean()),
        "fc2_bias_std": float(w["fc2.bias"].std()),
    }
    return {"weights": w, "diagnostics": diagnostics}
