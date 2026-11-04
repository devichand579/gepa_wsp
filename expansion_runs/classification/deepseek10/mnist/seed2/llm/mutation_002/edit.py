import numpy as np

def edit(w, f):
    # Extract features and labels from the full feedback pool
    X = f["features"].astype(np.float64)   # (2000, 128)
    y = f["y"].astype(np.int64)            # (2000,)
    n_samples, n_features = X.shape
    n_classes = 10

    # Compute class means
    mu = np.zeros((n_classes, n_features), dtype=np.float64)
    counts = np.zeros(n_classes, dtype=np.int64)
    for c in range(n_classes):
        mask = (y == c)
        counts[c] = np.sum(mask)
        if counts[c] > 0:
            mu[c] = X[mask].mean(axis=0)

    # Compute pooled within-class covariance
    S_w = np.zeros((n_features, n_features), dtype=np.float64)
    for c in range(n_classes):
        mask = (y == c)
        if counts[c] > 1:
            Xc = X[mask]
            Xc_centered = Xc - mu[c]
            S_w += Xc_centered.T @ Xc_centered
    S_w /= (n_samples - n_classes)

    # Regularise slightly for numerical stability
    ridge = 1e-3
    S_w += ridge * np.eye(n_features, dtype=np.float64)

    # Solve for LDA weights: W = mu @ inv(S_w)
    # Use solve instead of explicit inverse for better numerics
    W = np.linalg.solve(S_w, mu.T).T   # (10, 128)

    # Compute LDA biases: b_c = -0.5 * mu_c^T S_w^{-1} mu_c + log(prior_c)
    # Priors are roughly uniform, so log(prior) is constant and can be omitted.
    # We compute the quadratic term.
    b = np.zeros(n_classes, dtype=np.float64)
    for c in range(n_classes):
        b[c] = -0.5 * mu[c] @ np.linalg.solve(S_w, mu[c])

    # Write back to the weights dictionary
    w["fc2.weight"] = W.astype(np.float32)
    w["fc2.bias"] = b.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "lda_ridge": ridge,
        "class_counts": counts.tolist(),
        "fc2_weight_norm": float(np.linalg.norm(W)),
        "fc2_bias_norm": float(np.linalg.norm(b)),
    }

    return {"weights": w, "diagnostics": diagnostics}
