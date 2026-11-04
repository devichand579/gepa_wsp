import numpy as np

def edit(w, f):
    # Extract current weights
    fc1_w = w["fc1.weight"].copy()
    fc1_b = w["fc1.bias"].copy()
    fc2_w = w["fc2.weight"].copy()
    fc2_b = w["fc2.bias"].copy()

    # Get full feedback pool
    X = f["x"]                     # (2000,1,28,28)
    y = f["y"]                     # (2000,)
    H = f["features"]              # (2000,128) current hidden features after ReLU

    # Ensure H is float64 for stable linear algebra
    H = H.astype(np.float64)
    y = y.astype(np.int64)

    n_samples, n_features = H.shape
    n_classes = 10

    # Compute class means and pooled covariance
    means = np.zeros((n_classes, n_features), dtype=np.float64)
    counts = np.zeros(n_classes, dtype=np.float64)
    for c in range(n_classes):
        mask = (y == c)
        counts[c] = mask.sum()
        if counts[c] > 0:
            means[c] = H[mask].mean(axis=0)

    # Pooled within-class covariance
    S = np.zeros((n_features, n_features), dtype=np.float64)
    for c in range(n_classes):
        mask = (y == c)
        if counts[c] > 1:
            diff = H[mask] - means[c]
            S += diff.T @ diff
    S /= (n_samples - n_classes)

    # Regularize covariance slightly for numerical stability
    # Use a small ridge proportional to the average diagonal
    avg_var = np.trace(S) / n_features
    reg = 1e-3 * avg_var + 1e-8
    S_reg = S + reg * np.eye(n_features)

    # LDA weight matrix: W = S^{-1} * (means - global_mean).T
    # We can use the global mean as the reference class to reduce redundancy.
    global_mean = H.mean(axis=0)
    M = means - global_mean          # (10,128)
    # Solve S_reg * W.T = M.T  => W.T = S_reg^{-1} M.T
    W_T = np.linalg.solve(S_reg, M.T)   # (128,10)
    W_lda = W_T.T                       # (10,128)

    # Bias: for each class c, bias_c = - W_lda[c] @ global_mean + log(prior_c)
    # We can include prior from class frequencies. Use log(counts/n) for calibration.
    priors = counts / n_samples
    # Avoid log(0)
    priors = np.maximum(priors, 1e-12)
    log_priors = np.log(priors)
    b_lda = -W_lda @ global_mean + log_priors

    # Optionally scale the LDA weights to match the scale of current logits?
    # The current logits have std ~15. LDA weights might be smaller.
    # We can scale W_lda and b_lda by a factor to keep logits in a reasonable range.
    # But argmax is invariant to positive scaling, so we can leave as is.
    # However, for numerical stability and to avoid huge values, we clip.
    # The problem requires absolute values <= 1000.
    W_lda = np.clip(W_lda, -1000, 1000)
    b_lda = np.clip(b_lda, -1000, 1000)

    # Update weights
    w["fc2.weight"] = W_lda.astype(np.float32)
    w["fc2.bias"] = b_lda.astype(np.float32)

    # Keep fc1 unchanged
    w["fc1.weight"] = fc1_w
    w["fc1.bias"] = fc1_b

    # Diagnostics
    diag = {
        "method": "Fisher_LDA_on_current_features",
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "class_counts": counts.tolist(),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
        "fc2_weight_mean": float(w["fc2.weight"].mean()),
        "fc2_weight_std": float(w["fc2.weight"].std()),
        "fc2_bias_mean": float(w["fc2.bias"].mean()),
        "fc2_bias_std": float(w["fc2.bias"].std()),
    }

    return {"weights": w, "diagnostics": diag}
