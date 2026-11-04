import numpy as np

def edit(w, f):
    # Read current parameters
    fc1_w = w["fc1.weight"].astype(np.float64)
    fc1_b = w["fc1.bias"].astype(np.float64)
    fc2_w = w["fc2.weight"].astype(np.float64)
    fc2_b = w["fc2.bias"].astype(np.float64)

    # Full feedback features and labels
    H = f["features"].astype(np.float64)          # [2000, 128]
    y = f["y"].astype(np.int64)                   # [2000]
    n, d = H.shape
    C = 10

    # One-hot targets
    Y = np.zeros((n, C), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    # Class means and pooled covariance
    means = np.zeros((C, d), dtype=np.float64)
    counts = np.zeros(C, dtype=np.float64)
    for c in range(C):
        idx = (y == c)
        counts[c] = idx.sum()
        if counts[c] > 0:
            means[c] = H[idx].mean(axis=0)

    # Pooled within-class covariance
    Sigma = np.zeros((d, d), dtype=np.float64)
    for c in range(C):
        idx = (y == c)
        if counts[c] > 1:
            Xc = H[idx] - means[c]
            Sigma += Xc.T @ Xc
    Sigma /= max(n - C, 1)

    # Strong regularization to avoid overfitting and overconfidence
    # Scale lambda relative to average feature variance
    avg_var = np.trace(Sigma) / d
    lam = 0.5 * avg_var + 1e-4
    Sigma_reg = Sigma + lam * np.eye(d)

    # LDA weights: W = Sigma^-1 * means^T
    # Solve Sigma_reg * W^T = means^T
    W_lda = np.linalg.solve(Sigma_reg, means.T).T   # [C, d]

    # Bias for LDA: -0.5 * diag(means @ Sigma^-1 @ means^T) + log(priors)
    # Compute quadratic terms
    quad = np.sum(means * W_lda, axis=1)            # [C]
    priors = counts / n
    b_lda = -0.5 * quad + np.log(priors + 1e-12)

    # Shrink toward current fc2 weights to preserve calibration/scale
    # Choose alpha small enough to correct errors but not destroy current scale
    alpha = 0.35
    W_new = (1.0 - alpha) * fc2_w + alpha * W_lda
    b_new = (1.0 - alpha) * fc2_b + alpha * b_lda

    # Optional: rescale to match current weight norm roughly
    cur_norm = np.linalg.norm(fc2_w)
    new_norm = np.linalg.norm(W_new)
    if new_norm > 1e-12:
        scale = cur_norm / new_norm
        # Limit scale change to avoid extreme values
        scale = np.clip(scale, 0.5, 2.0)
        W_new *= scale
        b_new *= scale

    # Write back as float32
    w["fc2.weight"] = W_new.astype(np.float32)
    w["fc2.bias"] = b_new.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "alpha": float(alpha),
        "lambda": float(lam),
        "fc2_weight_norm_before": float(np.linalg.norm(fc2_w)),
        "fc2_weight_norm_after": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm_after": float(np.linalg.norm(w["fc2.bias"])),
        "feature_mean_norm": float(np.linalg.norm(H.mean(axis=0))),
        "class_counts": counts.astype(int).tolist(),
    }
    return {"weights": w, "diagnostics": diagnostics}
