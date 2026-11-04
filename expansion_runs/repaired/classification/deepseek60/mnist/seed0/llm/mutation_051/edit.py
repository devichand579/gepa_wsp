import numpy as np

def edit(w, f):
    # Extract feedback data
    x = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)  # (2000, 784)
    y = f["y"].astype(np.int64)
    N = x.shape[0]
    C = 10

    # Compute class means and overall mean of raw images
    mu_c = np.zeros((C, x.shape[1]), dtype=np.float64)
    for c in range(C):
        mu_c[c] = x[y == c].mean(axis=0)
    mu = x.mean(axis=0)

    # Copy current fc1 weights and biases
    W1 = w["fc1.weight"].astype(np.float64).copy()  # (128, 784)
    b1 = w["fc1.bias"].astype(np.float64).copy()    # (128,)

    # Replace the first 10 dead units with class-mean difference templates
    # and shift bias so the ReLU is mostly active (offset = 1.0).
    for c in range(C):
        W1[c] = mu_c[c] - mu
        b1[c] = -W1[c] @ mu + 1.0

    # Compute new hidden features for all feedback examples
    H = x @ W1.T + b1          # (2000, 128)
    H = np.maximum(H, 0.0)     # ReLU

    # Standardize features (z-score)
    mean_h = H.mean(axis=0)
    std_h = H.std(axis=0) + 1e-6
    H_std = (H - mean_h) / std_h

    # Compute class means in standardized feature space
    mu_c_h = np.zeros((C, H.shape[1]), dtype=np.float64)
    for c in range(C):
        mu_c_h[c] = H_std[y == c].mean(axis=0)
    mu_h = H_std.mean(axis=0)

    # Within-class scatter matrix
    Sw = np.zeros((H.shape[1], H.shape[1]), dtype=np.float64)
    for c in range(C):
        diff = H_std[y == c] - mu_c_h[c]
        Sw += diff.T @ diff
    cov = Sw / (N - C)

    # Regularize covariance (ridge-like) to improve conditioning
    alpha = 1e-3 * np.trace(cov) / cov.shape[0]
    cov_reg = cov + alpha * np.eye(cov.shape[0])

    # LDA weights and biases in standardized space
    inv_cov = np.linalg.inv(cov_reg)
    W_lda = mu_c_h @ inv_cov                     # (10, 128)
    priors = np.array([np.mean(y == c) for c in range(C)])
    b_lda = -0.5 * np.sum(mu_c_h * W_lda, axis=1) + np.log(priors)

    # Convert back to original feature space
    # H_std = (H - mean_h) / std_h
    # decision = W_lda @ H_std + b_lda = (W_lda / std_h) @ H + (b_lda - W_lda @ (mean_h / std_h))
    W2 = W_lda / std_h                           # (10, 128)
    b2 = b_lda - W2 @ mean_h

    # Update weights (float32, finite, within bounds)
    w["fc1.weight"] = W1.astype(np.float32)
    w["fc1.bias"]   = b1.astype(np.float32)
    w["fc2.weight"] = W2.astype(np.float32)
    w["fc2.bias"]   = b2.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "method": "revive_first10_templates_LDA",
        "alpha": float(alpha),
        "fc1_weight_norm": float(np.linalg.norm(w["fc1.weight"])),
        "fc1_bias_norm": float(np.linalg.norm(w["fc1.bias"])),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
        "fc1_first10_bias": [float(v) for v in w["fc1.bias"][:10]],
    }
    return {"weights": w, "diagnostics": diagnostics}
