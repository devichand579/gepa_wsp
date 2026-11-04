import numpy as np

def edit(w, f):
    # Extract full feedback pool
    F = f["features"].astype(np.float64)          # [2000, 128]
    y = f["y"].astype(np.int64)                   # [2000]
    N, D = F.shape
    C = 10

    # Standardise features (zero mean, unit variance per dimension)
    mu = F.mean(axis=0, keepdims=True)
    sigma = F.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    Fs = (F - mu) / sigma

    # Compute class means and priors
    means = np.zeros((C, D), dtype=np.float64)
    priors = np.zeros(C, dtype=np.float64)
    for c in range(C):
        mask = (y == c)
        if np.any(mask):
            means[c] = Fs[mask].mean(axis=0)
            priors[c] = mask.sum() / N
        else:
            priors[c] = 1.0 / C

    # Within-class scatter matrix
    Sw = np.zeros((D, D), dtype=np.float64)
    for c in range(C):
        mask = (y == c)
        if np.any(mask):
            Xc = Fs[mask] - means[c]
            Sw += Xc.T @ Xc

    # Ridge regularisation for numerical stability
    ridge = 1e-3
    Sw += ridge * np.eye(D)

    # Invert Sw (use solve for stability)
    try:
        Sw_inv = np.linalg.inv(Sw)
    except np.linalg.LinAlgError:
        Sw_inv = np.linalg.pinv(Sw)

    # LDA weights and biases
    W = np.zeros((C, D), dtype=np.float64)
    b = np.zeros(C, dtype=np.float64)
    for c in range(C):
        W[c] = Sw_inv @ means[c]
        b[c] = -0.5 * means[c] @ Sw_inv @ means[c] + np.log(priors[c] + 1e-12)

    # Optional: scale weights to a reasonable magnitude (softmax is scale-sensitive)
    # Keep as is; the scale is determined by the data.

    # Write back to fc2
    w["fc2.weight"] = W.astype(np.float32)
    w["fc2.bias"] = b.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "lda_ridge": ridge,
        "fc2_weight_mean": float(w["fc2.weight"].mean()),
        "fc2_weight_std": float(w["fc2.weight"].std()),
        "fc2_bias_mean": float(w["fc2.bias"].mean()),
        "fc2_bias_std": float(w["fc2.bias"].std()),
        "feature_mean_norm": float(np.linalg.norm(mu)),
        "feature_std_mean": float(sigma.mean()),
    }

    return {"weights": w, "diagnostics": diagnostics}
