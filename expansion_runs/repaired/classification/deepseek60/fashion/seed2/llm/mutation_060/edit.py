import numpy as np

def edit(w, f):
    # Extract features and labels from the feedback pool
    X = f["features"].astype(np.float64)   # (2000, 1568)
    y = f["y"].astype(np.int64)            # (2000,)
    n, p = X.shape
    C = 10

    # Standardize features (zero mean, unit variance)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X_std = (X - mean) / std

    # Class means and counts
    mu = np.zeros((C, p))
    counts = np.zeros(C)
    for c in range(C):
        mask = (y == c)
        counts[c] = mask.sum()
        if counts[c] > 0:
            mu[c] = X_std[mask].mean(axis=0)

    # Within-class centered data
    Xc = X_std - mu[y]

    # SVD of within-class centered data
    # Xc is (n, p); S has length min(n, p) = 1568
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    S2 = S ** 2
    denom = n - C
    eigenvalues = S2 / denom

    # Shrinkage: add a small ridge to stabilize the covariance
    lambda_ = 1e-3 * np.mean(eigenvalues)

    # Compute Sigma_inv @ mu_c for each class
    # Sigma_inv = V diag(1/(eigenvalues + lambda)) V^T
    Vt_mu = Vt @ mu.T                     # (p, C)
    inv_eig = 1.0 / (eigenvalues + lambda_)
    w_std = Vt.T @ (inv_eig[:, None] * Vt_mu)  # (p, C)
    w_std = w_std.T                       # (C, p)

    # Bias: b_c = -0.5 * mu_c^T Sigma_inv mu_c + log(prior_c)
    b_std = -0.5 * np.sum(mu * w_std, axis=1) + np.log(counts / n)

    # Convert back to original feature space
    # X_std = (X - mean) / std
    # w_std @ X_std + b_std = (w_std / std) @ X - (w_std / std) @ mean + b_std
    w_orig = w_std / std
    b_orig = b_std - w_orig @ mean

    # Scale uniformly if any parameter exceeds 1000
    max_abs = max(np.abs(w_orig).max(), np.abs(b_orig).max())
    if max_abs > 1000.0:
        scale = 1000.0 / max_abs
        w_orig *= scale
        b_orig *= scale

    # Write new head weights and biases
    w["head.weight"] = w_orig.astype(np.float32)
    w["head.bias"] = b_orig.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "max_abs_head_weight": float(np.abs(w_orig).max()),
        "max_abs_head_bias": float(np.abs(b_orig).max()),
        "lambda": float(lambda_),
        "mean_eigenvalue": float(np.mean(eigenvalues)),
        "min_eigenvalue": float(np.min(eigenvalues)),
        "max_eigenvalue": float(np.max(eigenvalues)),
    }

    return {"weights": w, "diagnostics": diagnostics}
