import numpy as np

def edit(w, f):
    # Extract full feedback pool
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float32)  # [2000, 784]
    y = f["y"].astype(np.int64)                                 # [2000]
    H = f["features"].astype(np.float32)                        # [2000, 128]
    n, d = H.shape
    n_classes = 10

    # Compute class means and global mean
    mu_global = H.mean(axis=0)                                  # [128]
    mu_c = np.zeros((n_classes, d), dtype=np.float32)
    counts = np.zeros(n_classes, dtype=np.float32)
    for c in range(n_classes):
        mask = (y == c)
        counts[c] = mask.sum()
        if counts[c] > 0:
            mu_c[c] = H[mask].mean(axis=0)

    # Compute pooled within-class scatter matrix S_w
    S_w = np.zeros((d, d), dtype=np.float64)
    for c in range(n_classes):
        mask = (y == c)
        if counts[c] > 1:
            Hc = H[mask]
            Hc_centered = Hc - mu_c[c]
            S_w += Hc_centered.T @ Hc_centered

    # Regularize S_w slightly to improve numerical stability (still closed-form)
    # Use a small ridge proportional to the average diagonal
    avg_diag = np.trace(S_w) / d
    reg = 1e-4 * avg_diag
    S_w_reg = S_w + reg * np.eye(d, dtype=np.float64)

    # Solve for LDA weights: W = S_w^{-1} @ mu_c^T
    # We want fc2.weight shape [10, 128] and fc2.bias shape [10]
    # LDA discriminant: delta_c(h) = h^T S_w^{-1} mu_c - 0.5 mu_c^T S_w^{-1} mu_c + log(pi_c)
    S_w_inv = np.linalg.inv(S_w_reg)
    W_lda = (S_w_inv @ mu_c.T).T                                # [10, 128]
    # Bias: -0.5 * mu_c^T S_w^{-1} mu_c + log(pi_c)
    # Compute quadratic terms
    quad = np.einsum('ij,jk,ik->i', mu_c, S_w_inv, mu_c)        # [10]
    log_prior = np.log(counts / n)
    b_lda = -0.5 * quad + log_prior

    # Convert to float32 and assign to fc2
    w["fc2.weight"] = W_lda.astype(np.float32)
    w["fc2.bias"] = b_lda.astype(np.float32)

    # Optionally, we can also slightly adjust fc1? Not needed for this proposal.
    # Ensure all weights are finite and within bounds
    for k in w:
        w[k] = np.nan_to_num(w[k], nan=0.0, posinf=1000.0, neginf=-1000.0)
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)

    # Diagnostics
    diagnostics = {
        "method": "LDA on current features",
        "reg": float(reg),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
        "fc2_weight_mean": float(w["fc2.weight"].mean()),
        "fc2_weight_std": float(w["fc2.weight"].std()),
        "fc2_bias_mean": float(w["fc2.bias"].mean()),
        "fc2_bias_std": float(w["fc2.bias"].std()),
    }
    return {"weights": w, "diagnostics": diagnostics}
