import numpy as np

def edit(w, f):
    # Read full feedback pool
    x = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)  # [2000, 784]
    y = f["y"].astype(np.int64)
    N, D = x.shape
    C = 10

    # Class means and global mean
    mu = np.zeros((C, D), dtype=np.float64)
    counts = np.zeros(C, dtype=np.float64)
    for c in range(C):
        idx = (y == c)
        counts[c] = idx.sum()
        if counts[c] > 0:
            mu[c] = x[idx].mean(axis=0)
    mu_all = x.mean(axis=0)

    # Within-class scatter Sw
    Sw = np.zeros((D, D), dtype=np.float64)
    for c in range(C):
        idx = (y == c)
        if counts[c] > 0:
            Xc = x[idx] - mu[c]
            Sw += Xc.T @ Xc
    Sw /= N

    # Between-class scatter Sb
    Sb = np.zeros((D, D), dtype=np.float64)
    for c in range(C):
        diff = (mu[c] - mu_all).reshape(-1, 1)
        Sb += counts[c] * (diff @ diff.T)
    Sb /= N

    # Regularize Sw for numerical stability
    ridge = 1e-3 * np.trace(Sw) / D
    if ridge <= 0:
        ridge = 1e-6
    Sw_reg = Sw + ridge * np.eye(D)

    # Solve generalized eigenproblem: top 128 eigenvectors of Sw^{-1} Sb
    L = np.linalg.cholesky(Sw_reg)
    # A = L^{-1} Sb L^{-T}
    tmp = np.linalg.solve(L, Sb)
    A = np.linalg.solve(L, tmp.T)
    A = (A + A.T) / 2.0  # symmetrize for eigh
    eigvals, eigvecs = np.linalg.eigh(A)
    # Sort descending
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    # Take top 128
    U = eigvecs[:, :128]  # [D, 128]
    # Convert back to original space: v = L^{-T} u
    V = np.linalg.solve(L.T, U)  # [D, 128]
    # W1 should be [128, D]
    W1 = V.T  # [128, 784]

    # Normalize each row to unit norm, then scale to match current fc1 norm
    norms = np.linalg.norm(W1, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    W1 = W1 / norms
    # Scale to have similar overall norm as current fc1.weight
    current_norm = np.linalg.norm(w["fc1.weight"])
    target_norm = current_norm if current_norm > 0 else 11.3
    W1 = W1 * (target_norm / np.sqrt(128))

    # Bias: center the projections at the global mean
    b1 = -W1 @ mu_all
    # Optionally scale bias to have similar norm as current
    # Keep as is, but ensure finite
    b1 = b1.astype(np.float32)
    W1 = W1.astype(np.float32)

    # Compute new hidden features for the full pool
    Z = x @ W1.T + b1  # [N, 128]
    H = np.maximum(Z, 0.0)  # ReLU

    # Fit fc2 by regularized LDA on H
    # Compute class means of H
    mu_h = np.zeros((C, 128), dtype=np.float64)
    for c in range(C):
        idx = (y == c)
        if counts[c] > 0:
            mu_h[c] = H[idx].mean(axis=0)
    mu_h_all = H.mean(axis=0)

    # Within-class scatter for H
    Sw_h = np.zeros((128, 128), dtype=np.float64)
    for c in range(C):
        idx = (y == c)
        if counts[c] > 0:
            Hc = H[idx] - mu_h[c]
            Sw_h += Hc.T @ Hc
    Sw_h /= N

    # Between-class scatter for H
    Sb_h = np.zeros((128, 128), dtype=np.float64)
    for c in range(C):
        diff = (mu_h[c] - mu_h_all).reshape(-1, 1)
        Sb_h += counts[c] * (diff @ diff.T)
    Sb_h /= N

    # Regularized LDA for fc2
    # We want W2 such that logits = W2 @ h + b2
    # Standard LDA: W2 = Sw^{-1} (mu_c - mu_all) for each class
    # Bias: b2_c = -0.5 * mu_c^T Sw^{-1} mu_c + log(prior_c)
    # But we can also just use the normal equations for linear discriminant.
    # Let's use a small ridge.
    ridge_h = 1e-3 * np.trace(Sw_h) / 128
    if ridge_h <= 0:
        ridge_h = 1e-6
    Sw_h_reg = Sw_h + ridge_h * np.eye(128)
    # Solve for W2: Sw_h_reg @ W2.T = (mu_h - mu_h_all).T
    # Actually W2 should be [10, 128]. We want W2 @ h + b2.
    # The LDA projection is W2 = (mu_h - mu_h_all) @ Sw_h_reg^{-1}
    # So W2 = np.linalg.solve(Sw_h_reg, (mu_h - mu_h_all).T).T
    diff_means = (mu_h - mu_h_all).T  # [128, 10]
    W2 = np.linalg.solve(Sw_h_reg, diff_means).T  # [10, 128]
    # Bias: b2_c = -0.5 * mu_h[c] @ W2[c] + log(counts[c]/N)
    b2 = np.zeros(C, dtype=np.float64)
    for c in range(C):
        b2[c] = -0.5 * mu_h[c] @ W2[c] + np.log(counts[c] / N)

    # Scale W2 and b2 to reasonable magnitudes (similar to current)
    # Current fc2 weight norm ~120, bias norm ~100.
    # Our LDA might have different scale. We can rescale to match current norms.
    current_w2_norm = np.linalg.norm(w["fc2.weight"])
    current_b2_norm = np.linalg.norm(w["fc2.bias"])
    new_w2_norm = np.linalg.norm(W2)
    new_b2_norm = np.linalg.norm(b2)
    if new_w2_norm > 0:
        W2 = W2 * (current_w2_norm / new_w2_norm)
    if new_b2_norm > 0:
        b2 = b2 * (current_b2_norm / new_b2_norm)

    # Convert to float32 and clip to safe range
    W1 = np.clip(W1, -1000, 1000).astype(np.float32)
    b1 = np.clip(b1, -1000, 1000).astype(np.float32)
    W2 = np.clip(W2, -1000, 1000).astype(np.float32)
    b2 = np.clip(b2, -1000, 1000).astype(np.float32)

    # Update weights
    w["fc1.weight"] = W1
    w["fc1.bias"] = b1
    w["fc2.weight"] = W2
    w["fc2.bias"] = b2

    # Diagnostics
    # Compute training accuracy on full pool
    logits = H @ W2.T + b2
    preds = np.argmax(logits, axis=1)
    acc = (preds == y).mean()
    diagnostics = {
        "train_accuracy": float(acc),
        "fc1_weight_norm": float(np.linalg.norm(W1)),
        "fc1_bias_norm": float(np.linalg.norm(b1)),
        "fc2_weight_norm": float(np.linalg.norm(W2)),
        "fc2_bias_norm": float(np.linalg.norm(b2)),
        "hidden_mean": float(H.mean()),
        "hidden_std": float(H.std()),
        "hidden_zero_fraction": float((H == 0).mean()),
    }
    return {"weights": w, "diagnostics": diagnostics}
