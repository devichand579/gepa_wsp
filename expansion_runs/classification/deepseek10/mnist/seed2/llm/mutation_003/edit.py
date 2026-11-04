import numpy as np

def edit(w, f):
    # Extract data
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)  # [2000, 784]
    y = f["y"].astype(np.int64)                                 # [2000]
    n = X.shape[0]
    num_classes = 10

    # Current fc1 parameters
    W1 = w["fc1.weight"].astype(np.float64)  # [128, 784]
    b1 = w["fc1.bias"].astype(np.float64)    # [128]

    # Compute pre-activations for all training data
    Z = X @ W1.T + b1  # [2000, 128]

    # --- LDA on Z to find new linear combination A (128x128) and bias c ---
    # Class means of Z
    means = np.zeros((num_classes, Z.shape[1]), dtype=np.float64)
    for c in range(num_classes):
        idx = (y == c)
        if np.any(idx):
            means[c] = Z[idx].mean(axis=0)
    global_mean = Z.mean(axis=0)

    # Within-class scatter
    Sw = np.zeros((Z.shape[1], Z.shape[1]), dtype=np.float64)
    for c in range(num_classes):
        idx = (y == c)
        if np.any(idx):
            Zc = Z[idx] - means[c]
            Sw += Zc.T @ Zc

    # Between-class scatter
    Sb = np.zeros((Z.shape[1], Z.shape[1]), dtype=np.float64)
    for c in range(num_classes):
        idx = (y == c)
        nc = np.sum(idx)
        if nc > 0:
            diff = (means[c] - global_mean).reshape(-1, 1)
            Sb += nc * (diff @ diff.T)

    # Regularize Sw to avoid singularity
    ridge = 1e-3 * np.trace(Sw) / Sw.shape[0] + 1e-8
    Sw_reg = Sw + ridge * np.eye(Sw.shape[0])

    # Solve generalized eigenvalue problem: Sb v = lambda Sw v
    # Equivalent to eig(Sw^{-1} Sb)
    Sw_inv = np.linalg.inv(Sw_reg)
    M = Sw_inv @ Sb
    eigvals, eigvecs = np.linalg.eig(M)
    # Sort by eigenvalue descending
    order = np.argsort(eigvals.real)[::-1]
    V = eigvecs[:, order].real  # [128, 128]
    # Normalize eigenvectors (optional)
    V = V / (np.linalg.norm(V, axis=0, keepdims=True) + 1e-12)

    # A is the transpose of V, so that new Z' = A @ Z + c
    A = V.T  # [128, 128]
    c = -A @ global_mean  # [128]

    # New fc1 parameters
    new_W1 = A @ W1  # [128, 784]
    new_b1 = A @ b1 + c  # [128]

    # Compute new hidden features (post-ReLU)
    Z_new = Z @ A.T + c  # [2000, 128]
    H = np.maximum(Z_new, 0.0)  # [2000, 128]

    # --- LDA on H to fit new fc2 ---
    # Class means of H
    means_h = np.zeros((num_classes, H.shape[1]), dtype=np.float64)
    for c in range(num_classes):
        idx = (y == c)
        if np.any(idx):
            means_h[c] = H[idx].mean(axis=0)
    global_mean_h = H.mean(axis=0)

    # Within-class scatter
    Sw_h = np.zeros((H.shape[1], H.shape[1]), dtype=np.float64)
    for c in range(num_classes):
        idx = (y == c)
        if np.any(idx):
            Hc = H[idx] - means_h[c]
            Sw_h += Hc.T @ Hc

    # Between-class scatter
    Sb_h = np.zeros((H.shape[1], H.shape[1]), dtype=np.float64)
    for c in range(num_classes):
        idx = (y == c)
        nc = np.sum(idx)
        if nc > 0:
            diff = (means_h[c] - global_mean_h).reshape(-1, 1)
            Sb_h += nc * (diff @ diff.T)

    # Regularize
    ridge_h = 1e-3 * np.trace(Sw_h) / Sw_h.shape[0] + 1e-8
    Sw_h_reg = Sw_h + ridge_h * np.eye(Sw_h.shape[0])

    # LDA solution for fc2: W2 = (Sw_h^{-1} (means_h - global_mean_h))^T
    # Actually, for multi-class LDA, the optimal weights are given by
    # W2 = (Sw_h^{-1} @ (means_h - global_mean_h).T).T  # [10, 128]
    # Bias b2 = - W2 @ global_mean_h + log(prior) (priors equal here)
    Sw_h_inv = np.linalg.inv(Sw_h_reg)
    diff_means = means_h - global_mean_h  # [10, 128]
    W2 = (Sw_h_inv @ diff_means.T).T  # [10, 128]
    b2 = -W2 @ global_mean_h  # [10]

    # Scale W2 and b2 to avoid extreme logits (optional, but helps numerical stability)
    # We can scale so that the maximum absolute value is reasonable, e.g., 10.
    scale = 10.0 / (np.max(np.abs(W2)) + 1e-12)
    W2 *= scale
    b2 *= scale

    # Build new weights dictionary
    new_w = {
        "fc1.weight": new_W1.astype(np.float32),
        "fc1.bias": new_b1.astype(np.float32),
        "fc2.weight": W2.astype(np.float32),
        "fc2.bias": b2.astype(np.float32),
    }

    # Diagnostics
    diagnostics = {
        "ridge_fc1": float(ridge),
        "ridge_fc2": float(ridge_h),
        "new_W1_norm": float(np.linalg.norm(new_W1)),
        "new_b1_norm": float(np.linalg.norm(new_b1)),
        "new_W2_norm": float(np.linalg.norm(W2)),
        "new_b2_norm": float(np.linalg.norm(b2)),
        "scale_fc2": float(scale),
    }

    return {"weights": new_w, "diagnostics": diagnostics}
