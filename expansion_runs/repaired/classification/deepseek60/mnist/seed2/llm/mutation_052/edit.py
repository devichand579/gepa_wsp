import numpy as np
import math

def edit(w, f):
    # Extract data
    x = f["x"]  # (2000, 1, 28, 28)
    y = f["y"]  # (2000,)
    N = x.shape[0]
    x_flat = x.reshape(N, -1).astype(np.float32)  # (2000, 784)

    # Current weights
    fc1_w = w["fc1.weight"].copy()  # (128, 784)
    fc1_b = w["fc1.bias"].copy()    # (128,)
    fc2_w = w["fc2.weight"].copy()  # (10, 128)
    fc2_b = w["fc2.bias"].copy()    # (10,)

    # Identify dead units: most negative biases
    dead_idx = np.argsort(fc1_b)[:20]  # replace 20 units

    # Compute LDA directions from raw images
    classes = np.unique(y)
    n_classes = len(classes)
    means = np.zeros((n_classes, x_flat.shape[1]), dtype=np.float32)
    for i, c in enumerate(classes):
        means[i] = x_flat[y == c].mean(axis=0)

    # Pooled within-class covariance
    cov = np.zeros((x_flat.shape[1], x_flat.shape[1]), dtype=np.float32)
    for i, c in enumerate(classes):
        Xc = x_flat[y == c] - means[i]
        cov += Xc.T @ Xc
    cov /= (N - n_classes)
    # Regularize covariance
    cov += np.eye(cov.shape[0], dtype=np.float32) * 1e-3

    # Between-class scatter
    overall_mean = x_flat.mean(axis=0)
    Sb = np.zeros_like(cov)
    for i, c in enumerate(classes):
        diff = means[i] - overall_mean
        Sb += np.outer(diff, diff) * np.sum(y == c)

    # Solve generalized eigenvalue problem: Sb v = lambda cov v
    # Use cov^-1 @ Sb
    cov_inv = np.linalg.inv(cov)
    M = cov_inv @ Sb
    eigvals, eigvecs = np.linalg.eigh(M)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    # Take top n_classes-1 LDA directions
    n_lda = min(n_classes - 1, len(dead_idx))
    lda_dirs = eigvecs[:, :n_lda].T  # (n_lda, 784)

    # If we need more directions, use PCA on raw images
    n_pca = len(dead_idx) - n_lda
    if n_pca > 0:
        # Center data
        Xc = x_flat - overall_mean
        # Compute PCA via SVD (economy)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        pca_dirs = Vt[:n_pca]  # (n_pca, 784)
        new_dirs = np.vstack([lda_dirs, pca_dirs])
    else:
        new_dirs = lda_dirs

    # Normalize new directions to have unit norm
    norms = np.linalg.norm(new_dirs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    new_dirs = new_dirs / norms

    # Replace dead units
    fc1_w[dead_idx] = new_dirs
    # Set biases to center activations: negative mean of projected data
    proj = x_flat @ new_dirs.T  # (N, len(dead_idx))
    # Use a small negative bias to allow some activation
    fc1_b[dead_idx] = -proj.mean(axis=0) * 0.5

    # Recompute hidden features for all data
    h = np.maximum(0, x_flat @ fc1_w.T + fc1_b)  # (2000, 128)

    # Refit fc2 using ridge regression on one-hot targets
    # Add bias by augmenting features
    H_aug = np.hstack([h, np.ones((N, 1), dtype=np.float32)])  # (2000, 129)
    Y_onehot = np.zeros((N, 10), dtype=np.float32)
    Y_onehot[np.arange(N), y] = 1.0

    # Ridge regression: W = (H^T H + alpha I)^-1 H^T Y
    # Use a moderate alpha to avoid overconfidence
    alpha = 1.0
    A = H_aug.T @ H_aug + alpha * np.eye(H_aug.shape[1], dtype=np.float32)
    B = H_aug.T @ Y_onehot
    W_ridge = np.linalg.solve(A, B)  # (129, 10)
    fc2_w = W_ridge[:-1].T  # (10, 128)
    fc2_b = W_ridge[-1]     # (10,)

    # Optional: scale logits to match current logit scale to improve calibration
    # Compute new logits
    logits_new = h @ fc2_w.T + fc2_b
    # Current logits std
    current_logits = f["logits"]
    current_std = current_logits.std()
    new_std = logits_new.std()
    if new_std > 1e-6:
        scale = current_std / new_std
        fc2_w *= scale
        fc2_b *= scale

    # Update weights
    w["fc1.weight"] = fc1_w.astype(np.float32)
    w["fc1.bias"] = fc1_b.astype(np.float32)
    w["fc2.weight"] = fc2_w.astype(np.float32)
    w["fc2.bias"] = fc2_b.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "replaced_units": dead_idx.tolist(),
        "num_lda": int(n_lda),
        "num_pca": int(n_pca),
        "alpha": float(alpha),
        "fc1_weight_norm": float(np.linalg.norm(fc1_w)),
        "fc2_weight_norm": float(np.linalg.norm(fc2_w)),
        "fc2_bias_norm": float(np.linalg.norm(fc2_b)),
    }

    return {"weights": w, "diagnostics": diagnostics}
