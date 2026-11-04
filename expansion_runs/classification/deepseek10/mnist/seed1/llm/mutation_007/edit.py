import numpy as np

def edit(w, f):
    # Read current weights
    W1 = w["fc1.weight"].copy()  # [128, 784]
    b1 = w["fc1.bias"].copy()    # [128]
    W2 = w["fc2.weight"].copy()  # [10, 128]
    b2 = w["fc2.bias"].copy()    # [10]

    # Feedback data
    x = f["x"]                   # [2000, 1, 28, 28]
    y = f["y"]                   # [2000]
    N = x.shape[0]
    X = x.reshape(N, -1).astype(np.float32)  # [2000, 784]

    # Compute class means in pixel space
    classes = np.arange(10)
    mu = np.zeros((10, 784), dtype=np.float32)
    counts = np.zeros(10, dtype=np.float32)
    for c in classes:
        mask = (y == c)
        if mask.any():
            mu[c] = X[mask].mean(axis=0)
            counts[c] = mask.sum()

    # Global mean
    mu_all = X.mean(axis=0)

    # Center class means
    M = mu - mu_all  # [10, 784]

    # Compute within-class scatter in pixel space (regularized)
    # We use a diagonal approximation for efficiency and stability.
    # Compute per-pixel variance across all examples
    var = X.var(axis=0) + 1e-3
    inv_var = 1.0 / var

    # LDA-like directions: scale centered class means by inverse variance
    # This gives 10 directions in pixel space.
    D = M * inv_var[None, :]  # [10, 784]

    # We need 128 hidden units. We can use the 10 LDA directions plus
    # additional PCA directions to fill the remaining 118 units.
    # Compute PCA on centered X.
    Xc = X - mu_all
    # Use SVD on a subset for speed (2000x784 is fine)
    # We only need top 118 components.
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # Vt is [784, 784] but we only need top 118 rows.
    # Actually SVD returns U [2000, 784], S [784], Vt [784, 784].
    # We want the top 118 principal directions (rows of Vt).
    n_pca = 128 - 10
    if n_pca > 0:
        P = Vt[:n_pca, :]  # [n_pca, 784]
    else:
        P = np.zeros((0, 784), dtype=np.float32)

    # Combine LDA directions and PCA directions
    # Normalize each row to have similar scale to current W1
    # Current W1 has std ~0.0758. We'll scale to match.
    target_std = 0.0758

    def normalize_rows(A):
        norms = np.linalg.norm(A, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return A / norms

    D_norm = normalize_rows(D)
    if P.shape[0] > 0:
        P_norm = normalize_rows(P)
        new_W1 = np.vstack([D_norm, P_norm])  # [128, 784]
    else:
        # If not enough, pad with zeros
        new_W1 = np.zeros((128, 784), dtype=np.float32)
        new_W1[:10] = D_norm

    # Scale to match current W1 std
    new_W1 = new_W1 * target_std

    # Keep bias zero as before
    new_b1 = np.zeros_like(b1)

    # We keep fc2 unchanged.
    # However, we must ensure the new fc1 features are compatible with fc2.
    # The scale of new_W1 might differ from original. We can rescale fc2
    # to compensate? But we want to keep fc2 unchanged.
    # Actually, if we change fc1, the features will have different scale.
    # To preserve the logits scale, we can compute the scale factor by
    # comparing the norm of features on the feedback set.
    # We have f["features"] which are the old fc2 inputs.
    # We can compute new features for the feedback set using new_W1 and b1,
    # then compare their norms and adjust new_W1 scale so that the average
    # feature norm matches the old average feature norm.
    old_features = f["features"]  # [2000, 128]
    old_norm = np.linalg.norm(old_features, axis=1).mean()

    # Compute new features
    new_features = np.maximum(X @ new_W1.T + new_b1, 0.0)
    new_norm = np.linalg.norm(new_features, axis=1).mean()
    if new_norm > 0:
        scale = old_norm / new_norm
        new_W1 = new_W1 * scale
        new_b1 = new_b1 * scale  # bias is zero anyway

    # Final assignment
    w["fc1.weight"] = new_W1.astype(np.float32)
    w["fc1.bias"] = new_b1.astype(np.float32)
    # fc2 unchanged
    w["fc2.weight"] = W2.astype(np.float32)
    w["fc2.bias"] = b2.astype(np.float32)

    # Diagnostics
    diag = {
        "old_fc1_norm": float(np.linalg.norm(W1)),
        "new_fc1_norm": float(np.linalg.norm(new_W1)),
        "old_feature_mean_norm": float(old_norm),
        "new_feature_mean_norm": float(np.linalg.norm(new_features, axis=1).mean()),
        "scale_applied": float(scale) if 'scale' in locals() else 1.0,
        "fc2_unchanged": True
    }
    return {"weights": w, "diagnostics": diag}
