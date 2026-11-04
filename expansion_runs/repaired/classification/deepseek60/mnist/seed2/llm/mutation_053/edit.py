import numpy as np

def edit(w, f):
    # Extract data
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float32)  # (2000, 784)
    y = f["y"].astype(np.int64)                                 # (2000,)
    H_cur = f["features"].astype(np.float32)                    # (2000, 128)
    logits_cur = f["logits"].astype(np.float32)                 # (2000, 10)

    W1 = w["fc1.weight"].astype(np.float32).copy()              # (128, 784)
    b1 = w["fc1.bias"].astype(np.float32).copy()                # (128,)
    W2 = w["fc2.weight"].astype(np.float32).copy()              # (10, 128)
    b2 = w["fc2.bias"].astype(np.float32).copy()                # (10,)

    N, D = X.shape
    n_classes = 10
    n_units = W1.shape[0]

    # Compute importance of each fc1 unit: std of activation * norm of fc2 column
    std_act = np.std(H_cur, axis=0)                             # (128,)
    col_norm = np.linalg.norm(W2, axis=0)                       # (128,)
    importance = std_act * col_norm
    # Replace the 39 least important units
    n_replace = 39
    replace_idx = np.argsort(importance)[:n_replace]

    # Center X for PCA/LDA
    X_mean = X.mean(axis=0)
    Xc = X - X_mean

    # PCA: top 30 directions
    # Use SVD on centered data (2000 x 784)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V_pca = Vt[:30].T                                           # (784, 30)
    Z_pca = Xc @ V_pca                                          # (2000, 30)

    # LDA on PCA features: 9 directions
    # Compute class means
    mu = np.zeros((n_classes, 30), dtype=np.float32)
    for c in range(n_classes):
        mu[c] = Z_pca[y == c].mean(axis=0)
    mu_all = Z_pca.mean(axis=0)

    # Within-class scatter
    Sw = np.zeros((30, 30), dtype=np.float32)
    for c in range(n_classes):
        Zc = Z_pca[y == c] - mu[c]
        Sw += Zc.T @ Zc
    # Between-class scatter
    Sb = np.zeros((30, 30), dtype=np.float32)
    for c in range(n_classes):
        diff = (mu[c] - mu_all).reshape(-1, 1)
        Sb += (y == c).sum() * (diff @ diff.T)

    # Regularize Sw to avoid singularity
    Sw_reg = Sw + 1e-3 * np.eye(30, dtype=np.float32)
    # Solve generalized eigenproblem: Sw_reg^{-1} Sb
    # Use scipy? Not allowed. Use np.linalg.eig on Sw_reg^{-1} Sb
    Sw_inv = np.linalg.inv(Sw_reg)
    M = Sw_inv @ Sb
    eigvals, eigvecs = np.linalg.eig(M)
    # Sort by eigenvalue descending
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    # Take top 9 LDA directions in PCA space
    V_lda_pca = eigvecs[:, :9].real.astype(np.float32)          # (30, 9)
    # Map back to original 784 space
    V_lda = V_pca @ V_lda_pca                                   # (784, 9)

    # Combine: 9 LDA + 30 PCA = 39 directions
    V_new = np.concatenate([V_lda, V_pca], axis=1)              # (784, 39)

    # Normalize each direction to unit norm
    norms = np.linalg.norm(V_new, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    V_new = V_new / norms

    # Replace selected fc1 units
    for i, idx_unit in enumerate(replace_idx):
        W1[idx_unit] = V_new[:, i]
        # Set bias to center the activation: bias = -mean(X @ v)
        proj_mean = (X @ V_new[:, i]).mean()
        b1[idx_unit] = -proj_mean

    # Compute new hidden features for all training data
    H_new = np.maximum(0, X @ W1.T + b1)                        # (2000, 128)

    # Refit fc2 via ridge regression on one-hot targets
    # Augment H_new with bias column
    H_aug = np.concatenate([H_new, np.ones((N, 1), dtype=np.float32)], axis=1)  # (2000, 129)
    Y_onehot = np.zeros((N, n_classes), dtype=np.float32)
    Y_onehot[np.arange(N), y] = 1.0

    # Ridge regression: solve (H_aug^T H_aug + alpha I) W_aug = H_aug^T Y
    alpha = 1.0  # moderate regularization
    A = H_aug.T @ H_aug + alpha * np.eye(H_aug.shape[1], dtype=np.float32)
    B = H_aug.T @ Y_onehot
    W_aug = np.linalg.solve(A, B)                               # (129, 10)

    W2_new = W_aug[:128].T                                      # (10, 128)
    b2_new = W_aug[128]                                         # (10,)

    # Compute new logits on training set
    logits_new = H_new @ W2_new.T + b2_new                      # (2000, 10)

    # Scale logits to match current logit standard deviation
    std_cur = np.std(logits_cur)
    std_new = np.std(logits_new)
    if std_new > 1e-8:
        scale = std_cur / std_new
    else:
        scale = 1.0
    W2_new *= scale
    b2_new *= scale

    # Clip to safe range
    W1 = np.clip(W1, -1000, 1000).astype(np.float32)
    b1 = np.clip(b1, -1000, 1000).astype(np.float32)
    W2_new = np.clip(W2_new, -1000, 1000).astype(np.float32)
    b2_new = np.clip(b2_new, -1000, 1000).astype(np.float32)

    # Update weights
    w["fc1.weight"] = W1
    w["fc1.bias"] = b1
    w["fc2.weight"] = W2_new
    w["fc2.bias"] = b2_new

    # Diagnostics
    diagnostics = {
        "replaced_units": replace_idx.tolist(),
        "num_lda": 9,
        "num_pca": 30,
        "ridge_alpha": alpha,
        "scale_factor": float(scale),
        "fc1_weight_norm": float(np.linalg.norm(W1)),
        "fc2_weight_norm": float(np.linalg.norm(W2_new)),
        "fc2_bias_norm": float(np.linalg.norm(b2_new)),
        "new_logit_std": float(std_new * scale),
        "current_logit_std": float(std_cur)
    }

    return {"weights": w, "diagnostics": diagnostics}
