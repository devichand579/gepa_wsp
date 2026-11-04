import numpy as np
import math

def edit(w, f):
    # ------------------------------------------------------------------
    # 1. Extract feedback data
    # ------------------------------------------------------------------
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)  # (2000, 784)
    y = f["y"].astype(np.int64)
    N = X.shape[0]
    num_classes = 10

    # Current fc1 parameters
    W1 = w["fc1.weight"].astype(np.float64)  # (128, 784)
    b1 = w["fc1.bias"].astype(np.float64)    # (128,)

    # ------------------------------------------------------------------
    # 2. Compute class means and within-class scatter
    # ------------------------------------------------------------------
    mu = np.mean(X, axis=0)  # (784,)
    mu_c = np.zeros((num_classes, 784))
    n_c = np.zeros(num_classes)
    for c in range(num_classes):
        mask = (y == c)
        n_c[c] = np.sum(mask)
        if n_c[c] > 0:
            mu_c[c] = np.mean(X[mask], axis=0)

    # Sw = X^T X - sum_c n_c * mu_c mu_c^T
    XtX = X.T @ X
    Sw = XtX.copy()
    for c in range(num_classes):
        if n_c[c] > 0:
            Sw -= n_c[c] * np.outer(mu_c[c], mu_c[c])

    # Ridge regularisation for numerical stability
    Sw_reg = Sw + 1e-3 * np.eye(784)

    # ------------------------------------------------------------------
    # 3. LDA directions (up to 9 non-zero)
    # ------------------------------------------------------------------
    M = np.zeros((784, num_classes))
    for c in range(num_classes):
        M[:, c] = mu_c[c] - mu

    # Y = Sw_reg^{-1} M
    Y = np.linalg.solve(Sw_reg, M)          # (784, 10)
    A = M.T @ Y                             # (10, 10)
    eigvals, eigvecs = np.linalg.eigh(A)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    lda_dirs = []
    for i in range(min(9, num_classes - 1)):
        u = eigvecs[:, i]
        v = Y @ u
        v = v / (np.linalg.norm(v) + 1e-12)
        lda_dirs.append(v)
    lda_dirs = np.array(lda_dirs)           # (9, 784)

    # ------------------------------------------------------------------
    # 4. PCA directions for the remaining units
    # ------------------------------------------------------------------
    C = (X - mu).T @ (X - mu) / N
    eigvals_pca, eigvecs_pca = np.linalg.eigh(C)
    idx_pca = np.argsort(eigvals_pca)[::-1]
    eigvecs_pca = eigvecs_pca[:, idx_pca]

    n_pca = 39 - len(lda_dirs)
    pca_dirs = eigvecs_pca[:, :n_pca].T     # (n_pca, 784)

    # Combine all new directions
    new_dirs = np.vstack([lda_dirs, pca_dirs])  # (39, 784)

    # ------------------------------------------------------------------
    # 5. Replace selected fc1 units
    # ------------------------------------------------------------------
    # Units to replace (from previous successful proposal)
    replace_units = [0, 96, 36, 24, 72, 108, 48, 84, 60, 12, 3, 26, 111, 15,
                     124, 74, 78, 120, 64, 41, 79, 103, 126, 17, 123, 116, 42,
                     4, 38, 5, 76, 73, 37, 50, 86, 31, 77, 106, 49]
    K = len(replace_units)

    W1_new = W1.copy()
    b1_new = b1.copy()

    for i, unit in enumerate(replace_units):
        if i < len(new_dirs):
            d = new_dirs[i]
            W1_new[unit] = d
            # Set bias so that the unit is active for ~half the training samples
            proj = X @ d
            b1_new[unit] = -np.median(proj)

    # ------------------------------------------------------------------
    # 6. Compute new hidden features
    # ------------------------------------------------------------------
    H = np.maximum(0, X @ W1_new.T + b1_new)  # (2000, 128)

    # ------------------------------------------------------------------
    # 7. Closed-form ridge regression for fc2
    # ------------------------------------------------------------------
    alpha = 1e-4
    H_aug = np.hstack([H, np.ones((N, 1))])   # (2000, 129)
    Y_onehot = np.eye(num_classes)[y]         # (2000, 10)

    # Solve (H_aug^T H_aug + alpha I) W_aug = H_aug^T Y_onehot
    A_mat = H_aug.T @ H_aug + alpha * np.eye(H_aug.shape[1])
    B_mat = H_aug.T @ Y_onehot
    W_aug = np.linalg.solve(A_mat, B_mat)     # (129, 10)

    W2_new = W_aug[:128].T                    # (10, 128)
    b2_new = W_aug[128]                       # (10,)

    # ------------------------------------------------------------------
    # 8. Update weights and clip to safe range
    # ------------------------------------------------------------------
    w["fc1.weight"] = np.clip(W1_new, -1000, 1000).astype(np.float32)
    w["fc1.bias"]   = np.clip(b1_new, -1000, 1000).astype(np.float32)
    w["fc2.weight"] = np.clip(W2_new, -1000, 1000).astype(np.float32)
    w["fc2.bias"]   = np.clip(b2_new, -1000, 1000).astype(np.float32)

    # ------------------------------------------------------------------
    # 9. Diagnostics
    # ------------------------------------------------------------------
    # Full-pool accuracy
    logits_full = H @ W2_new.T + b2_new
    pred_full = np.argmax(logits_full, axis=1)
    acc_full = np.mean(pred_full == y)

    # Paired batch accuracy
    X_batch = f["batch.x"].reshape(f["batch.x"].shape[0], -1).astype(np.float64)
    y_batch = f["batch.y"].astype(np.int64)
    H_batch = np.maximum(0, X_batch @ W1_new.T + b1_new)
    logits_batch = H_batch @ W2_new.T + b2_new
    pred_batch = np.argmax(logits_batch, axis=1)
    acc_batch = np.mean(pred_batch == y_batch)

    diagnostics = {
        "full_pool_accuracy": float(acc_full),
        "batch_accuracy": float(acc_batch),
        "replaced_units": replace_units,
        "num_lda_dirs": len(lda_dirs),
        "num_pca_dirs": n_pca,
        "ridge_alpha": alpha,
        "fc1_weight_norm": float(np.linalg.norm(w["fc1.weight"])),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
    }

    return {"weights": w, "diagnostics": diagnostics}
