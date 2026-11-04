import numpy as np

def edit(w, f):
    # Extract data
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float32)  # (2000, 784)
    y = f["y"].astype(np.int64)                                 # (2000,)
    N, D = X.shape
    C = 10

    # Current fc1 parameters
    W1_cur = w["fc1.weight"].astype(np.float32)  # (128, 784)
    b1_cur = w["fc1.bias"].astype(np.float32)    # (128,)

    # Helper: compute candidate features and their F-values
    def compute_f_values(W_cand, b_cand):
        # W_cand: (K, 784), b_cand: (K,)
        # Compute features: relu(X @ W_cand.T + b_cand)
        # X: (N, 784), W_cand.T: (784, K) -> (N, K)
        Z = X @ W_cand.T + b_cand[None, :]  # (N, K)
        H = np.maximum(Z, 0.0)               # (N, K)
        # Compute class means and overall mean for each feature
        # One-hot labels
        Y_onehot = np.zeros((N, C), dtype=np.float32)
        Y_onehot[np.arange(N), y] = 1.0
        # Counts per class
        counts = Y_onehot.sum(axis=0)  # (C,)
        # Class means: (C, K)
        class_means = (Y_onehot.T @ H) / counts[:, None]
        overall_mean = H.mean(axis=0)  # (K,)
        # Between-class variance: sum_c n_c (mean_c - overall_mean)^2
        between = np.sum(counts[:, None] * (class_means - overall_mean[None, :])**2, axis=0)  # (K,)
        # Within-class variance: sum_c sum_{i in c} (H_i - mean_c)^2
        # We can compute total variance minus between-class variance
        total_var = np.sum((H - overall_mean[None, :])**2, axis=0)  # (K,)
        within = total_var - between
        # F-value: (between / (C-1)) / (within / (N-C))
        # Add small epsilon to avoid division by zero
        F = (between / (C - 1)) / (within / (N - C) + 1e-12)
        return F, H

    # 1. Current fc1 rows as candidates
    W_cand_list = [W1_cur]
    b_cand_list = [b1_cur]

    # 2. LDA directions
    # Compute class means
    Y_onehot = np.zeros((N, C), dtype=np.float32)
    Y_onehot[np.arange(N), y] = 1.0
    counts = Y_onehot.sum(axis=0)
    class_means = (Y_onehot.T @ X) / counts[:, None]  # (C, 784)
    overall_mean = X.mean(axis=0)  # (784,)
    # Within-class scatter Sw
    Sw = np.zeros((D, D), dtype=np.float32)
    for c in range(C):
        idx = (y == c)
        Xc = X[idx]
        mean_c = class_means[c]
        diff = Xc - mean_c
        Sw += diff.T @ diff
    # Between-class scatter Sb
    Sb = np.zeros((D, D), dtype=np.float32)
    for c in range(C):
        diff = class_means[c] - overall_mean
        Sb += counts[c] * np.outer(diff, diff)
    # Solve generalized eigenproblem: Sb v = lambda Sw v
    # Use regularization for Sw
    Sw_reg = Sw + 1e-3 * np.eye(D, dtype=np.float32)
    # Compute inv(Sw_reg) @ Sb
    M = np.linalg.solve(Sw_reg, Sb)
    eigvals, eigvecs = np.linalg.eig(M)
    # Sort by eigenvalue descending
    idx = np.argsort(eigvals.real)[::-1]
    eigvecs = eigvecs.real[:, idx]
    # Take top 9 directions (since C-1 = 9)
    lda_dirs = eigvecs[:, :9].T  # (9, 784)
    # Normalize
    lda_dirs = lda_dirs / (np.linalg.norm(lda_dirs, axis=1, keepdims=True) + 1e-12)
    # Create positive and negative candidates
    for d in lda_dirs:
        # Center the data
        proj = X @ d
        b_pos = -proj.mean()
        W_cand_list.append(d[None, :])
        b_cand_list.append(np.array([b_pos], dtype=np.float32))
        W_cand_list.append(-d[None, :])
        b_cand_list.append(np.array([-b_pos], dtype=np.float32))

    # 3. Class-mean directions
    for c in range(C):
        d = class_means[c] - overall_mean
        d = d / (np.linalg.norm(d) + 1e-12)
        proj = X @ d
        b_pos = -proj.mean()
        W_cand_list.append(d[None, :])
        b_cand_list.append(np.array([b_pos], dtype=np.float32))
        W_cand_list.append(-d[None, :])
        b_cand_list.append(np.array([-b_pos], dtype=np.float32))

    # 4. Ridge regression directions for several lambdas
    # Center X and Y
    X_mean = X.mean(axis=0)
    X_centered = X - X_mean
    Y_mean = Y_onehot.mean(axis=0)
    Y_centered = Y_onehot - Y_mean
    # Precompute X^T X
    XTX = X_centered.T @ X_centered
    XTY = X_centered.T @ Y_centered
    lambdas = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    for lam in lambdas:
        # Solve (XTX + lam I) W = XTY
        W_ridge = np.linalg.solve(XTX + lam * np.eye(D, dtype=np.float32), XTY)  # (784, 10)
        for c in range(C):
            d = W_ridge[:, c]
            d = d / (np.linalg.norm(d) + 1e-12)
            proj = X @ d
            b_pos = -proj.mean()
            W_cand_list.append(d[None, :])
            b_cand_list.append(np.array([b_pos], dtype=np.float32))
            W_cand_list.append(-d[None, :])
            b_cand_list.append(np.array([-b_pos], dtype=np.float32))

    # 5. PCA directions (top 128)
    # Compute covariance of X_centered
    cov = X_centered.T @ X_centered / (N - 1)
    eigvals_pca, eigvecs_pca = np.linalg.eigh(cov)
    # Sort descending
    idx_pca = np.argsort(eigvals_pca)[::-1]
    eigvecs_pca = eigvecs_pca[:, idx_pca]
    pca_dirs = eigvecs_pca[:, :128].T  # (128, 784)
    for d in pca_dirs:
        proj = X @ d
        b_pos = -proj.mean()
        W_cand_list.append(d[None, :])
        b_cand_list.append(np.array([b_pos], dtype=np.float32))
        W_cand_list.append(-d[None, :])
        b_cand_list.append(np.array([-b_pos], dtype=np.float32))

    # 6. Random projections (200 directions)
    rng = np.random.RandomState(42)
    rand_dirs = rng.randn(200, D).astype(np.float32)
    rand_dirs = rand_dirs / (np.linalg.norm(rand_dirs, axis=1, keepdims=True) + 1e-12)
    for d in rand_dirs:
        proj = X @ d
        b_pos = -proj.mean()
        W_cand_list.append(d[None, :])
        b_cand_list.append(np.array([b_pos], dtype=np.float32))
        W_cand_list.append(-d[None, :])
        b_cand_list.append(np.array([-b_pos], dtype=np.float32))

    # Concatenate all candidates
    W_cand = np.vstack(W_cand_list).astype(np.float32)  # (K, 784)
    b_cand = np.concatenate(b_cand_list).astype(np.float32)  # (K,)
    K = W_cand.shape[0]

    # Compute F-values for all candidates
    F_vals, H_all = compute_f_values(W_cand, b_cand)  # H_all: (N, K)

    # Select top 128 candidates by F-value
    top_idx = np.argsort(F_vals)[::-1][:128]
    W1_new = W_cand[top_idx]  # (128, 784)
    b1_new = b_cand[top_idx]  # (128,)

    # Compute new hidden features
    Z_new = X @ W1_new.T + b1_new[None, :]
    H_new = np.maximum(Z_new, 0.0)  # (N, 128)

    # Fit fc2 using ridge regression with intercept
    # Center H_new and Y_onehot
    H_mean = H_new.mean(axis=0)
    H_centered = H_new - H_mean
    Y_mean = Y_onehot.mean(axis=0)
    Y_centered = Y_onehot - Y_mean
    # Solve for W2: (H_centered^T H_centered + lambda I) W2 = H_centered^T Y_centered
    # Use a small lambda for regularization
    lam_fc2 = 1e-3
    HTH = H_centered.T @ H_centered
    HTY = H_centered.T @ Y_centered
    W2 = np.linalg.solve(HTH + lam_fc2 * np.eye(128, dtype=np.float32), HTY)  # (128, 10)
    W2 = W2.T  # (10, 128)
    b2 = Y_mean - H_mean @ W2.T  # (10,)

    # Update weights
    w["fc1.weight"] = W1_new.astype(np.float32)
    w["fc1.bias"] = b1_new.astype(np.float32)
    w["fc2.weight"] = W2.astype(np.float32)
    w["fc2.bias"] = b2.astype(np.float32)

    # Compute diagnostics
    logits_new = H_new @ W2.T + b2[None, :]
    preds = np.argmax(logits_new, axis=1)
    acc = np.mean(preds == y)
    # Cross-entropy
    probs = np.exp(logits_new - logits_new.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    ce = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))

    diagnostics = {
        "train_accuracy": float(acc),
        "train_cross_entropy": float(ce),
        "num_candidates": int(K),
        "selected_fc1_weight_mean": float(W1_new.mean()),
        "selected_fc1_weight_std": float(W1_new.std()),
        "fc2_weight_mean": float(W2.mean()),
        "fc2_weight_std": float(W2.std()),
        "fc2_bias_mean": float(b2.mean()),
        "fc2_bias_std": float(b2.std()),
    }

    return {"weights": w, "diagnostics": diagnostics}
