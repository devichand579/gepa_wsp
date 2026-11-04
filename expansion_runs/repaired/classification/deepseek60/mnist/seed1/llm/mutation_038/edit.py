import numpy as np
import math

def edit(w, f):
    # Extract training data
    X = f["x"].reshape(2000, 784).astype(np.float64)
    y = f["y"].astype(np.int64)
    N = X.shape[0]
    n_classes = 10
    
    # Compute class means and overall mean
    mu = np.zeros((n_classes, 784), dtype=np.float64)
    counts = np.zeros(n_classes, dtype=np.float64)
    for c in range(n_classes):
        mask = (y == c)
        counts[c] = mask.sum()
        if counts[c] > 0:
            mu[c] = X[mask].mean(axis=0)
    overall_mean = X.mean(axis=0)
    
    # Within-class scatter
    X_centered = X - mu[y]
    Sw = X_centered.T @ X_centered / N
    # Between-class scatter
    Sb = np.zeros((784, 784), dtype=np.float64)
    for c in range(n_classes):
        diff = (mu[c] - overall_mean).reshape(-1, 1)
        Sb += counts[c] * (diff @ diff.T)
    Sb /= N
    
    # Regularized inverse of Sw
    reg = 1e-3
    Sw_reg = Sw + reg * np.eye(784)
    Sw_inv = np.linalg.inv(Sw_reg)
    
    # LDA directions (columns of Sw_inv @ (mu - overall_mean).T)
    lda_dirs = Sw_inv @ (mu - overall_mean).T  # 784 x 10
    lda_dirs = lda_dirs[:, 1:]  # take 9 directions (drop first which is ~0)
    
    # PCA directions (top 128 eigenvectors of total covariance)
    C_total = (X - overall_mean).T @ (X - overall_mean) / N
    eigvals, eigvecs = np.linalg.eigh(C_total)
    idx = np.argsort(eigvals)[::-1]
    pca_dirs = eigvecs[:, idx[:128]].T  # 128 x 784
    
    # Class mean directions
    class_mean_dirs = mu - overall_mean  # 10 x 784
    
    # Pairwise class mean differences
    pairwise_dirs = []
    for i in range(n_classes):
        for j in range(i+1, n_classes):
            pairwise_dirs.append(mu[i] - mu[j])
    pairwise_dirs = np.array(pairwise_dirs)  # 45 x 784
    
    # Current fc1 rows
    current_fc1 = w["fc1.weight"].astype(np.float64)  # 128 x 784
    
    # Random projections
    rng = np.random.RandomState(42)
    random_dirs = rng.randn(100, 784)
    
    # Combine all candidates
    candidates = np.vstack([
        lda_dirs.T,          # 9 x 784
        pca_dirs,            # 128 x 784
        class_mean_dirs,     # 10 x 784
        pairwise_dirs,       # 45 x 784
        current_fc1,         # 128 x 784
        random_dirs          # 100 x 784
    ])
    # Normalize each candidate to unit norm
    norms = np.linalg.norm(candidates, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    candidates = candidates / norms
    
    # Compute F-value for each candidate
    S = X @ candidates.T  # N x num_candidates
    num_cand = candidates.shape[0]
    # Class means of scores
    S_mu = np.zeros((n_classes, num_cand))
    for c in range(n_classes):
        mask = (y == c)
        if counts[c] > 0:
            S_mu[c] = S[mask].mean(axis=0)
    S_overall = S.mean(axis=0)
    # Between-class variance
    between = np.zeros(num_cand)
    for c in range(n_classes):
        diff = S_mu[c] - S_overall
        between += counts[c] * (diff ** 2)
    between /= N
    # Within-class variance
    within = np.zeros(num_cand)
    for c in range(n_classes):
        mask = (y == c)
        if counts[c] > 0:
            diff = S[mask] - S_mu[c]
            within += (diff ** 2).sum(axis=0)
    within /= N
    # Avoid division by zero
    within = np.maximum(within, 1e-12)
    F = between / within
    
    # Select top 128 candidates
    top_idx = np.argsort(F)[::-1][:128]
    selected = candidates[top_idx]  # 128 x 784
    
    # Set new fc1 weights
    w["fc1.weight"] = selected.astype(np.float32)
    # Set fc1 bias to zero (or small negative to sparsify)
    w["fc1.bias"] = np.zeros(128, dtype=np.float32)
    
    # Compute hidden features with new fc1
    H = X @ selected.T  # N x 128
    H = np.maximum(H, 0.0)  # ReLU
    
    # Fit fc2 via ridge regression with intercept
    # Augment H with a column of ones
    H_aug = np.hstack([H, np.ones((N, 1))])
    # One-hot targets
    Y = np.zeros((N, n_classes))
    Y[np.arange(N), y] = 1.0
    
    # Ridge regression: W = (H_aug^T H_aug + lambda I)^{-1} H_aug^T Y
    # Use a small lambda for regularization
    lam = 1.0
    A = H_aug.T @ H_aug + lam * np.eye(129)
    B = H_aug.T @ Y
    W_aug = np.linalg.solve(A, B)  # 129 x 10
    
    fc2_weight = W_aug[:-1].T  # 10 x 128
    fc2_bias = W_aug[-1]       # 10
    
    w["fc2.weight"] = fc2_weight.astype(np.float32)
    w["fc2.bias"] = fc2_bias.astype(np.float32)
    
    # Diagnostics
    train_pred = np.argmax(H @ fc2_weight.T + fc2_bias, axis=1)
    train_acc = (train_pred == y).mean()
    diagnostics = {
        "train_accuracy": float(train_acc),
        "selected_indices": top_idx[:10].tolist(),
        "fc1_weight_std": float(w["fc1.weight"].std()),
        "fc2_weight_std": float(w["fc2.weight"].std()),
        "fc2_bias_mean": float(w["fc2.bias"].mean()),
    }
    return {"weights": w, "diagnostics": diagnostics}
