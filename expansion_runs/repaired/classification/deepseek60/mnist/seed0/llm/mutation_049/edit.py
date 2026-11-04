import numpy as np

def edit(w, f):
    # Extract data
    X = f["x"]  # (2000, 1, 28, 28)
    y = f["y"]  # (2000,)
    N = X.shape[0]
    X_flat = X.reshape(N, -1).astype(np.float32)  # (2000, 784)
    
    # Current weights
    W1 = w["fc1.weight"].copy()  # (128, 784)
    b1 = w["fc1.bias"].copy()    # (128,)
    W2 = w["fc2.weight"].copy()  # (10, 128)
    b2 = w["fc2.bias"].copy()    # (10,)
    
    # Helper: forward pass
    def forward(W1, b1, W2, b2, X):
        H = np.maximum(0, X @ W1.T + b1)  # (N, 128)
        logits = H @ W2.T + b2            # (N, 10)
        return H, logits
    
    # Evaluate current accuracy on full pool
    _, logits_cur = forward(W1, b1, W2, b2, X_flat)
    pred_cur = np.argmax(logits_cur, axis=1)
    acc_cur = np.mean(pred_cur == y)
    
    # We will try several candidate fc1/fc2 combinations and pick the best on full pool.
    # Candidate 1: Keep fc1, fit fc2 with LDA (shrinkage)
    # Candidate 2: Keep fc1, fit fc2 with ridge regression (standardized)
    # Candidate 3: Replace fc1 with class means + PCA, fit fc2 with LDA
    
    best_W1, best_b1, best_W2, best_b2 = W1, b1, W2, b2
    best_acc = acc_cur
    best_ce = -np.mean(np.log(np.exp(logits_cur - logits_cur.max(axis=1, keepdims=True)) / 
                              np.sum(np.exp(logits_cur - logits_cur.max(axis=1, keepdims=True)), axis=1))[np.arange(N), y])
    
    # --- Candidate 1: LDA on current features ---
    H_cur, _ = forward(W1, b1, W2, b2, X_flat)
    # Compute class means and pooled covariance
    classes = np.unique(y)
    means = np.array([H_cur[y == c].mean(axis=0) for c in classes])  # (10, 128)
    overall_mean = H_cur.mean(axis=0)  # (128,)
    # Pooled covariance
    cov = np.zeros((128, 128), dtype=np.float64)
    for c in classes:
        Hc = H_cur[y == c]
        Hc_centered = Hc - means[c]
        cov += Hc_centered.T @ Hc_centered
    cov /= (N - len(classes))
    # Shrinkage
    alpha = 0.1
    cov_shrunk = (1 - alpha) * cov + alpha * np.trace(cov) / 128 * np.eye(128)
    # LDA weights: W2 = cov^{-1} @ (means - overall_mean).T
    # Add small ridge for numerical stability
    cov_reg = cov_shrunk + 1e-6 * np.eye(128)
    W2_lda = np.linalg.solve(cov_reg, (means - overall_mean).T).T  # (10, 128)
    b2_lda = -W2_lda @ overall_mean
    # Evaluate
    _, logits_lda = forward(W1, b1, W2_lda, b2_lda, X_flat)
    pred_lda = np.argmax(logits_lda, axis=1)
    acc_lda = np.mean(pred_lda == y)
    if acc_lda > best_acc:
        best_acc = acc_lda
        best_W2, best_b2 = W2_lda, b2_lda
    
    # --- Candidate 2: Ridge regression on standardized current features ---
    # Standardize features
    H_mean = H_cur.mean(axis=0)
    H_std = H_cur.std(axis=0) + 1e-6
    H_stdized = (H_cur - H_mean) / H_std
    # One-hot targets
    Y_onehot = np.zeros((N, 10), dtype=np.float64)
    Y_onehot[np.arange(N), y] = 1.0
    # Ridge with several lambdas
    for lam in [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
        # Solve (H^T H + lam I) W = H^T Y
        A = H_stdized.T @ H_stdized + lam * np.eye(128)
        B = H_stdized.T @ Y_onehot
        W_ridge = np.linalg.solve(A, B).T  # (10, 128)
        b_ridge = Y_onehot.mean(axis=0) - W_ridge @ H_mean  # approximate
        # Adjust for standardization: logits = W_ridge @ ((H - H_mean)/H_std) + b
        # = (W_ridge / H_std) @ H + (b - W_ridge @ (H_mean/H_std))
        W2_ridge = W_ridge / H_std
        b2_ridge = b_ridge - W2_ridge @ H_mean
        _, logits_ridge = forward(W1, b1, W2_ridge, b2_ridge, X_flat)
        pred_ridge = np.argmax(logits_ridge, axis=1)
        acc_ridge = np.mean(pred_ridge == y)
        if acc_ridge > best_acc:
            best_acc = acc_ridge
            best_W2, best_b2 = W2_ridge, b2_ridge
    
    # --- Candidate 3: New fc1 with class means + PCA, then LDA on new features ---
    # Compute class mean images
    class_means = np.array([X_flat[y == c].mean(axis=0) for c in classes])  # (10, 784)
    # Compute PCA on all images (centered)
    X_centered = X_flat - X_flat.mean(axis=0)
    # Use SVD for PCA
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # Vt is (min(N,784), 784). Take top 118 components
    n_pca = 118
    pca_components = Vt[:n_pca]  # (118, 784)
    # New fc1 weights: 10 class means + 118 PCA components
    W1_new = np.vstack([class_means, pca_components]).astype(np.float32)  # (128, 784)
    # Biases: set to 0 for class means? Actually we can set bias to -0.5 * norm^2? 
    # Better: set bias to 0 and let fc2 handle. But we can center the features by subtracting mean activation.
    # Compute new features
    H_new = np.maximum(0, X_flat @ W1_new.T)  # (2000, 128)
    # Standardize new features
    H_new_mean = H_new.mean(axis=0)
    H_new_std = H_new.std(axis=0) + 1e-6
    H_new_stdized = (H_new - H_new_mean) / H_new_std
    # Fit LDA on new features
    means_new = np.array([H_new_stdized[y == c].mean(axis=0) for c in classes])
    overall_mean_new = H_new_stdized.mean(axis=0)
    cov_new = np.zeros((128, 128), dtype=np.float64)
    for c in classes:
        Hc = H_new_stdized[y == c]
        Hc_centered = Hc - means_new[c]
        cov_new += Hc_centered.T @ Hc_centered
    cov_new /= (N - len(classes))
    alpha = 0.1
    cov_new_shrunk = (1 - alpha) * cov_new + alpha * np.trace(cov_new) / 128 * np.eye(128)
    cov_new_reg = cov_new_shrunk + 1e-6 * np.eye(128)
    W2_new_lda = np.linalg.solve(cov_new_reg, (means_new - overall_mean_new).T).T
    b2_new_lda = -W2_new_lda @ overall_mean_new
    # Convert back to original feature space
    W2_new = W2_new_lda / H_new_std
    b2_new = b2_new_lda - W2_new @ H_new_mean
    # Evaluate
    _, logits_new = forward(W1_new, np.zeros(128, dtype=np.float32), W2_new, b2_new, X_flat)
    pred_new = np.argmax(logits_new, axis=1)
    acc_new = np.mean(pred_new == y)
    if acc_new > best_acc:
        best_acc = acc_new
        best_W1, best_b1 = W1_new, np.zeros(128, dtype=np.float32)
        best_W2, best_b2 = W2_new, b2_new
    
    # --- Candidate 4: Keep current fc1, but add class-mean templates to dead units? 
    # We'll skip to keep it simple.
    
    # Prepare final weights
    w_out = {
        "fc1.weight": best_W1.astype(np.float32),
        "fc1.bias": best_b1.astype(np.float32),
        "fc2.weight": best_W2.astype(np.float32),
        "fc2.bias": best_b2.astype(np.float32),
    }
    
    # Diagnostics
    diag = {
        "acc_cur": float(acc_cur),
        "acc_best": float(best_acc),
        "method": "lda_ridge_pca",
        "W1_norm": float(np.linalg.norm(best_W1)),
        "W2_norm": float(np.linalg.norm(best_W2)),
    }
    return {"weights": w_out, "diagnostics": diag}
