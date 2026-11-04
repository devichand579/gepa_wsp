import numpy as np

def edit(w, f):
    # Load data
    X = f["x"].reshape(2000, -1).astype(np.float64)  # (2000, 784)
    y = f["y"].astype(np.int64)
    F_cur = f["features"].astype(np.float64)  # (2000, 128)
    
    classes = np.unique(y)
    n_classes = len(classes)
    n_samples = X.shape[0]
    mu = X.mean(axis=0)
    
    # Compute class means and within-class scatter
    S_w = np.zeros((784, 784))
    class_means = []
    for c in classes:
        Xc = X[y == c]
        mu_c = Xc.mean(axis=0)
        class_means.append(mu_c)
        Xc_centered = Xc - mu_c
        S_w += Xc_centered.T @ Xc_centered
    class_means = np.array(class_means)
    
    # Between-class scatter
    S_b = np.zeros((784, 784))
    for i, c in enumerate(classes):
        n_c = np.sum(y == c)
        diff = class_means[i] - mu
        S_b += n_c * np.outer(diff, diff)
    
    # Regularize S_w
    eps = 1e-3
    S_w_reg = S_w + eps * np.eye(784)
    
    # LDA directions: solve generalized eigenproblem S_b v = lambda S_w v
    L = np.linalg.cholesky(S_w_reg)
    L_inv = np.linalg.inv(L)
    M = L_inv @ S_b @ L_inv.T
    eigvals, eigvecs = np.linalg.eigh(M)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    n_lda = min(9, n_classes - 1)
    lda_dirs = L_inv.T @ eigvecs[:, :n_lda]
    lda_dirs = lda_dirs / np.linalg.norm(lda_dirs, axis=0, keepdims=True)
    
    # Within-class PCA directions
    eigvals_w, eigvecs_w = np.linalg.eigh(S_w_reg)
    idx_w = np.argsort(eigvals_w)[::-1]
    eigvecs_w = eigvecs_w[:, idx_w]
    n_pca = 128 - n_lda
    pca_dirs = eigvecs_w[:, :n_pca]
    
    # Combine into new fc1 weights
    W1 = np.hstack([lda_dirs, pca_dirs]).T  # (128, 784)
    
    # Compute bias so that projected features are centered
    proj = X @ W1.T
    b1 = -proj.mean(axis=0)
    Z = np.maximum(0, proj + b1)  # (2000, 128)
    
    # Standardize features for ridge fitting
    mu_z = Z.mean(axis=0)
    sigma_z = Z.std(axis=0) + 1e-8
    Z_norm = (Z - mu_z) / sigma_z
    
    # Ridge fitting helper
    def fit_ridge(Z_in, y_in, lam):
        n, d = Z_in.shape
        Z_aug = np.hstack([Z_in, np.ones((n, 1))])
        Y = np.zeros((n, 10))
        Y[np.arange(n), y_in] = 1.0
        A = Z_aug.T @ Z_aug + lam * np.eye(d + 1)
        B = Z_aug.T @ Y
        W = np.linalg.solve(A, B)
        return W
    
    def evaluate_cv(Z_in, y_in, lam, n_folds=5):
        n = Z_in.shape[0]
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)
        folds = np.array_split(indices, n_folds)
        accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            W = fit_ridge(Z_in[train_idx], y_in[train_idx], lam)
            Z_val_aug = np.hstack([Z_in[val_idx], np.ones((len(val_idx), 1))])
            logits = Z_val_aug @ W
            preds = np.argmax(logits, axis=1)
            accs.append(np.mean(preds == y_in[val_idx]))
        return np.mean(accs)
    
    # Evaluate new features and current features
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    best_acc_new = -1
    best_lam_new = None
    for lam in lambdas:
        acc = evaluate_cv(Z_norm, y, lam)
        if acc > best_acc_new:
            best_acc_new = acc
            best_lam_new = lam
    
    # For current features, also standardize for fair comparison
    mu_cur = F_cur.mean(axis=0)
    sigma_cur = F_cur.std(axis=0) + 1e-8
    F_cur_norm = (F_cur - mu_cur) / sigma_cur
    best_acc_cur = -1
    best_lam_cur = None
    for lam in lambdas:
        acc = evaluate_cv(F_cur_norm, y, lam)
        if acc > best_acc_cur:
            best_acc_cur = acc
            best_lam_cur = lam
    
    if best_acc_new > best_acc_cur:
        # Use new fc1 and fc2
        W2 = fit_ridge(Z_norm, y, best_lam_new)  # (129, 10)
        # Fold standardization into fc1 and fc2
        # Z_norm = (relu(X W1^T + b1) - mu_z) / sigma_z
        # We can adjust b1 and W2 to absorb mu_z and sigma_z
        # Let Z = relu(X W1^T + b1). Then Z_norm = (Z - mu_z) / sigma_z.
        # We want fc2 to compute Z_norm @ W2[:128] + W2[128].
        # This equals (Z / sigma_z) @ W2[:128] - (mu_z / sigma_z) @ W2[:128] + W2[128].
        # We can scale W1 and b1: W1_scaled = W1 / sigma_z[:, None], b1_scaled = b1 / sigma_z - mu_z / sigma_z? 
        # Actually Z = relu(X W1^T + b1). Then Z / sigma_z = relu(X (W1/sigma_z)^T + b1/sigma_z).
        # So we can set W1_new = W1 / sigma_z[:, None], b1_new = b1 / sigma_z.
        # Then fc2_weight = W2[:128].T, fc2_bias = W2[128] - (mu_z / sigma_z) @ W2[:128].
        W1_new = W1 / sigma_z[:, None]
        b1_new = b1 / sigma_z
        fc2_weight = W2[:128, :].T
        fc2_bias = W2[128, :] - (mu_z / sigma_z) @ W2[:128, :]
        
        w["fc1.weight"] = W1_new.astype(np.float32)
        w["fc1.bias"] = b1_new.astype(np.float32)
        w["fc2.weight"] = fc2_weight.astype(np.float32)
        w["fc2.bias"] = fc2_bias.astype(np.float32)
        diagnostics = {
            "used_new": True,
            "best_acc_new": float(best_acc_new),
            "best_acc_cur": float(best_acc_cur),
            "best_lam_new": float(best_lam_new),
            "best_lam_cur": float(best_lam_cur),
            "n_lda": n_lda,
            "n_pca": n_pca
        }
    else:
        # Keep current fc1, refit fc2 on current features
        W2 = fit_ridge(F_cur_norm, y, best_lam_cur)
        # Fold standardization into fc2
        # F_cur_norm = (F_cur - mu_cur) / sigma_cur
        # fc2_weight = W2[:128].T / sigma_cur[:, None]? Wait: F_cur_norm @ W2[:128] + W2[128]
        # = (F_cur - mu_cur)/sigma_cur @ W2[:128] + W2[128]
        # = F_cur @ (W2[:128]/sigma_cur[:, None]) - mu_cur/sigma_cur @ W2[:128] + W2[128]
        fc2_weight = (W2[:128, :] / sigma_cur[:, None]).T
        fc2_bias = W2[128, :] - (mu_cur / sigma_cur) @ W2[:128, :]
        w["fc2.weight"] = fc2_weight.astype(np.float32)
        w["fc2.bias"] = fc2_bias.astype(np.float32)
        diagnostics = {
            "used_new": False,
            "best_acc_new": float(best_acc_new),
            "best_acc_cur": float(best_acc_cur),
            "best_lam_new": float(best_lam_new),
            "best_lam_cur": float(best_lam_cur)
        }
    
    return {"weights": w, "diagnostics": diagnostics}
