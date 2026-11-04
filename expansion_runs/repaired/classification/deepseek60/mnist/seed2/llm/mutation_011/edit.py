import numpy as np

def edit(w, f):
    # Extract data
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)  # (2000, 784)
    y = f["y"].astype(np.int64)
    N, D_in = X.shape
    C = 10
    
    # Current weights
    W1_cur = w["fc1.weight"].astype(np.float64)  # (128, 784)
    b1_cur = w["fc1.bias"].astype(np.float64)    # (128,)
    W2_cur = w["fc2.weight"].astype(np.float64)  # (10, 128)
    b2_cur = w["fc2.bias"].astype(np.float64)    # (10,)
    H_cur = f["features"].astype(np.float64)     # (2000, 128)
    
    # Helper: compute features
    def compute_features(W1, b1):
        H = X @ W1.T + b1
        return np.maximum(H, 0)
    
    # Helper: fit fc2
    def fit_fc2(H, y, method, param):
        N, D = H.shape
        if method == 'lda':
            M = np.zeros((C, D))
            for c in range(C):
                M[c] = H[y == c].mean(axis=0)
            Sw = np.zeros((D, D))
            for c in range(C):
                Hc = H[y == c]
                diff = Hc - M[c]
                Sw += diff.T @ diff
            Sw /= (N - C)
            Sw_reg = Sw + param * np.eye(D)
            W2 = np.linalg.solve(Sw_reg, M.T).T  # (C, D)
            pi = np.array([np.sum(y == c) for c in range(C)]) / N
            b2 = -0.5 * np.sum(M * W2, axis=1) + np.log(pi + 1e-12)
            return W2, b2
        elif method == 'ridge':
            Y = np.zeros((N, C))
            Y[np.arange(N), y] = 1.0
            H_aug = np.hstack([H, np.ones((N, 1))])
            A = H_aug.T @ H_aug + param * np.eye(D + 1)
            B = H_aug.T @ Y
            W_aug = np.linalg.solve(A, B)
            W2 = W_aug[:D].T
            b2 = W_aug[D]
            return W2, b2
        else:
            raise ValueError
    
    def predict(H, W2, b2):
        logits = H @ W2.T + b2
        return np.argmax(logits, axis=1)
    
    def cv_accuracy(H, y, method, param, n_folds=5):
        N = H.shape[0]
        indices = np.arange(N)
        rng = np.random.RandomState(42)
        rng.shuffle(indices)
        fold_size = N // n_folds
        accs = []
        for i in range(n_folds):
            val_idx = indices[i*fold_size:(i+1)*fold_size]
            train_idx = np.setdiff1d(indices, val_idx)
            H_train, y_train = H[train_idx], y[train_idx]
            H_val, y_val = H[val_idx], y[val_idx]
            W2, b2 = fit_fc2(H_train, y_train, method, param)
            pred = predict(H_val, W2, b2)
            accs.append(np.mean(pred == y_val))
        return np.mean(accs)
    
    # Candidate feature extractors
    candidates = []
    # 0: current fc1
    candidates.append(("current", W1_cur, b1_cur))
    
    # 1: current fc1 with some rows replaced by LDA directions from raw X
    try:
        M_x = np.zeros((C, D_in))
        for c in range(C):
            M_x[c] = X[y == c].mean(axis=0)
        Sw_x = np.zeros((D_in, D_in))
        for c in range(C):
            Xc = X[y == c]
            diff = Xc - M_x[c]
            Sw_x += diff.T @ diff
        Sw_x /= (N - C)
        # regularize
        lam_x = 1e-3
        Sw_x_reg = Sw_x + lam_x * np.eye(D_in)
        W_lda = np.linalg.solve(Sw_x_reg, M_x.T).T  # (10, 784)
        # normalize
        W_lda = W_lda / (np.linalg.norm(W_lda, axis=1, keepdims=True) + 1e-12)
        # importance of current hidden units
        imp = np.sum(np.abs(W2_cur), axis=0)
        idx_replace = np.argsort(imp)[:10]
        W1_new = W1_cur.copy()
        b1_new = b1_cur.copy()
        for i, idx in enumerate(idx_replace):
            W1_new[idx] = W_lda[i]
            b1_new[idx] = -np.mean(X @ W_lda[i])
        candidates.append(("cur+lda", W1_new, b1_new))
    except Exception:
        pass
    
    # 2: PCA on raw X
    try:
        mean_X = X.mean(axis=0)
        Xc = X - mean_X
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        W_pca = Vt[:128]
        b_pca = - (mean_X @ W_pca.T)
        candidates.append(("pca", W_pca, b_pca))
    except Exception:
        pass
    
    # 3: PCA + LDA
    try:
        mean_X = X.mean(axis=0)
        Xc = X - mean_X
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        W_pca = Vt[:118]
        # reuse W_lda from above if available
        if 'W_lda' in locals():
            W_comb = np.vstack([W_pca, W_lda])
            b_comb = - (mean_X @ W_comb.T)
            candidates.append(("pca+lda", W_comb, b_comb))
    except Exception:
        pass
    
    # Evaluate candidates
    best_acc = -1
    best_config = None
    for name, W1, b1 in candidates:
        H = compute_features(W1, b1)
        for method in ['lda', 'ridge']:
            for param in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
                acc = cv_accuracy(H, y, method, param, n_folds=5)
                if acc > best_acc:
                    best_acc = acc
                    best_config = (name, W1, b1, method, param)
    
    # Retrain on full data with best config
    name, W1_best, b1_best, method_best, param_best = best_config
    H_best = compute_features(W1_best, b1_best)
    W2_best, b2_best = fit_fc2(H_best, y, method_best, param_best)
    
    # Update weights
    w["fc1.weight"] = W1_best.astype(np.float32)
    w["fc1.bias"] = b1_best.astype(np.float32)
    w["fc2.weight"] = W2_best.astype(np.float32)
    w["fc2.bias"] = b2_best.astype(np.float32)
    
    # Diagnostics
    diagnostics = {
        "best_candidate": name,
        "best_method": method_best,
        "best_param": float(param_best),
        "cv_accuracy": float(best_acc),
        "fc1_weight_norm": float(np.linalg.norm(w["fc1.weight"])),
        "fc1_bias_norm": float(np.linalg.norm(w["fc1.bias"])),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
        "fc2_weight_max_abs": float(np.max(np.abs(w["fc2.weight"]))),
        "fc2_bias_max_abs": float(np.max(np.abs(w["fc2.bias"]))),
    }
    return {"weights": w, "diagnostics": diagnostics}
