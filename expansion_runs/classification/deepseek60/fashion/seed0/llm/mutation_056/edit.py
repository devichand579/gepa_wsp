import numpy as np

def edit(w, f):
    # Extract features and labels from the feedback pool
    X = f["features"]  # shape (2000, 1568)
    y = f["y"]         # shape (2000,)
    N, p = X.shape
    classes = np.unique(y)
    C = len(classes)
    
    # Compute class means
    mu_c = np.zeros((C, p), dtype=np.float64)
    for c in classes:
        mu_c[c] = X[y == c].mean(axis=0)
    
    # Compute within-class scatter matrix
    X_centered = X - mu_c[y]
    S_w = X_centered.T @ X_centered  # shape (p, p)
    
    # Cross-validation to select shrinkage parameter alpha
    # We'll use a simple 5-fold CV on the feedback pool.
    # For each alpha, compute LDA on training folds and evaluate on validation fold.
    alphas = np.logspace(-4, 2, 20)  # from 1e-4 to 100
    best_alpha = None
    best_acc = -1.0
    
    # Precompute fold indices
    np.random.seed(42)
    indices = np.random.permutation(N)
    n_folds = 5
    fold_size = N // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else N
        folds.append(indices[start:end])
    
    for alpha in alphas:
        accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]
            
            # Compute class means on training data
            mu_c_train = np.zeros((C, p), dtype=np.float64)
            for c in classes:
                mu_c_train[c] = X_train[y_train == c].mean(axis=0)
            
            # Compute within-class scatter on training data
            X_train_centered = X_train - mu_c_train[y_train]
            S_w_train = X_train_centered.T @ X_train_centered
            
            # Regularize: S = S_w_train / (N_train - C) + alpha * I
            # Actually, we can just use S_w_train + alpha * I * (N_train - C) to keep scale.
            # But for simplicity, use S = S_w_train + alpha * np.eye(p) * (N_train - C)
            # However, the scale of S_w_train depends on N_train. Let's use S = S_w_train / (N_train - C) + alpha * np.eye(p)
            # This is equivalent to shrinkage towards identity with variance alpha.
            N_train = len(train_idx)
            S = S_w_train / (N_train - C) + alpha * np.eye(p)
            
            # Solve for weights: W = S^{-1} mu_c^T
            # We can solve S * W = mu_c_train.T
            W = np.linalg.solve(S, mu_c_train.T).T  # shape (C, p)
            # Bias: b_c = -0.5 * mu_c^T S^{-1} mu_c + log(prior_c)
            # Compute mu_c^T W_c for each class
            # mu_c_train is (C, p), W is (C, p). We need sum over p of mu_c * W_c.
            # Actually, mu_c^T S^{-1} mu_c = mu_c^T W_c (since W_c = S^{-1} mu_c)
            # So b_c = -0.5 * np.sum(mu_c_train * W, axis=1) + np.log(prior_c)
            priors = np.array([np.mean(y_train == c) for c in classes])
            b = -0.5 * np.sum(mu_c_train * W, axis=1) + np.log(priors)
            
            # Evaluate on validation
            logits = X_val @ W.T + b
            preds = np.argmax(logits, axis=1)
            acc = np.mean(preds == y_val)
            accs.append(acc)
        
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_alpha = alpha
    
    # Train final LDA on all data with best_alpha
    S = S_w / (N - C) + best_alpha * np.eye(p)
    W = np.linalg.solve(S, mu_c.T).T  # shape (C, p)
    priors = np.array([np.mean(y == c) for c in classes])
    b = -0.5 * np.sum(mu_c * W, axis=1) + np.log(priors)
    
    # Update the head weights and biases
    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)
    
    # Diagnostics
    diagnostics = {
        "best_alpha": float(best_alpha),
        "cv_accuracy": float(best_acc),
        "head_weight_norm": float(np.linalg.norm(w["head.weight"])),
        "head_bias_norm": float(np.linalg.norm(w["head.bias"])),
        "head_weight_max_abs": float(np.max(np.abs(w["head.weight"]))),
        "head_bias_max_abs": float(np.max(np.abs(w["head.bias"]))),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
