def edit(w, f):
    import numpy as np
    
    # Load full feedback pool features and labels
    X = f["features"].astype(np.float64)  # (2000, 1568)
    y = f["y"].astype(np.int64)           # (2000,)
    n_classes = 10
    n_features = X.shape[1]
    n_samples = X.shape[0]
    
    # Compute class means and priors
    mu = np.zeros((n_classes, n_features))
    priors = np.zeros(n_classes)
    for c in range(n_classes):
        idx = (y == c)
        mu[c] = X[idx].mean(axis=0)
        priors[c] = idx.sum() / n_samples
    
    # Compute within-class scatter S_w
    S_w = np.zeros((n_features, n_features))
    for c in range(n_classes):
        idx = (y == c)
        Xc = X[idx]
        Xc_centered = Xc - mu[c]
        S_w += Xc_centered.T @ Xc_centered
    
    # Function to fit LDA with shrinkage alpha
    def lda_fit(alpha):
        mean_var = np.trace(S_w) / n_features
        S = (1 - alpha) * S_w + alpha * mean_var * np.eye(n_features)
        W = np.linalg.solve(S, mu.T)  # (n_features, n_classes)
        b = np.zeros(n_classes)
        for c in range(n_classes):
            b[c] = -0.5 * mu[c] @ W[:, c] + np.log(priors[c] + 1e-12)
        return W.T, b  # (n_classes, n_features), (n_classes,)
    
    # Cross-validation to select alpha
    alphas = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
    best_alpha = 0.1
    best_cv_acc = 0.0
    
    n_folds = 5
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n_samples // n_folds
    
    for alpha in alphas:
        cv_acc = 0.0
        for fold in range(n_folds):
            val_idx = indices[fold*fold_size:(fold+1)*fold_size]
            train_idx = np.setdiff1d(indices, val_idx)
            
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]
            
            # Compute S_w on train
            mu_train = np.zeros((n_classes, n_features))
            priors_train = np.zeros(n_classes)
            for c in range(n_classes):
                idx = (y_train == c)
                mu_train[c] = X_train[idx].mean(axis=0)
                priors_train[c] = idx.sum() / len(train_idx)
            
            S_w_train = np.zeros((n_features, n_features))
            for c in range(n_classes):
                idx = (y_train == c)
                Xc = X_train[idx]
                Xc_centered = Xc - mu_train[c]
                S_w_train += Xc_centered.T @ Xc_centered
            
            mean_var = np.trace(S_w_train) / n_features
            S = (1 - alpha) * S_w_train + alpha * mean_var * np.eye(n_features)
            W = np.linalg.solve(S, mu_train.T)
            b = np.zeros(n_classes)
            for c in range(n_classes):
                b[c] = -0.5 * mu_train[c] @ W[:, c] + np.log(priors_train[c] + 1e-12)
            
            logits = X_val @ W + b
            preds = np.argmax(logits, axis=1)
            cv_acc += (preds == y_val).mean()
        
        cv_acc /= n_folds
        if cv_acc > best_cv_acc:
            best_cv_acc = cv_acc
            best_alpha = alpha
    
    # Fit on full data with best_alpha
    mean_var = np.trace(S_w) / n_features
    S = (1 - best_alpha) * S_w + best_alpha * mean_var * np.eye(n_features)
    W = np.linalg.solve(S, mu.T)  # (n_features, n_classes)
    b = np.zeros(n_classes)
    for c in range(n_classes):
        b[c] = -0.5 * mu[c] @ W[:, c] + np.log(priors[c] + 1e-12)
    
    # Update head weights only
    w["head.weight"] = W.T.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)
    
    # Diagnostics
    logits_full = X @ W + b
    preds_full = np.argmax(logits_full, axis=1)
    acc_full = (preds_full == y).mean()
    
    batch_ids = f["batch.ids"]
    X_batch = X[batch_ids]
    y_batch = y[batch_ids]
    logits_batch = X_batch @ W + b
    preds_batch = np.argmax(logits_batch, axis=1)
    acc_batch = (preds_batch == y_batch).mean()
    
    diagnostics = {
        "best_alpha": float(best_alpha),
        "cv_acc": float(best_cv_acc),
        "full_pool_acc": float(acc_full),
        "paired_batch_acc": float(acc_batch),
        "head_weight_std": float(w["head.weight"].std()),
        "head_bias_mean": float(w["head.bias"].mean())
    }
    
    return {"weights": w, "diagnostics": diagnostics}
