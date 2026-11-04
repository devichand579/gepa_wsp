def edit(w, f):
    import numpy as np
    
    X = f["features"].astype(np.float64)
    y = f["y"].astype(np.int64)
    n, d = X.shape
    n_classes = 10
    
    W0 = w["head.weight"].astype(np.float64)
    b0 = w["head.bias"].astype(np.float64)
    
    # One-hot targets
    Y = np.zeros((n, n_classes))
    Y[np.arange(n), y] = 1.0
    
    # Augment X with bias
    X_aug = np.hstack([X, np.ones((n, 1))])
    W0_aug = np.vstack([W0.T, b0[None, :]])  # (d+1, 10)
    
    # CV setup
    np.random.seed(42)
    indices = np.random.permutation(n)
    n_folds = 5
    fold_size = n // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = n if i == n_folds - 1 else (i + 1) * fold_size
        val_idx = indices[start:end]
        train_idx = np.setdiff1d(indices, val_idx)
        folds.append((train_idx, val_idx))
    
    def eval_ridge(lam, use_prior=False):
        accs = []
        for train_idx, val_idx in folds:
            X_tr = X_aug[train_idx]
            Y_tr = Y[train_idx]
            X_val = X_aug[val_idx]
            y_val = y[val_idx]
            if use_prior:
                A = X_tr.T @ X_tr + lam * np.eye(d + 1)
                B = X_tr.T @ Y_tr + lam * W0_aug
            else:
                A = X_tr.T @ X_tr + lam * np.eye(d + 1)
                B = X_tr.T @ Y_tr
            W_aug = np.linalg.solve(A, B)
            logits = X_val @ W_aug
            pred = np.argmax(logits, axis=1)
            acc = np.mean(pred == y_val)
            accs.append(acc)
        return np.mean(accs)
    
    def eval_lda(alpha):
        accs = []
        for train_idx, val_idx in folds:
            X_tr = X[train_idx]
            y_tr = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]
            n_tr = len(train_idx)
            # class means
            mu = np.zeros((n_classes, d))
            priors = np.zeros(n_classes)
            for c in range(n_classes):
                mask = (y_tr == c)
                mu[c] = X_tr[mask].mean(axis=0)
                priors[c] = mask.sum() / n_tr
            # pooled covariance using centered data
            mu_y = mu[y_tr]
            X_centered = X_tr - mu_y
            Sigma = (X_centered.T @ X_centered) / (n_tr - n_classes)
            # shrinkage
            trace = np.trace(Sigma)
            Sigma_shrunk = (1 - alpha) * Sigma + alpha * (trace / d) * np.eye(d)
            # solve for W and bias
            inv_Sigma_mu = np.linalg.solve(Sigma_shrunk, mu.T)  # (d, n_classes)
            W = inv_Sigma_mu.T  # (n_classes, d)
            b = -0.5 * np.sum(mu * inv_Sigma_mu.T, axis=1) + np.log(priors + 1e-12)
            logits = X_val @ W.T + b
            pred = np.argmax(logits, axis=1)
            acc = np.mean(pred == y_val)
            accs.append(acc)
        return np.mean(accs)
    
    def eval_nearest_centroid():
        accs = []
        for train_idx, val_idx in folds:
            X_tr = X[train_idx]
            y_tr = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]
            n_tr = len(train_idx)
            mu = np.zeros((n_classes, d))
            priors = np.zeros(n_classes)
            for c in range(n_classes):
                mask = (y_tr == c)
                mu[c] = X_tr[mask].mean(axis=0)
                priors[c] = mask.sum() / n_tr
            W = 2 * mu
            b = -np.sum(mu ** 2, axis=1) + np.log(priors + 1e-12)
            logits = X_val @ W.T + b
            pred = np.argmax(logits, axis=1)
            acc = np.mean(pred == y_val)
            accs.append(acc)
        return np.mean(accs)
    
    # Evaluate methods
    best_acc = -1
    best_method = None
    best_params = None
    
    # Ridge lambdas
    lambdas = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
    for lam in lambdas:
        acc = eval_ridge(lam, use_prior=False)
        if acc > best_acc:
            best_acc = acc
            best_method = "ridge"
            best_params = {"lambda": lam}
        acc = eval_ridge(lam, use_prior=True)
        if acc > best_acc:
            best_acc = acc
            best_method = "ridge_prior"
            best_params = {"lambda": lam}
    
    # LDA alphas
    alphas = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.95, 0.99]
    for alpha in alphas:
        acc = eval_lda(alpha)
        if acc > best_acc:
            best_acc = acc
            best_method = "lda"
            best_params = {"alpha": alpha}
    
    # Nearest centroid
    acc = eval_nearest_centroid()
    if acc > best_acc:
        best_acc = acc
        best_method = "nearest_centroid"
        best_params = {}
    
    # Fit on full data with best method
    if best_method == "ridge":
        lam = best_params["lambda"]
        A = X_aug.T @ X_aug + lam * np.eye(d + 1)
        B = X_aug.T @ Y
        W_aug = np.linalg.solve(A, B)
        W_new = W_aug[:-1, :].T
        b_new = W_aug[-1, :]
    elif best_method == "ridge_prior":
        lam = best_params["lambda"]
        A = X_aug.T @ X_aug + lam * np.eye(d + 1)
        B = X_aug.T @ Y + lam * W0_aug
        W_aug = np.linalg.solve(A, B)
        W_new = W_aug[:-1, :].T
        b_new = W_aug[-1, :]
    elif best_method == "lda":
        alpha = best_params["alpha"]
        mu = np.zeros((n_classes, d))
        priors = np.zeros(n_classes)
        for c in range(n_classes):
            mask = (y == c)
            mu[c] = X[mask].mean(axis=0)
            priors[c] = mask.sum() / n
        mu_y = mu[y]
        X_centered = X - mu_y
        Sigma = (X_centered.T @ X_centered) / (n - n_classes)
        trace = np.trace(Sigma)
        Sigma_shrunk = (1 - alpha) * Sigma + alpha * (trace / d) * np.eye(d)
        inv_Sigma_mu = np.linalg.solve(Sigma_shrunk, mu.T)
        W_new = inv_Sigma_mu.T
        b_new = -0.5 * np.sum(mu * inv_Sigma_mu.T, axis=1) + np.log(priors + 1e-12)
    elif best_method == "nearest_centroid":
        mu = np.zeros((n_classes, d))
        priors = np.zeros(n_classes)
        for c in range(n_classes):
            mask = (y == c)
            mu[c] = X[mask].mean(axis=0)
            priors[c] = mask.sum() / n
        W_new = 2 * mu
        b_new = -np.sum(mu ** 2, axis=1) + np.log(priors + 1e-12)
    
    # Clip to ensure finite and <=1000
    W_new = np.clip(W_new, -1000, 1000)
    b_new = np.clip(b_new, -1000, 1000)
    
    # Update weights
    w["head.weight"] = W_new.astype(np.float32)
    w["head.bias"] = b_new.astype(np.float32)
    
    # Compute training accuracy with new head
    logits_train = X @ W_new.T + b_new
    pred_train = np.argmax(logits_train, axis=1)
    train_acc = np.mean(pred_train == y)
    
    diagnostics = {
        "best_method": best_method,
        "best_params": best_params,
        "cv_accuracy": float(best_acc),
        "train_accuracy": float(train_acc),
        "W_mean": float(np.mean(W_new)),
        "W_std": float(np.std(W_new)),
        "b_mean": float(np.mean(b_new)),
        "b_std": float(np.std(b_new))
    }
    
    return {"weights": w, "diagnostics": diagnostics}
