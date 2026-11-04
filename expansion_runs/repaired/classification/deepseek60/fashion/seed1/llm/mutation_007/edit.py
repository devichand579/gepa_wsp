import numpy as np

def edit(w, f):
    X = f["features"].astype(np.float64)  # (2000, 1568)
    y = f["y"].astype(np.int64)
    N, D = X.shape
    num_classes = 10
    
    # Current head
    W0 = w["head.weight"].astype(np.float64)  # (10, 1568)
    b0 = w["head.bias"].astype(np.float64)    # (10,)
    
    # Standardize features
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1e-6
    Xs = (X - mean) / std
    
    # One-hot targets
    Y = np.zeros((N, num_classes))
    Y[np.arange(N), y] = 1.0
    
    # Convert current head to standardized space
    W0_std = W0 * std
    b0_std = b0 + W0 @ mean
    
    # 5-fold CV setup
    n_folds = 5
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    lambdas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
    
    best_acc = -1
    best_method = None
    best_lambda = None
    
    # Precompute fold data
    fold_data = []
    for i in range(n_folds):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        X_train = Xs[train_idx]
        Y_train = Y[train_idx]
        X_val = Xs[val_idx]
        Y_val = Y[val_idx]
        y_val = y[val_idx]
        
        X_train_aug = np.hstack([X_train, np.ones((len(train_idx), 1))])
        X_val_aug = np.hstack([X_val, np.ones((len(val_idx), 1))])
        
        # LDA statistics
        mu_c = np.zeros((num_classes, D))
        n_c = np.zeros(num_classes)
        for c in range(num_classes):
            mask = (y[train_idx] == c)
            n_c[c] = mask.sum()
            if n_c[c] > 0:
                mu_c[c] = X_train[mask].mean(axis=0)
        XtX = X_train.T @ X_train
        Sw = XtX - sum(n_c[c] * np.outer(mu_c[c], mu_c[c]) for c in range(num_classes))
        
        fold_data.append({
            'X_train_aug': X_train_aug,
            'Y_train': Y_train,
            'X_val_aug': X_val_aug,
            'y_val': y_val,
            'Sw': Sw,
            'mu_c': mu_c,
            'n_c': n_c
        })
    
    # Evaluate methods
    for method in ['ridge', 'ridge_prior', 'lda']:
        for lam in lambdas:
            accs = []
            for fd in fold_data:
                if method == 'ridge':
                    X_aug = fd['X_train_aug']
                    Y_train = fd['Y_train']
                    XtX = X_aug.T @ X_aug
                    XtY = X_aug.T @ Y_train
                    W_aug = np.linalg.solve(XtX + lam * np.eye(XtX.shape[0]), XtY)
                elif method == 'ridge_prior':
                    W0_aug = np.vstack([W0_std.T, b0_std.reshape(1, -1)])
                    X_aug = fd['X_train_aug']
                    Y_train = fd['Y_train']
                    XtX = X_aug.T @ X_aug
                    XtY = X_aug.T @ Y_train
                    W_aug = np.linalg.solve(XtX + lam * np.eye(XtX.shape[0]), XtY + lam * W0_aug)
                elif method == 'lda':
                    Sw = fd['Sw']
                    mu_c = fd['mu_c']
                    n_c = fd['n_c']
                    Sw_reg = Sw + lam * np.eye(D)
                    W = np.linalg.solve(Sw_reg, mu_c.T)
                    pi = n_c / n_c.sum()
                    b = -0.5 * np.sum(mu_c * W.T, axis=1) + np.log(pi + 1e-12)
                    W_aug = np.vstack([W, b.reshape(1, -1)])
                
                logits = fd['X_val_aug'] @ W_aug
                preds = np.argmax(logits, axis=1)
                acc = (preds == fd['y_val']).mean()
                accs.append(acc)
            mean_acc = np.mean(accs)
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_method = method
                best_lambda = lam
    
    # Retrain on full data
    X_aug_full = np.hstack([Xs, np.ones((N, 1))])
    if best_method == 'ridge':
        XtX = X_aug_full.T @ X_aug_full
        XtY = X_aug_full.T @ Y
        W_aug = np.linalg.solve(XtX + best_lambda * np.eye(XtX.shape[0]), XtY)
    elif best_method == 'ridge_prior':
        W0_aug = np.vstack([W0_std.T, b0_std.reshape(1, -1)])
        XtX = X_aug_full.T @ X_aug_full
        XtY = X_aug_full.T @ Y
        W_aug = np.linalg.solve(XtX + best_lambda * np.eye(XtX.shape[0]), XtY + best_lambda * W0_aug)
    elif best_method == 'lda':
        mu_c = np.zeros((num_classes, D))
        n_c = np.zeros(num_classes)
        for c in range(num_classes):
            mask = (y == c)
            n_c[c] = mask.sum()
            if n_c[c] > 0:
                mu_c[c] = Xs[mask].mean(axis=0)
        XtX = Xs.T @ Xs
        Sw = XtX - sum(n_c[c] * np.outer(mu_c[c], mu_c[c]) for c in range(num_classes))
        Sw_reg = Sw + best_lambda * np.eye(D)
        W = np.linalg.solve(Sw_reg, mu_c.T)
        pi = n_c / n_c.sum()
        b = -0.5 * np.sum(mu_c * W.T, axis=1) + np.log(pi + 1e-12)
        W_aug = np.vstack([W, b.reshape(1, -1)])
    
    # Convert back to original feature space
    W_std = W_aug[:-1, :].T  # (10, D)
    b_std = W_aug[-1, :]     # (10,)
    W_orig = W_std / std
    b_orig = b_std - (mean / std) @ W_std.T
    
    # Clip to safe range
    W_orig = np.clip(W_orig, -1000, 1000)
    b_orig = np.clip(b_orig, -1000, 1000)
    
    w["head.weight"] = W_orig.astype(np.float32)
    w["head.bias"] = b_orig.astype(np.float32)
    
    diagnostics = {
        "best_method": best_method,
        "best_lambda": best_lambda,
        "cv_accuracy": float(best_acc),
        "W_mean": float(W_orig.mean()),
        "W_std": float(W_orig.std()),
        "b_mean": float(b_orig.mean()),
        "b_std": float(b_orig.std())
    }
    
    return {"weights": w, "diagnostics": diagnostics}
