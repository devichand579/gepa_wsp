import numpy as np

def edit(w, f):
    # Extract current weights
    fc1_w = w["fc1.weight"].copy()
    fc1_b = w["fc1.bias"].copy()
    fc2_w = w["fc2.weight"].copy()
    fc2_b = w["fc2.bias"].copy()
    
    # Get data
    X_cur = f["features"].astype(np.float64)  # [2000, 128]
    y = f["y"].astype(np.int64)
    n = len(y)
    
    def one_hot(y, num_classes=10):
        Y = np.zeros((len(y), num_classes))
        Y[np.arange(len(y)), y] = 1.0
        return Y
    
    def ridge_cv(X, y, lambdas):
        Y = one_hot(y)
        indices = np.arange(n)
        np.random.seed(0)
        np.random.shuffle(indices)
        folds = np.array_split(indices, 5)
        best_acc = -1
        best_lambda = None
        for lam in lambdas:
            accs = []
            for i in range(5):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
                X_train = X[train_idx]
                Y_train = Y[train_idx]
                X_val = X[val_idx]
                y_val = y[val_idx]
                X_train_aug = np.hstack([X_train, np.ones((len(X_train), 1))])
                X_val_aug = np.hstack([X_val, np.ones((len(X_val), 1))])
                A = X_train_aug.T @ X_train_aug + lam * np.eye(X_train_aug.shape[1])
                B = X_train_aug.T @ Y_train
                W = np.linalg.solve(A, B)
                pred = X_val_aug @ W
                pred_class = np.argmax(pred, axis=1)
                acc = np.mean(pred_class == y_val)
                accs.append(acc)
            mean_acc = np.mean(accs)
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_lambda = lam
        return best_acc, best_lambda
    
    # Evaluate current fc2 via CV on current features
    acc_cur, lam_cur = ridge_cv(X_cur, y, [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    
    # LDA on current features
    def lda_fit(X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]
        means = np.zeros((n_classes, n_features))
        for i, c in enumerate(classes):
            means[i] = X[y == c].mean(axis=0)
        global_mean = X.mean(axis=0)
        Sw = np.zeros((n_features, n_features))
        for i, c in enumerate(classes):
            Xc = X[y == c] - means[i]
            Sw += Xc.T @ Xc
        # Regularize Sw slightly for numerical stability
        Sw += 1e-6 * np.eye(n_features)
        # LDA weights: W = (means - global_mean) @ inv(Sw)
        W = (means - global_mean) @ np.linalg.inv(Sw)
        # Bias: b = -W @ global_mean + log(prior) (prior uniform)
        b = -W @ global_mean
        return W, b
    
    # CV for LDA
    indices = np.arange(n)
    np.random.seed(0)
    np.random.shuffle(indices)
    folds = np.array_split(indices, 5)
    lda_accs = []
    for i in range(5):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
        X_train = X_cur[train_idx]
        y_train = y[train_idx]
        X_val = X_cur[val_idx]
        y_val = y[val_idx]
        W_lda, b_lda = lda_fit(X_train, y_train)
        pred = X_val @ W_lda.T + b_lda
        pred_class = np.argmax(pred, axis=1)
        acc = np.mean(pred_class == y_val)
        lda_accs.append(acc)
    acc_lda = np.mean(lda_accs)
    
    # Choose best
    if acc_lda > acc_cur:
        # Fit LDA on full data
        W_lda, b_lda = lda_fit(X_cur, y)
        new_fc2_w = W_lda.astype(np.float32)
        new_fc2_b = b_lda.astype(np.float32)
        used = "lda"
    else:
        # Keep current fc2
        new_fc2_w = fc2_w
        new_fc2_b = fc2_b
        used = "current"
    
    # Ensure finite and within bounds
    new_fc2_w = np.clip(new_fc2_w, -1000, 1000).astype(np.float32)
    new_fc2_b = np.clip(new_fc2_b, -1000, 1000).astype(np.float32)
    
    w["fc1.weight"] = fc1_w
    w["fc1.bias"] = fc1_b
    w["fc2.weight"] = new_fc2_w
    w["fc2.bias"] = new_fc2_b
    
    diagnostics = {
        "acc_cur_cv": float(acc_cur),
        "acc_lda_cv": float(acc_lda),
        "used": used,
        "fc2_weight_norm": float(np.linalg.norm(new_fc2_w)),
        "fc2_bias_norm": float(np.linalg.norm(new_fc2_b)),
    }
    return {"weights": w, "diagnostics": diagnostics}
