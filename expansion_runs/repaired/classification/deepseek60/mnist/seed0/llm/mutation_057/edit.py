import numpy as np

def edit(w, f):
    # Extract training data
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)  # (2000, 784)
    y = f["y"].astype(np.int64)
    N = X.shape[0]
    classes = np.arange(10)
    
    # Compute class means and global mean
    mean_all = X.mean(axis=0)
    mean_c = np.zeros((10, X.shape[1]))
    for c in classes:
        mean_c[c] = X[y == c].mean(axis=0)
    
    # Compute within-class scatter matrix
    Sw = np.zeros((X.shape[1], X.shape[1]))
    for c in classes:
        Xc = X[y == c] - mean_c[c]
        Sw += Xc.T @ Xc
    
    # Regularize Sw to ensure invertibility
    reg_sw = 1e-2
    Sw_reg = Sw + reg_sw * np.eye(Sw.shape[0])
    
    # Compute Fisher linear discriminant directions (one per class)
    W_lda = np.zeros((10, X.shape[1]))
    b_lda = np.zeros(10)
    for c in classes:
        w_c = np.linalg.solve(Sw_reg, mean_c[c] - mean_all)
        w_c = w_c / (np.linalg.norm(w_c) + 1e-12)
        W_lda[c] = w_c
        b_lda[c] = -w_c @ mean_all
    
    # Update the first 10 (dead) units of fc1
    W1 = w["fc1.weight"].copy()
    b1 = w["fc1.bias"].copy()
    W1[:10] = W_lda.astype(np.float32)
    b1[:10] = b_lda.astype(np.float32)
    
    # Recompute hidden features for all training examples
    H = X @ W1.T + b1  # (2000, 128)
    H = np.maximum(H, 0)  # ReLU
    
    # Standardize features
    mean_h = H.mean(axis=0)
    std_h = H.std(axis=0)
    std_h = np.maximum(std_h, 1e-3)
    H_std = (H - mean_h) / std_h
    
    # One-hot labels
    Y = np.zeros((N, 10))
    Y[np.arange(N), y] = 1.0
    
    # 5-fold cross-validation to select ridge lambda
    lambdas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    best_lambda = 1.0
    best_acc = -1.0
    n_folds = 5
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    for lam in lambdas:
        accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            H_train = H_std[train_idx]
            Y_train = Y[train_idx]
            H_val = H_std[val_idx]
            
            # Augment with bias column
            H_train_aug = np.hstack([H_train, np.ones((len(train_idx), 1))])
            H_val_aug = np.hstack([H_val, np.ones((len(val_idx), 1))])
            
            # Ridge regression with no penalty on bias
            reg_matrix = np.eye(H_train_aug.shape[1]) * lam
            reg_matrix[-1, -1] = 0.0
            
            A = H_train_aug.T @ H_train_aug + reg_matrix
            B = H_train_aug.T @ Y_train
            W_aug = np.linalg.solve(A, B)
            
            logits_val = H_val_aug @ W_aug
            preds = np.argmax(logits_val, axis=1)
            acc = np.mean(preds == y[val_idx])
            accs.append(acc)
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lambda = lam
    
    # Final fit on all training data with best lambda
    H_aug = np.hstack([H_std, np.ones((N, 1))])
    reg_matrix = np.eye(H_aug.shape[1]) * best_lambda
    reg_matrix[-1, -1] = 0.0
    A = H_aug.T @ H_aug + reg_matrix
    B = H_aug.T @ Y
    W_aug = np.linalg.solve(A, B)
    
    # Convert back to original feature space
    W2_std = W_aug[:128].T  # (10, 128)
    b2_std = W_aug[128]     # (10,)
    W2 = W2_std / std_h
    b2 = b2_std - W2 @ mean_h
    
    # Clip to safe range
    W2 = np.clip(W2, -1000.0, 1000.0)
    b2 = np.clip(b2, -1000.0, 1000.0)
    
    # Write back to w
    w["fc1.weight"] = W1.astype(np.float32)
    w["fc1.bias"] = b1.astype(np.float32)
    w["fc2.weight"] = W2.astype(np.float32)
    w["fc2.bias"] = b2.astype(np.float32)
    
    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_acc),
        "new_fc1_weight_norm": float(np.linalg.norm(W_lda)),
        "new_fc1_bias_mean": float(b_lda.mean()),
        "fc2_weight_norm": float(np.linalg.norm(W2)),
        "fc2_bias_norm": float(np.linalg.norm(b2)),
    }
    return {"weights": w, "diagnostics": diagnostics}
