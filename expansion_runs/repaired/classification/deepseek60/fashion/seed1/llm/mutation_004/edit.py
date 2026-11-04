def edit(w, f):
    import numpy as np
    
    # Load full feedback features and labels
    F = f["features"].astype(np.float64)  # (2000, 1568)
    y = f["y"].astype(np.int64)           # (2000,)
    N, D = F.shape
    C = 10
    
    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0
    
    # Ridge penalty grid for cross-validation
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    best_lambda = 1e-2
    best_acc = -1.0
    
    # 5-fold cross-validation to select lambda
    indices = np.random.RandomState(42).permutation(N)
    folds = np.array_split(indices, 5)
    
    for lam in lambdas:
        accs = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            
            F_train = F[train_idx]
            F_val = F[val_idx]
            Y_train = Y[train_idx]
            
            # Standardize using training statistics
            mean_train = F_train.mean(axis=0)
            std_train = F_train.std(axis=0)
            std_train[std_train < 1e-6] = 1e-6
            
            Z_train = (F_train - mean_train) / std_train
            Z_val = (F_val - mean_train) / std_train
            
            # Center features and targets
            Zc_train = Z_train - Z_train.mean(axis=0)
            Yc_train = Y_train - Y_train.mean(axis=0)
            
            # Closed-form ridge solution
            A = Zc_train.T @ Zc_train + lam * np.eye(D)
            B = Zc_train.T @ Yc_train
            W = np.linalg.solve(A, B)
            
            # Predict validation
            Zc_val = Z_val - Z_train.mean(axis=0)
            logits = Zc_val @ W + Y_train.mean(axis=0)
            pred = np.argmax(logits, axis=1)
            acc = np.mean(pred == y[val_idx])
            accs.append(acc)
        
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lambda = lam
    
    # Final training on all data with best lambda
    mean_all = F.mean(axis=0)
    std_all = F.std(axis=0)
    std_all[std_all < 1e-6] = 1e-6
    
    Z_all = (F - mean_all) / std_all
    Zc_all = Z_all - Z_all.mean(axis=0)
    Yc_all = Y - Y.mean(axis=0)
    
    A = Zc_all.T @ Zc_all + best_lambda * np.eye(D)
    B = Zc_all.T @ Yc_all
    W = np.linalg.solve(A, B)  # (D, C)
    
    # Map back to original feature space
    W_orig = W / std_all[:, None]          # (D, C)
    bias_orig = Y.mean(axis=0) - mean_all @ W_orig  # (C,)
    
    # Update head weights
    w["head.weight"] = W_orig.T.astype(np.float32)  # (C, D)
    w["head.bias"] = bias_orig.astype(np.float32)   # (C,)
    
    # Diagnostics
    diag = {
        "best_lambda": float(best_lambda),
        "cv_acc": float(best_acc),
        "max_abs_weight": float(np.max(np.abs(w["head.weight"]))),
        "max_abs_bias": float(np.max(np.abs(w["head.bias"]))),
        "mean_weight": float(np.mean(w["head.weight"])),
        "std_weight": float(np.std(w["head.weight"])),
    }
    return {"weights": w, "diagnostics": diag}
