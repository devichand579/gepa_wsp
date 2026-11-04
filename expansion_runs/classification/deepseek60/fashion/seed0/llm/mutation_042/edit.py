import numpy as np

def edit(w, f):
    # Extract features and labels
    X = f["features"].astype(np.float64)  # (2000, 1568)
    y = f["y"].astype(np.int64)           # (2000,)
    N, D = X.shape
    C = 10
    
    # Augment with bias column
    X_aug = np.concatenate([X, np.ones((N, 1), dtype=np.float64)], axis=1)  # (2000, 1569)
    Y = np.eye(C)[y]  # (2000, 10)
    
    # Cross-validation to select lambda
    lambdas = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
    n_folds = 5
    fold_size = N // n_folds
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    best_lambda = None
    best_acc = -1.0
    cv_results = {}
    
    for lam in lambdas:
        fold_accs = []
        for i in range(n_folds):
            val_idx = indices[i*fold_size:(i+1)*fold_size]
            train_idx = np.setdiff1d(indices, val_idx)
            
            X_train = X_aug[train_idx]
            Y_train = Y[train_idx]
            X_val = X_aug[val_idx]
            y_val = y[val_idx]
            
            # Regularization matrix: do not regularize bias
            reg = np.eye(D+1) * lam
            reg[-1, -1] = 0.0
            
            # Solve ridge regression: (X^T X + reg) W = X^T Y
            A = X_train.T @ X_train + reg
            B = X_train.T @ Y_train
            W = np.linalg.solve(A, B)  # (D+1, C)
            
            # Predict
            logits = X_val @ W
            preds = np.argmax(logits, axis=1)
            acc = np.mean(preds == y_val)
            fold_accs.append(acc)
        
        mean_acc = np.mean(fold_accs)
        cv_results[lam] = mean_acc
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lambda = lam
    
    # Fit final head on all data with best lambda
    reg = np.eye(D+1) * best_lambda
    reg[-1, -1] = 0.0
    A = X_aug.T @ X_aug + reg
    B = X_aug.T @ Y
    W_final = np.linalg.solve(A, B)  # (D+1, C)
    
    # Extract weights and bias
    head_weight = W_final[:-1, :].T.astype(np.float32)  # (10, 1568)
    head_bias = W_final[-1, :].astype(np.float32)       # (10,)
    
    # Clip to safe range
    head_weight = np.clip(head_weight, -1000.0, 1000.0)
    head_bias = np.clip(head_bias, -1000.0, 1000.0)
    
    # Update weights
    w["head.weight"] = head_weight
    w["head.bias"] = head_bias
    
    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_acc),
        "cv_results": {str(k): float(v) for k, v in cv_results.items()},
        "head_weight_norm": float(np.linalg.norm(head_weight)),
        "head_bias_norm": float(np.linalg.norm(head_bias)),
        "head_weight_max_abs": float(np.max(np.abs(head_weight))),
        "head_bias_max_abs": float(np.max(np.abs(head_bias)))
    }
    
    return {"weights": w, "diagnostics": diagnostics}
