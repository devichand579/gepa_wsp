import numpy as np

def edit(w, f):
    # Extract features and labels
    X = f["features"].astype(np.float64)
    y = f["y"].astype(np.int64)
    N, D = X.shape
    
    # Augment features with bias column
    X_aug = np.hstack([X, np.ones((N, 1))])
    Y = np.eye(10)[y]
    
    # Precompute full data statistics
    XtX = X_aug.T @ X_aug
    XtY = X_aug.T @ Y
    
    # 5-fold cross-validation to select lambda
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, 5)
    
    lambdas = [0.1, 1.0, 10.0, 100.0, 1000.0, 3162.2776601683795, 10000.0, 100000.0]
    best_lambda = None
    best_acc = -1.0
    
    for lam in lambdas:
        accs = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            
            X_train = X_aug[train_idx]
            Y_train = Y[train_idx]
            X_val = X_aug[val_idx]
            y_val = y[val_idx]
            
            XtX_train = X_train.T @ X_train
            XtY_train = X_train.T @ Y_train
            
            reg = np.eye(D + 1) * lam
            reg[-1, -1] = 0.0  # do not regularize bias
            
            A = XtX_train + reg
            W_aug = np.linalg.solve(A, XtY_train).T  # (10, D+1)
            
            logits = X_val @ W_aug.T
            pred = np.argmax(logits, axis=1)
            acc = np.mean(pred == y_val)
            accs.append(acc)
        
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lambda = lam
    
    # Fit on full data with best lambda
    reg = np.eye(D + 1) * best_lambda
    reg[-1, -1] = 0.0
    A = XtX + reg
    W_aug = np.linalg.solve(A, XtY).T  # (10, D+1)
    
    W = W_aug[:, :D]
    b = W_aug[:, D]
    
    # Clip to ensure bounds
    W = np.clip(W, -1000.0, 1000.0)
    b = np.clip(b, -1000.0, 1000.0)
    
    # Update weights
    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)
    
    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_acc),
        "W_mean": float(W.mean()),
        "W_std": float(W.std()),
        "b_mean": float(b.mean()),
        "b_std": float(b.std()),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
