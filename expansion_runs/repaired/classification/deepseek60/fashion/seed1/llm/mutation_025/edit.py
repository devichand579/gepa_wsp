import numpy as np

def edit(w, f):
    # Extract features and labels from the full feedback pool
    X = f["features"].astype(np.float64)  # (2000, 1568)
    y = f["y"].astype(int)                # (2000,)
    N, D = X.shape
    C = 10
    
    # One-hot encode labels
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0
    
    # Candidate regularization strengths
    lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
    
    # 5-fold cross-validation to select lambda
    n_folds = 5
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    best_lambda = None
    best_acc = -1.0
    
    for lam in lambdas:
        accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            
            X_train = X[train_idx]
            Y_train = Y[train_idx]
            X_val = X[val_idx]
            Y_val = Y[val_idx]
            
            # Center data
            X_mean = X_train.mean(axis=0)
            Y_mean = Y_train.mean(axis=0)
            X_c = X_train - X_mean
            Y_c = Y_train - Y_mean
            
            # Compute covariance and its eigendecomposition
            C_mat = X_c.T @ X_c  # (D, D)
            s, V = np.linalg.eigh(C_mat)
            s = np.maximum(s, 0.0)  # clip numerical negatives
            
            XTY = X_c.T @ Y_c  # (D, C)
            Vt_XTY = V.T @ XTY  # (D, C)
            
            # Ridge solution: W = V diag(1/(s+lam)) V^T XTY
            W = V @ (Vt_XTY / (s + lam)[:, None])  # (D, C)
            b = Y_mean - X_mean @ W                # (C,)
            
            # Predict validation set
            logits_val = X_val @ W + b
            preds = np.argmax(logits_val, axis=1)
            true = np.argmax(Y_val, axis=1)
            acc = np.mean(preds == true)
            accs.append(acc)
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lambda = lam
    
    # Fit on the full dataset using the best lambda
    lam = best_lambda
    X_mean = X.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    X_c = X - X_mean
    Y_c = Y - Y_mean
    
    C_mat = X_c.T @ X_c
    s, V = np.linalg.eigh(C_mat)
    s = np.maximum(s, 0.0)
    XTY = X_c.T @ Y_c
    Vt_XTY = V.T @ XTY
    W = V @ (Vt_XTY / (s + lam)[:, None])  # (D, C)
    b = Y_mean - X_mean @ W                # (C,)
    
    # Transpose to match head.weight shape (C, D)
    W_head = W.T  # (10, 1568)
    b_head = b    # (10,)
    
    # Ensure maximum absolute value does not exceed 1000
    max_abs = max(np.max(np.abs(W_head)), np.max(np.abs(b_head)))
    if max_abs > 1000:
        scale = 1000.0 / max_abs
        W_head *= scale
        b_head *= scale
    
    # Update weights in-place
    w["head.weight"] = W_head.astype(np.float32)
    w["head.bias"] = b_head.astype(np.float32)
    
    # Diagnostics
    diagnostics = {
        "best_lambda": float(lam),
        "cv_accuracy": float(best_acc),
        "max_abs_weight": float(max_abs),
        "scale_applied": float(1000.0 / max_abs) if max_abs > 1000 else 1.0,
    }
    
    return {"weights": w, "diagnostics": diagnostics}
