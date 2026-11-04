import numpy as np

def edit(w, f):
    # Extract features and labels
    F = f["features"].astype(np.float64)  # (2000, 1568)
    y = f["y"].astype(np.int64)           # (2000,)
    N, D = F.shape
    num_classes = 10
    
    # Current head parameters
    W0 = w["head.weight"].astype(np.float64)  # (10, 1568)
    b0 = w["head.bias"].astype(np.float64)    # (10,)
    
    # Augment features with a bias column
    X = np.hstack([F, np.ones((N, 1))])  # (2000, 1569)
    Y = np.zeros((N, num_classes))
    Y[np.arange(N), y] = 1.0
    
    # Augmented prior (current head + bias)
    W0_aug = np.hstack([W0, b0[:, None]])  # (10, 1569)
    
    # Candidate regularization strengths
    lambdas = [1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    
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
            y_val = y[val_idx]
            
            XtX = X_train.T @ X_train
            XtY = X_train.T @ Y_train
            
            # Solve (XtX + lam I) W_aug^T = XtY + lam W0_aug^T
            d, V = np.linalg.eigh(XtX)
            A = V.T @ XtY
            B = V.T @ W0_aug.T
            W_aug_T = V @ ((A + lam * B) / (d[:, None] + lam))
            W_aug = W_aug_T.T  # (10, 1569)
            
            logits = X_val @ W_aug.T
            preds = np.argmax(logits, axis=1)
            accs.append(np.mean(preds == y_val))
        
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lambda = lam
    
    # Fit on full data with best lambda
    XtX = X.T @ X
    XtY = X.T @ Y
    d, V = np.linalg.eigh(XtX)
    A = V.T @ XtY
    B = V.T @ W0_aug.T
    W_aug_T = V @ ((A + best_lambda * B) / (d[:, None] + best_lambda))
    W_aug = W_aug_T.T  # (10, 1569)
    
    # Split back into weight and bias
    W_new = W_aug[:, :D]
    b_new = W_aug[:, D]
    
    # Rescale to maximum allowed magnitude (1000) to reduce cross-entropy
    max_abs = max(np.max(np.abs(W_new)), np.max(np.abs(b_new)))
    if max_abs > 0:
        scale = 1000.0 / max_abs
        W_new *= scale
        b_new *= scale
    
    # Update weights
    w["head.weight"] = W_new.astype(np.float32)
    w["head.bias"] = b_new.astype(np.float32)
    
    # Diagnostics
    diagnostics = {
        "best_lambda": best_lambda,
        "cv_accuracy": float(best_acc),
        "max_abs_weight": float(np.max(np.abs(W_new))),
        "max_abs_bias": float(np.max(np.abs(b_new))),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
