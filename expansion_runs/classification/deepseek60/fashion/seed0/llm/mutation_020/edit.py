import numpy as np

def edit(w, f):
    # Extract features and labels from the full feedback pool
    X = f["features"]  # (2000, 1568)
    y = f["y"]         # (2000,)
    n_samples, n_features = X.shape
    n_classes = 10
    
    # One-hot encode labels
    Y = np.zeros((n_samples, n_classes), dtype=np.float32)
    Y[np.arange(n_samples), y] = 1.0
    
    # Candidate regularization strengths
    lambdas = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]
    
    # 5-fold cross-validation
    n_folds = 5
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    best_lambda = None
    best_cv_acc = -1.0
    
    for lam in lambdas:
        accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            
            X_train_raw = X[train_idx]
            Y_train = Y[train_idx]
            X_val_raw = X[val_idx]
            y_val = y[val_idx]
            
            # Center using training mean only
            X_mean_train = X_train_raw.mean(axis=0)
            X_train = X_train_raw - X_mean_train
            X_val = X_val_raw - X_mean_train
            
            # SVD of training features
            U, S, Vt = np.linalg.svd(X_train, full_matrices=False)
            S_inv = S / (S**2 + lam)
            W = Vt.T @ (S_inv[:, None] * (U.T @ Y_train))  # (1568, 10)
            W = W.T  # (10, 1568)
            
            b_prime = Y_train.mean(axis=0)  # (10,)
            b = b_prime - X_mean_train @ W.T  # (10,)
            
            # Predict on validation fold
            logits = X_val @ W.T + b
            preds = np.argmax(logits, axis=1)
            acc = np.mean(preds == y_val)
            accs.append(acc)
        
        mean_acc = np.mean(accs)
        if mean_acc > best_cv_acc:
            best_cv_acc = mean_acc
            best_lambda = lam
    
    # Fit final model on all data using best_lambda
    X_mean = X.mean(axis=0)
    Xc = X - X_mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    S_inv = S / (S**2 + best_lambda)
    W = Vt.T @ (S_inv[:, None] * (U.T @ Y))  # (1568, 10)
    W = W.T  # (10, 1568)
    
    b_prime = Y.mean(axis=0)
    b = b_prime - X_mean @ W.T  # (10,)
    
    # Scale W and b to sharpen logits without changing argmax
    # We want to maximize scale k such that max(abs(W*k)) <= 1000 and max(abs(b*k)) <= 1000
    max_W = np.max(np.abs(W))
    max_b = np.max(np.abs(b))
    k = 1000.0 / max(max_W, max_b, 1e-8)
    
    # Apply scale
    W_scaled = W * k
    b_scaled = b * k
    
    # Update weights
    w["head.weight"] = W_scaled.astype(np.float32)
    w["head.bias"] = b_scaled.astype(np.float32)
    
    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lambda),
        "best_cv_accuracy": float(best_cv_acc),
        "scale_k": float(k),
        "max_abs_head_weight": float(np.max(np.abs(W_scaled))),
        "max_abs_head_bias": float(np.max(np.abs(b_scaled))),
        "head_weight_mean": float(np.mean(W_scaled)),
        "head_weight_std": float(np.std(W_scaled)),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
