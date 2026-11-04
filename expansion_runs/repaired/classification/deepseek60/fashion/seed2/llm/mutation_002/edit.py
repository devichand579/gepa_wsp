import numpy as np
import math

def edit(w, f):
    # Extract features and labels
    X = f["features"].astype(np.float64)
    y = f["y"].astype(np.int64)
    N, D = X.shape
    C = 10
    
    # Current head weights (PyTorch layout: [out, in])
    W0 = w["head.weight"].astype(np.float64).T  # D x C
    b0 = w["head.bias"].astype(np.float64)      # C
    
    # One-hot targets
    Y = np.eye(C)[y]
    
    # Stratified 5-fold CV
    np.random.seed(42)
    indices = np.arange(N)
    n_folds = 5
    fold_indices = [[] for _ in range(n_folds)]
    for c in range(C):
        c_idx = indices[y == c]
        np.random.shuffle(c_idx)
        for i, idx in enumerate(c_idx):
            fold_indices[i % n_folds].append(idx)
    fold_indices = [np.array(f) for f in fold_indices]
    
    # Candidate lambdas for ridge with prior
    lambdas = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
    
    best_val_acc = -1.0
    best_lambda = None
    best_W = None
    best_b = None
    
    for lam in lambdas:
        val_accs = []
        for k in range(n_folds):
            val_idx = fold_indices[k]
            train_idx = np.concatenate([fold_indices[j] for j in range(n_folds) if j != k])
            
            X_train = X[train_idx]
            Y_train = Y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]
            
            # Augment with bias
            X_train_aug = np.hstack([X_train, np.ones((len(train_idx), 1))])
            W0_aug = np.vstack([W0, b0])
            
            # Ridge with prior
            A = X_train_aug.T @ X_train_aug + lam * np.eye(D + 1)
            B = X_train_aug.T @ Y_train + lam * W0_aug
            W_aug = np.linalg.solve(A, B)
            W = W_aug[:-1, :]
            b = W_aug[-1, :]
            
            # Evaluate
            logits = X_val @ W + b
            preds = np.argmax(logits, axis=1)
            acc = np.mean(preds == y_val)
            val_accs.append(acc)
        
        mean_val_acc = np.mean(val_accs)
        if mean_val_acc > best_val_acc:
            best_val_acc = mean_val_acc
            best_lambda = lam
            best_W = W
            best_b = b
    
    # Retrain on full pool with best lambda
    X_aug = np.hstack([X, np.ones((N, 1))])
    W0_aug = np.vstack([W0, b0])
    A = X_aug.T @ X_aug + best_lambda * np.eye(D + 1)
    B = X_aug.T @ Y + best_lambda * W0_aug
    W_aug = np.linalg.solve(A, B)
    W_final = W_aug[:-1, :]
    b_final = W_aug[-1, :]
    
    # Compute final training accuracy
    logits = X @ W_final + b_final
    preds = np.argmax(logits, axis=1)
    train_acc = np.mean(preds == y)
    
    # Update weights (transpose back to [out, in])
    w["head.weight"] = W_final.T.astype(np.float32)
    w["head.bias"] = b_final.astype(np.float32)
    
    # Clip to safe range
    w["head.weight"] = np.clip(w["head.weight"], -1000.0, 1000.0)
    w["head.bias"] = np.clip(w["head.bias"], -1000.0, 1000.0)
    
    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_val_acc": float(best_val_acc),
        "train_acc_full": float(train_acc),
        "head_weight_std": float(np.std(w["head.weight"])),
        "head_bias_mean": float(np.mean(w["head.bias"])),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
