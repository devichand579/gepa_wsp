import numpy as np

def edit(w, f):
    # Extract data
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float32)  # (2000, 784)
    y = f["y"].astype(np.int64)
    N = X.shape[0]
    
    # Current weights
    W1_cur = w["fc1.weight"].astype(np.float32)  # (128, 784)
    b1_cur = w["fc1.bias"].astype(np.float32)    # (128,)
    W2_cur = w["fc2.weight"].astype(np.float32)  # (10, 128)
    b2_cur = w["fc2.bias"].astype(np.float32)    # (10,)
    
    # Current features
    H_cur = f["features"].astype(np.float32)     # (2000, 128)
    
    # Feature importance from current fc2 weights
    imp = np.sum(np.abs(W2_cur), axis=0)         # (128,)
    sorted_idx = np.argsort(imp)[::-1]           # descending
    
    # Validation split using the provided batch
    batch_ids = f["batch.ids"].astype(np.int64)
    val_mask = np.zeros(N, dtype=bool)
    val_mask[batch_ids] = True
    train_mask = ~val_mask
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    H_cur_train = H_cur[train_mask]
    H_cur_val = H_cur[val_mask]
    
    # PCA on training set
    X_mean = X_train.mean(axis=0)
    X_train_centered = X_train - X_mean
    X_val_centered = X_val - X_mean
    # SVD for PCA (use full_matrices=False for efficiency)
    U, S, Vt = np.linalg.svd(X_train_centered, full_matrices=False)
    # Vt shape (min(N_train, 784), 784). We need up to 128 components.
    
    best_acc = -1.0
    best_K = 0
    best_lambda = 1.0
    best_W2 = None
    best_b2 = None
    
    # Grid search over K (number of kept current features) and lambda
    K_options = [0, 32, 64, 96, 128]
    lambda_options = [0.01, 0.1, 1.0, 10.0]
    
    for K in K_options:
        kept = sorted_idx[:K]
        new_count = 128 - K
        if new_count > Vt.shape[0]:
            new_count = Vt.shape[0]
            K = 128 - new_count
            kept = sorted_idx[:K]
        
        # Build training features
        H_train = np.zeros((X_train.shape[0], 128), dtype=np.float32)
        if K > 0:
            H_train[:, :K] = H_cur_train[:, kept]
        if new_count > 0:
            proj_train = X_train_centered @ Vt[:new_count].T
            H_train[:, K:] = np.maximum(proj_train, 0.0)
        
        # Build validation features
        H_val = np.zeros((X_val.shape[0], 128), dtype=np.float32)
        if K > 0:
            H_val[:, :K] = H_cur_val[:, kept]
        if new_count > 0:
            proj_val = X_val_centered @ Vt[:new_count].T
            H_val[:, K:] = np.maximum(proj_val, 0.0)
        
        # One-hot encode labels for training
        Y_train = np.zeros((X_train.shape[0], 10), dtype=np.float32)
        Y_train[np.arange(X_train.shape[0]), y_train] = 1.0
        
        # Augment with bias column
        H_train_aug = np.hstack([H_train, np.ones((H_train.shape[0], 1), dtype=np.float32)])
        H_val_aug = np.hstack([H_val, np.ones((H_val.shape[0], 1), dtype=np.float32)])
        
        for lam in lambda_options:
            # Ridge regression: (H^T H + lam I) W = H^T Y
            A = H_train_aug.T @ H_train_aug + lam * np.eye(H_train_aug.shape[1], dtype=np.float32)
            B = H_train_aug.T @ Y_train
            try:
                W = np.linalg.solve(A, B)
            except np.linalg.LinAlgError:
                W = np.linalg.lstsq(A, B, rcond=None)[0]
            
            # Predict on validation
            logits_val = H_val_aug @ W
            pred_val = np.argmax(logits_val, axis=1)
            acc = np.mean(pred_val == y_val)
            
            if acc > best_acc:
                best_acc = acc
                best_K = K
                best_lambda = lam
                best_W2 = W[:128].T.copy()  # (10, 128)
                best_b2 = W[128].copy()     # (10,)
    
    # If no improvement found, fallback to K=64, lambda=1.0
    if best_W2 is None:
        best_K = 64
        best_lambda = 1.0
    
    # Now retrain on full data using best_K and best_lambda
    # Recompute PCA on full data
    X_mean_full = X.mean(axis=0)
    X_centered_full = X - X_mean_full
    U_full, S_full, Vt_full = np.linalg.svd(X_centered_full, full_matrices=False)
    
    kept = sorted_idx[:best_K]
    new_count = 128 - best_K
    if new_count > Vt_full.shape[0]:
        new_count = Vt_full.shape[0]
        best_K = 128 - new_count
        kept = sorted_idx[:best_K]
    
    # Build new fc1 weights and biases
    new_W1 = np.zeros((128, 784), dtype=np.float32)
    new_b1 = np.zeros(128, dtype=np.float32)
    
    if best_K > 0:
        new_W1[:best_K] = W1_cur[kept]
        new_b1[:best_K] = b1_cur[kept]
    if new_count > 0:
        new_W1[best_K:] = Vt_full[:new_count]
        new_b1[best_K:] = 0.0
    
    # Compute new features for all data
    H_all = np.zeros((N, 128), dtype=np.float32)
    if best_K > 0:
        H_all[:, :best_K] = H_cur[:, kept]
    if new_count > 0:
        proj_all = X_centered_full @ Vt_full[:new_count].T
        H_all[:, best_K:] = np.maximum(proj_all, 0.0)
    
    # Fit fc2 on all data
    Y_all = np.zeros((N, 10), dtype=np.float32)
    Y_all[np.arange(N), y] = 1.0
    H_all_aug = np.hstack([H_all, np.ones((N, 1), dtype=np.float32)])
    A = H_all_aug.T @ H_all_aug + best_lambda * np.eye(H_all_aug.shape[1], dtype=np.float32)
    B = H_all_aug.T @ Y_all
    try:
        W_all = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        W_all = np.linalg.lstsq(A, B, rcond=None)[0]
    
    new_W2 = W_all[:128].T.astype(np.float32)
    new_b2 = W_all[128].astype(np.float32)
    
    # Update weights
    w["fc1.weight"] = new_W1
    w["fc1.bias"] = new_b1
    w["fc2.weight"] = new_W2
    w["fc2.bias"] = new_b2
    
    # Diagnostics
    diagnostics = {
        "best_K": int(best_K),
        "best_lambda": float(best_lambda),
        "val_accuracy": float(best_acc),
        "new_fc1_weight_mean": float(new_W1.mean()),
        "new_fc1_weight_std": float(new_W1.std()),
        "new_fc2_weight_mean": float(new_W2.mean()),
        "new_fc2_weight_std": float(new_W2.std()),
        "new_fc2_bias_mean": float(new_b2.mean()),
        "new_fc2_bias_std": float(new_b2.std()),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
