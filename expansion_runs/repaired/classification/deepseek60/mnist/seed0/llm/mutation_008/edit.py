import numpy as np

def edit(w, f):
    # Extract current weights
    fc1_w = w["fc1.weight"].astype(np.float32)  # (128, 784)
    fc1_b = w["fc1.bias"].astype(np.float32)    # (128,)
    fc2_w = w["fc2.weight"].astype(np.float32)  # (10, 128)
    fc2_b = w["fc2.bias"].astype(np.float32)    # (10,)

    # Feedback data
    X = f["x"].astype(np.float32)               # (2000, 1, 28, 28)
    y = f["y"].astype(np.int64)                 # (2000,)
    H_orig = f["features"].astype(np.float32)   # (2000, 128) - original features
    N = X.shape[0]

    # One-hot targets
    Y_orig = np.zeros((N, 10), dtype=np.float32)
    Y_orig[np.arange(N), y] = 1.0

    # Build augmented dataset with shifts dy, dx in [-2, 2]
    shifts = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)]
    # Pad images to 32x32
    X_pad = np.pad(X, ((0, 0), (0, 0), (2, 2), (2, 2)), mode='constant')
    
    H_list = []
    y_list = []
    orig_idx_list = []
    
    for dy, dx in shifts:
        # Crop shifted 28x28
        X_shift = X_pad[:, :, 2+dy:2+dy+28, 2+dx:2+dx+28]
        X_flat = X_shift.reshape(N, -1)  # (N, 784)
        # Compute features using current fc1
        H = X_flat @ fc1_w.T + fc1_b  # (N, 128)
        H = np.maximum(H, 0.0)        # ReLU
        H_list.append(H)
        y_list.append(y)
        orig_idx_list.append(np.arange(N))
    
    H_all = np.concatenate(H_list, axis=0)       # (50000, 128)
    y_all = np.concatenate(y_list, axis=0)       # (50000,)
    orig_idx = np.concatenate(orig_idx_list, axis=0)  # (50000,)
    
    # One-hot for augmented
    Y_all = np.zeros((H_all.shape[0], 10), dtype=np.float32)
    Y_all[np.arange(H_all.shape[0]), y_all] = 1.0

    # Precompute per-original-image statistics for efficient CV
    G_list = []
    C_list = []
    sum_H_list = []
    sum_Y_list = []
    n_list = []
    for i in range(N):
        mask = (orig_idx == i)
        H_i = H_all[mask]          # (n_i, 128)
        Y_i = Y_all[mask]          # (n_i, 10)
        G_i = H_i.T @ H_i          # (128, 128)
        C_i = H_i.T @ Y_i          # (128, 10)
        sum_H_i = H_i.sum(axis=0)  # (128,)
        sum_Y_i = Y_i.sum(axis=0)  # (10,)
        n_i = H_i.shape[0]
        G_list.append(G_i)
        C_list.append(C_i)
        sum_H_list.append(sum_H_i)
        sum_Y_list.append(sum_Y_i)
        n_list.append(n_i)
    
    G_all = np.stack(G_list)       # (N, 128, 128)
    C_all = np.stack(C_list)       # (N, 128, 10)
    sum_H_all = np.stack(sum_H_list)  # (N, 128)
    sum_Y_all = np.stack(sum_Y_list)  # (N, 10)
    n_all = np.array(n_list, dtype=np.float32)  # (N,)

    # Candidate regularization parameters
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    lambda_bs = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    
    best_acc = -1.0
    best_params = (None, None)
    
    # 5-fold CV
    fold_size = N // 5
    indices = np.arange(N)
    # Shuffle deterministically for reproducibility
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    
    for lam in lambdas:
        for lam_b in lambda_bs:
            acc_sum = 0.0
            for fold in range(5):
                val_mask = np.zeros(N, dtype=bool)
                val_mask[indices[fold*fold_size:(fold+1)*fold_size]] = True
                train_mask = ~val_mask
                
                # Aggregate training statistics
                G = G_all[train_mask].sum(axis=0)          # (128, 128)
                C = C_all[train_mask].sum(axis=0)          # (128, 10)
                sum_H = sum_H_all[train_mask].sum(axis=0)  # (128,)
                sum_Y = sum_Y_all[train_mask].sum(axis=0)  # (10,)
                n = n_all[train_mask].sum()                # scalar
                
                # Build normal equations with prior regularization
                M = np.zeros((129, 129), dtype=np.float64)
                R = np.zeros((129, 10), dtype=np.float64)
                
                # Feature block
                M[:128, :128] = G + lam * np.eye(128)
                M[:128, 128] = sum_H
                M[128, :128] = sum_H
                M[128, 128] = n + lam_b
                
                # Right side
                R[:128, :] = C + lam * fc2_w.T
                R[128, :] = sum_Y + lam_b * fc2_b
                
                # Solve
                try:
                    X_sol = np.linalg.solve(M, R)  # (129, 10)
                except np.linalg.LinAlgError:
                    continue
                
                W_new = X_sol[:128, :].T  # (10, 128)
                b_new = X_sol[128, :]     # (10,)
                
                # Validate on original features of validation fold
                val_idx = indices[fold*fold_size:(fold+1)*fold_size]
                H_val = H_orig[val_idx]   # (n_val, 128)
                y_val = y[val_idx]
                logits = H_val @ W_new.T + b_new
                preds = np.argmax(logits, axis=1)
                acc = np.mean(preds == y_val)
                acc_sum += acc
            
            avg_acc = acc_sum / 5.0
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_params = (lam, lam_b)
    
    # If no valid solution found, keep original
    if best_params[0] is None:
        return w
    
    lam, lam_b = best_params
    
    # Train on all augmented data with best parameters
    G = G_all.sum(axis=0)
    C = C_all.sum(axis=0)
    sum_H = sum_H_all.sum(axis=0)
    sum_Y = sum_Y_all.sum(axis=0)
    n = n_all.sum()
    
    M = np.zeros((129, 129), dtype=np.float64)
    R = np.zeros((129, 10), dtype=np.float64)
    
    M[:128, :128] = G + lam * np.eye(128)
    M[:128, 128] = sum_H
    M[128, :128] = sum_H
    M[128, 128] = n + lam_b
    
    R[:128, :] = C + lam * fc2_w.T
    R[128, :] = sum_Y + lam_b * fc2_b
    
    X_sol = np.linalg.solve(M, R)
    W_new = X_sol[:128, :].T
    b_new = X_sol[128, :]
    
    # Update weights
    w["fc2.weight"] = W_new.astype(np.float32)
    w["fc2.bias"] = b_new.astype(np.float32)
    
    # Diagnostics
    diagnostics = {
        "best_lambda": float(lam),
        "best_lambda_b": float(lam_b),
        "cv_accuracy": float(best_acc),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
        "num_augmented": int(H_all.shape[0])
    }
    
    return {"weights": w, "diagnostics": diagnostics}
