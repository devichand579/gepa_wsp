import numpy as np

def edit(w, f):
    # Extract training data and current hidden features
    X = f["x"].reshape(2000, -1).astype(np.float32)          # [2000, 784]
    y = f["y"].astype(np.int64)                               # [2000]
    H = f["features"].astype(np.float32)                      # [2000, 128] current ReLU features
    
    # Data augmentation: shifted versions of the images, then recompute features using current fc1
    W1 = w["fc1.weight"].astype(np.float32)                   # [128, 784]
    b1 = w["fc1.bias"].astype(np.float32)                     # [128]
    
    X_img = X.reshape(-1, 28, 28)
    aug_H = [H]
    aug_y = [y]
    
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            X_shift = np.roll(X_img, shift=(dx, dy), axis=(1, 2))
            if dx > 0:
                X_shift[:, :dx, :] = 0
            elif dx < 0:
                X_shift[:, dx:, :] = 0
            if dy > 0:
                X_shift[:, :, :dy] = 0
            elif dy < 0:
                X_shift[:, :, dy:] = 0
            X_shift_flat = X_shift.reshape(-1, 784)
            H_shift = X_shift_flat @ W1.T + b1
            H_shift = np.maximum(H_shift, 0)
            aug_H.append(H_shift)
            aug_y.append(y)
    
    H_aug = np.vstack(aug_H)                                  # [N_aug, 128]
    y_aug = np.concatenate(aug_y)                             # [N_aug]
    N_aug = H_aug.shape[0]
    
    # One-hot labels
    Y_aug = np.zeros((N_aug, 10), dtype=np.float32)
    Y_aug[np.arange(N_aug), y_aug] = 1.0
    
    # Add bias column to features
    H_aug_b = np.hstack([H_aug, np.ones((N_aug, 1), dtype=np.float32)])  # [N_aug, 129]
    
    # 5-fold cross-validation to choose ridge lambda
    rng = np.random.RandomState(42)
    indices = rng.permutation(N_aug)
    folds = np.array_split(indices, 5)
    
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    best_lambda = 1e-3
    best_acc = -1.0
    
    for lam in lambdas:
        accs = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            
            H_train = H_aug_b[train_idx]
            Y_train = Y_aug[train_idx]
            H_val = H_aug_b[val_idx]
            y_val = y_aug[val_idx]
            
            # Ridge regression: W = (H^T H + lam I)^-1 H^T Y
            A = H_train.T @ H_train + lam * np.eye(H_train.shape[1], dtype=np.float32)
            B = H_train.T @ Y_train
            W = np.linalg.solve(A, B)  # [129, 10]
            
            logits_val = H_val @ W
            preds = np.argmax(logits_val, axis=1)
            acc = np.mean(preds == y_val)
            accs.append(acc)
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lambda = lam
    
    # Fit final fc2 on all augmented data with best lambda
    A = H_aug_b.T @ H_aug_b + best_lambda * np.eye(H_aug_b.shape[1], dtype=np.float32)
    B = H_aug_b.T @ Y_aug
    W_final = np.linalg.solve(A, B)  # [129, 10]
    
    # Update fc2 weights and bias
    w["fc2.weight"] = W_final[:128, :].T.astype(np.float32)   # [10, 128]
    w["fc2.bias"] = W_final[128, :].astype(np.float32)        # [10]
    
    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_acc),
        "fc2_weight_mean": float(w["fc2.weight"].mean()),
        "fc2_weight_std": float(w["fc2.weight"].std()),
        "fc2_bias_mean": float(w["fc2.bias"].mean()),
        "fc2_bias_std": float(w["fc2.bias"].std()),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
