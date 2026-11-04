import numpy as np
import math

def edit(w, f):
    # Extract original feedback data
    x_orig = f['x']          # (2000,1,28,28)
    y_orig = f['y']          # (2000,)
    F_orig = f['features']   # (2000,1568)
    
    # Compute feature mean and std from original features
    mean = F_orig.mean(axis=0)
    std = F_orig.std(axis=0) + 1e-6
    
    # Standardize original features
    F_orig_std = (F_orig - mean) / std
    
    # --- Hyperparameter selection via 5-fold CV on original features ---
    n = F_orig_std.shape[0]
    indices = np.arange(n)
    np.random.seed(0)
    np.random.shuffle(indices)
    folds = np.array_split(indices, 5)
    
    lambdas = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_lambda = 1.0
    best_acc = -1.0
    
    # One-hot labels
    Y = np.zeros((n, 10), dtype=np.float32)
    Y[np.arange(n), y_orig] = 1.0
    
    for lam in lambdas:
        accs = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            
            X_train = F_orig_std[train_idx]
            Y_train = Y[train_idx]
            X_val = F_orig_std[val_idx]
            y_val = y_orig[val_idx]
            
            # Add bias column
            X_train_aug = np.hstack([X_train, np.ones((len(train_idx), 1), dtype=np.float32)])
            X_val_aug = np.hstack([X_val, np.ones((len(val_idx), 1), dtype=np.float32)])
            
            # Ridge solution: W = (X^T X + lambda I)^-1 X^T Y
            # We only regularize weights, not bias (last column)
            d = X_train_aug.shape[1]
            reg = np.eye(d, dtype=np.float32) * lam
            reg[-1, -1] = 0.0  # do not regularize bias
            
            XtX = X_train_aug.T @ X_train_aug
            XtY = X_train_aug.T @ Y_train
            W = np.linalg.solve(XtX + reg, XtY)  # (d, 10)
            
            # Predict on validation
            logits_val = X_val_aug @ W
            preds = np.argmax(logits_val, axis=1)
            acc = np.mean(preds == y_val)
            accs.append(acc)
        
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lambda = lam
    
    # --- Generate augmented data from all original images ---
    # We will create 9 shifted versions per image (dx, dy in {-2,0,2})
    shifts = [(-2,-2), (-2,0), (-2,2), (0,-2), (0,0), (0,2), (2,-2), (2,0), (2,2)]
    B = x_orig.shape[0]
    # Pad images once
    x_pad = np.pad(x_orig, ((0,0), (0,0), (2,2), (2,2)), mode='constant')
    
    # We will process in batches to save memory
    batch_size = 500
    F_aug_list = []
    y_aug_list = []
    
    # Define conv2d and maxpool2d for feature computation
    def conv2d(x, weight, bias, padding=1):
        N, C_in, H, W = x.shape
        C_out, _, kH, kW = weight.shape
        x_pad = np.pad(x, ((0,0), (0,0), (padding, padding), (padding, padding)), mode='constant')
        patches = np.lib.stride_tricks.sliding_window_view(x_pad, (kH, kW), axis=(2,3))
        out = np.einsum('nchwkl,ockl->nohw', patches, weight) + bias.reshape(1, C_out, 1, 1)
        return out
    
    def maxpool2d(x, kernel_size=2, stride=2):
        N, C, H, W = x.shape
        H_out = H // stride
        W_out = W // stride
        x_reshaped = x.reshape(N, C, H_out, stride, W_out, stride)
        out = x_reshaped.max(axis=(3,5))
        return out
    
    def compute_features(x_batch, w):
        h = conv2d(x_batch, w['conv1.weight'], w['conv1.bias'], padding=1)
        h = np.maximum(h, 0)
        h = maxpool2d(h, 2, 2)
        h = conv2d(h, w['conv2.weight'], w['conv2.bias'], padding=1)
        h = np.maximum(h, 0)
        h = maxpool2d(h, 2, 2)
        h = h.reshape(h.shape[0], -1)
        return h
    
    for start in range(0, B, batch_size):
        end = min(start + batch_size, B)
        x_batch = x_orig[start:end]  # (b,1,28,28)
        y_batch = y_orig[start:end]
        b = x_batch.shape[0]
        
        # Pad batch
        x_batch_pad = np.pad(x_batch, ((0,0), (0,0), (2,2), (2,2)), mode='constant')
        
        for dx, dy in shifts:
            # Crop shifted version
            x_shift = x_batch_pad[:, :, 2+dy:2+dy+28, 2+dx:2+dx+28]
            # Compute features
            F_shift = compute_features(x_shift, w)
            F_aug_list.append(F_shift)
            y_aug_list.append(y_batch)
    
    F_aug = np.vstack(F_aug_list)
    y_aug = np.concatenate(y_aug_list)
    
    # Standardize augmented features using original mean/std
    F_aug_std = (F_aug - mean) / std
    
    # --- Train final head on all augmented data ---
    Y_aug = np.zeros((len(y_aug), 10), dtype=np.float32)
    Y_aug[np.arange(len(y_aug)), y_aug] = 1.0
    
    X_aug = np.hstack([F_aug_std, np.ones((len(y_aug), 1), dtype=np.float32)])
    d = X_aug.shape[1]
    reg = np.eye(d, dtype=np.float32) * best_lambda
    reg[-1, -1] = 0.0
    
    XtX = X_aug.T @ X_aug
    XtY = X_aug.T @ Y_aug
    W_aug = np.linalg.solve(XtX + reg, XtY)  # (d, 10)
    
    # Extract weights and bias for standardized features
    W_std = W_aug[:-1, :].T  # (10, 1568)
    b_std = W_aug[-1, :]     # (10,)
    
    # Convert back to original feature scale
    # logits = F_std @ W_std.T + b_std = ((F - mean)/std) @ W_std.T + b_std
    #        = F @ (W_std/std).T + (b_std - mean @ (W_std/std).T)
    W_new = W_std / std[None, :]
    b_new = b_std - mean @ W_new.T
    
    # Update weights
    w['head.weight'] = W_new.astype(np.float32)
    w['head.bias'] = b_new.astype(np.float32)
    
    # Diagnostics
    diagnostics = {
        'best_lambda': float(best_lambda),
        'best_cv_acc': float(best_acc),
        'head_weight_norm': float(np.linalg.norm(w['head.weight'])),
        'head_bias_norm': float(np.linalg.norm(w['head.bias'])),
        'head_weight_max_abs': float(np.max(np.abs(w['head.weight']))),
        'head_bias_max_abs': float(np.max(np.abs(w['head.bias']))),
        'num_augmented': int(len(y_aug))
    }
    
    return {"weights": w, "diagnostics": diagnostics}
