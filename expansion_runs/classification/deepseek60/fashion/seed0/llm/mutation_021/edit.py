import numpy as np
import math

def edit(w, f):
    # Extract feedback data
    x = f["x"]  # (2000, 1, 28, 28)
    y = f["y"]  # (2000,)
    N = x.shape[0]
    
    # Helper: convolution with padding=1, stride=1
    def conv2d(x, weight, bias, padding=1):
        # x: (N, C_in, H, W)
        # weight: (C_out, C_in, K, K)
        # bias: (C_out,)
        N, C_in, H, W = x.shape
        C_out, _, K, _ = weight.shape
        if padding > 0:
            x_pad = np.pad(x, ((0,0), (0,0), (padding, padding), (padding, padding)), mode='constant')
        else:
            x_pad = x
        H_out = H + 2*padding - K + 1
        W_out = W + 2*padding - K + 1
        # Use sliding_window_view for patches
        patches = np.lib.stride_tricks.sliding_window_view(x_pad, (K, K), axis=(2,3))
        # patches shape: (N, C_in, H_out, W_out, K, K)
        # Reshape for matmul: (N, H_out, W_out, C_in*K*K)
        patches = patches.transpose(0, 2, 3, 1, 4, 5).reshape(N, H_out, W_out, C_in*K*K)
        weight_flat = weight.reshape(C_out, C_in*K*K).T  # (C_in*K*K, C_out)
        # Matmul: (N, H_out, W_out, C_in*K*K) @ (C_in*K*K, C_out) -> (N, H_out, W_out, C_out)
        out = patches @ weight_flat
        out = out.transpose(0, 3, 1, 2)  # (N, C_out, H_out, W_out)
        out += bias.reshape(1, C_out, 1, 1)
        return out

    def maxpool2d(x, kernel=2, stride=2):
        N, C, H, W = x.shape
        H_out = H // kernel
        W_out = W // kernel
        x_reshaped = x.reshape(N, C, H_out, kernel, W_out, kernel)
        return x_reshaped.max(axis=(3, 5))

    # --- Step 1: Learn conv1 filters via PCA on 3x3 patches ---
    # Pad images with 1 zero on each side
    x_pad = np.pad(x, ((0,0), (0,0), (1,1), (1,1)), mode='constant')
    # Extract 3x3 patches: (N, 28, 28, 1, 3, 3) -> (N*28*28, 9)
    patches1 = np.lib.stride_tricks.sliding_window_view(x_pad, (3,3), axis=(2,3))
    # patches1 shape: (N, 1, 28, 28, 3, 3)
    patches1 = patches1.transpose(0, 2, 3, 1, 4, 5).reshape(-1, 9)
    mean1 = patches1.mean(axis=0)
    patches1_centered = patches1 - mean1
    cov1 = patches1_centered.T @ patches1_centered / (patches1.shape[0] - 1)
    eigvals1, eigvecs1 = np.linalg.eigh(cov1)
    # Sort descending
    idx1 = np.argsort(eigvals1)[::-1]
    eigvecs1 = eigvecs1[:, idx1]
    # Take top 16 components
    comp1 = eigvecs1[:, :16]  # (9, 16)
    # Reshape to (16, 1, 3, 3)
    conv1_weight = comp1.T.reshape(16, 1, 3, 3)
    # Bias: -dot(mean1, comp)
    conv1_bias = -comp1.T @ mean1  # (16,)
    
    # --- Step 2: Compute conv1 output for all images ---
    # Process in batches to save memory
    batch_size = 200
    conv1_out_list = []
    for i in range(0, N, batch_size):
        xb = x[i:i+batch_size]
        out = conv2d(xb, conv1_weight, conv1_bias, padding=1)
        out = np.maximum(out, 0)  # ReLU
        out = maxpool2d(out, 2, 2)  # (B, 16, 14, 14)
        conv1_out_list.append(out)
    conv1_out = np.concatenate(conv1_out_list, axis=0)  # (N, 16, 14, 14)
    
    # --- Step 3: Learn conv2 filters via PCA on 3x3 patches of conv1_out ---
    # Pad conv1_out with 1 zero on each side
    conv1_pad = np.pad(conv1_out, ((0,0), (0,0), (1,1), (1,1)), mode='constant')
    # Extract 3x3 patches: (N, 16, 14, 14, 3, 3) -> (N*14*14, 16*9)
    patches2 = np.lib.stride_tricks.sliding_window_view(conv1_pad, (3,3), axis=(2,3))
    # patches2 shape: (N, 16, 14, 14, 3, 3)
    patches2 = patches2.transpose(0, 2, 3, 1, 4, 5).reshape(-1, 16*9)
    mean2 = patches2.mean(axis=0)
    patches2_centered = patches2 - mean2
    cov2 = patches2_centered.T @ patches2_centered / (patches2.shape[0] - 1)
    eigvals2, eigvecs2 = np.linalg.eigh(cov2)
    idx2 = np.argsort(eigvals2)[::-1]
    eigvecs2 = eigvecs2[:, idx2]
    # Take top 32 components
    comp2 = eigvecs2[:, :32]  # (144, 32)
    # Reshape to (32, 16, 3, 3)
    conv2_weight = comp2.T.reshape(32, 16, 3, 3)
    conv2_bias = -comp2.T @ mean2  # (32,)
    
    # --- Step 4: Compute conv2 output for all images ---
    conv2_out_list = []
    for i in range(0, N, batch_size):
        xb = conv1_out[i:i+batch_size]
        out = conv2d(xb, conv2_weight, conv2_bias, padding=1)
        out = np.maximum(out, 0)  # ReLU
        out = maxpool2d(out, 2, 2)  # (B, 32, 7, 7)
        conv2_out_list.append(out)
    conv2_out = np.concatenate(conv2_out_list, axis=0)  # (N, 32, 7, 7)
    features = conv2_out.reshape(N, -1)  # (N, 1568)
    
    # --- Step 5: Train ridge classifier on features ---
    # Standardize features
    feat_mean = features.mean(axis=0)
    feat_std = features.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    features_std = (features - feat_mean) / feat_std
    
    # One-hot targets
    Y = np.zeros((N, 10))
    Y[np.arange(N), y] = 1.0
    
    # Center features and targets
    X_centered = features_std - features_std.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)
    
    # Cross-validation to choose lambda
    # 5-fold CV
    n_folds = 5
    fold_size = N // n_folds
    lambdas = np.logspace(-3, 3, 13)
    best_lambda = 1.0
    best_cv_acc = 0.0
    
    # Precompute X^T X and X^T Y for full data? For CV we need per fold.
    # We'll do simple loop over folds.
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    for lam in lambdas:
        cv_acc = 0.0
        for fold in range(n_folds):
            val_idx = indices[fold*fold_size:(fold+1)*fold_size]
            train_idx = np.setdiff1d(indices, val_idx)
            X_train = X_centered[train_idx]
            Y_train = Y_centered[train_idx]
            X_val = X_centered[val_idx]
            Y_val = Y[val_idx]  # original one-hot for accuracy
            
            # Solve ridge: W = (X^T X + lam I)^-1 X^T Y
            # Use SVD or solve. X is (N_train, D). D=1568. N_train ~1600.
            # It's more efficient to solve in dual if N < D, but here N~1600, D=1568, similar.
            # We can use np.linalg.solve on (D,D) matrix. D=1568, so 1568x1568 is fine.
            XtX = X_train.T @ X_train
            XtY = X_train.T @ Y_train
            W = np.linalg.solve(XtX + lam * np.eye(XtX.shape[0]), XtY)  # (D, 10)
            # Bias: since we centered, bias = -mean_X @ W + mean_Y. But we can just compute predictions on centered val.
            # Actually, we need to add back the means. For accuracy, we can just compute logits = X_val @ W + b.
            # b = Y.mean(axis=0) - features_std.mean(axis=0) @ W
            # But we used X_centered = features_std - features_std.mean(axis=0). So X_val is already centered.
            # The model is: logits = X_centered @ W + Y.mean(axis=0)
            logits_val = X_val @ W + Y.mean(axis=0)
            preds = np.argmax(logits_val, axis=1)
            true_labels = np.argmax(Y_val, axis=1)
            cv_acc += np.mean(preds == true_labels)
        cv_acc /= n_folds
        if cv_acc > best_cv_acc:
            best_cv_acc = cv_acc
            best_lambda = lam
    
    # Train final model on all data with best_lambda
    XtX = X_centered.T @ X_centered
    XtY = X_centered.T @ Y_centered
    W = np.linalg.solve(XtX + best_lambda * np.eye(XtX.shape[0]), XtY)  # (1568, 10)
    W = W.T  # (10, 1568)
    b = Y.mean(axis=0) - features_std.mean(axis=0) @ W.T  # (10,)
    
    # --- Step 6: Update weights ---
    w["conv1.weight"] = conv1_weight.astype(np.float32)
    w["conv1.bias"] = conv1_bias.astype(np.float32)
    w["conv2.weight"] = conv2_weight.astype(np.float32)
    w["conv2.bias"] = conv2_bias.astype(np.float32)
    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)
    
    # Clip to [-1000, 1000] just in case
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)
    
    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_cv_acc),
        "conv1_weight_std": float(conv1_weight.std()),
        "conv2_weight_std": float(conv2_weight.std()),
        "head_weight_mean": float(W.mean()),
        "head_weight_std": float(W.std()),
        "head_weight_max_abs": float(np.abs(W).max()),
        "head_bias_mean": float(b.mean()),
        "head_bias_std": float(b.std()),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
