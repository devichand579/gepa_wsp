import numpy as np
import math

def edit(w, f):
    # Extract feedback arrays
    X_orig = f["features"].astype(np.float32)   # (2000, 1568)
    y_orig = f["y"].astype(np.int64)            # (2000,)
    x_orig = f["x"].astype(np.float32)          # (2000, 1, 28, 28)

    # --- Helper functions for forward pass on flipped images ---
    def conv2d(x, weight, bias, padding=1):
        N, C_in, H, W = x.shape
        C_out, _, K, _ = weight.shape
        x_pad = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant')
        out = np.zeros((N, C_out, H, W), dtype=np.float32)
        for i in range(K):
            for j in range(K):
                x_slice = x_pad[:, :, i:i+H, j:j+W]
                w_slice = weight[:, :, i, j]
                out += np.einsum('nchw,oc->nohw', x_slice, w_slice)
        out += bias.reshape(1, C_out, 1, 1)
        return out

    def maxpool2d(x, kernel=2, stride=2):
        N, C, H, W = x.shape
        H_out = H // kernel
        W_out = W // kernel
        x_reshaped = x.reshape(N, C, H_out, kernel, W_out, kernel)
        return x_reshaped.max(axis=(3,5))

    # Horizontal flip
    x_flip = x_orig[:, :, :, ::-1].copy()

    # Forward pass for flipped images using current conv weights
    h = conv2d(x_flip, w["conv1.weight"], w["conv1.bias"], padding=1)
    h = np.maximum(h, 0)
    h = maxpool2d(h, 2, 2)
    h = conv2d(h, w["conv2.weight"], w["conv2.bias"], padding=1)
    h = np.maximum(h, 0)
    h = maxpool2d(h, 2, 2)
    X_flip = h.reshape(h.shape[0], -1)

    # Combine original and flipped features
    X = np.vstack([X_orig, X_flip]).astype(np.float32)
    y = np.concatenate([y_orig, y_orig]).astype(np.int64)

    # One-hot targets
    Y = np.eye(10, dtype=np.float32)[y]

    # Standardize features
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X_std = (X - mean) / std

    # Cross-validation to choose ridge lambda
    lambdas = [0.01, 0.1, 1.0, 10.0, 100.0]
    n_folds = 5
    indices = np.random.permutation(len(y))
    fold_size = len(y) // n_folds
    best_lambda = 1.0
    best_acc = -1.0

    for lam in lambdas:
        accs = []
        for fold in range(n_folds):
            val_idx = indices[fold*fold_size:(fold+1)*fold_size]
            train_idx = np.setdiff1d(indices, val_idx)
            X_train = X_std[train_idx]
            Y_train = Y[train_idx]
            X_val = X_std[val_idx]
            Y_val = y[val_idx]

            X_mean = X_train.mean(axis=0)
            Y_mean = Y_train.mean(axis=0)
            X_centered = X_train - X_mean
            Y_centered = Y_train - Y_mean

            # SVD-based ridge solution
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            D = S / (S**2 + lam)
            UT_Y = U.T @ Y_centered
            W = Vt.T @ (D[:, None] * UT_Y)
            bias = Y_mean - X_mean @ W

            scores = X_val @ W + bias
            pred = scores.argmax(axis=1)
            acc = (pred == Y_val).mean()
            accs.append(acc)
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lambda = lam

    # Final fit on all data with best lambda
    X_mean = X_std.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    X_centered = X_std - X_mean
    Y_centered = Y - Y_mean

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    D = S / (S**2 + best_lambda)
    UT_Y = U.T @ Y_centered
    W_std = Vt.T @ (D[:, None] * UT_Y)
    bias_std = Y_mean - X_mean @ W_std

    # Convert back to original feature space
    # X_std = (X - mean) / std
    # logits = W_std @ X_std + bias_std
    #        = (W_std / std) @ X + (bias_std - W_std @ (mean / std))
    head_weight = (W_std / std[:, None]).astype(np.float32)
    head_bias = (bias_std - W_std @ (mean / std)).astype(np.float32)

    # Clip to safe range
    head_weight = np.clip(head_weight, -1000, 1000)
    head_bias = np.clip(head_bias, -1000, 1000)

    # Update weights
    w["head.weight"] = head_weight
    w["head.bias"] = head_bias

    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_acc),
        "head_weight_mean": float(head_weight.mean()),
        "head_weight_std": float(head_weight.std()),
        "head_weight_max_abs": float(np.abs(head_weight).max()),
        "head_bias_mean": float(head_bias.mean()),
        "head_bias_std": float(head_bias.std()),
    }

    return {"weights": w, "diagnostics": diagnostics}
