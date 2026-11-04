import numpy as np

def edit(w, f):
    # Read current conv weights
    conv1_w = w["conv1.weight"].astype(np.float32)
    conv1_b = w["conv1.bias"].astype(np.float32)
    conv2_w = w["conv2.weight"].astype(np.float32)
    conv2_b = w["conv2.bias"].astype(np.float32)

    # Helper: conv2d with padding=1, stride=1
    def conv2d(x, weight, bias, padding=1):
        N, C_in, H, W = x.shape
        C_out, _, K, _ = weight.shape
        x_pad = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant')
        windows = np.lib.stride_tricks.sliding_window_view(x_pad, (K, K), axis=(2,3))
        # windows: (N, C_in, H, W, K, K)
        windows = windows.transpose(0,2,3,1,4,5).reshape(N, H, W, C_in*K*K)
        weight_flat = weight.reshape(C_out, C_in*K*K)
        out = windows @ weight_flat.T + bias
        out = out.transpose(0,3,1,2)
        return out

    def relu(x):
        return np.maximum(x, 0)

    def maxpool2d(x):
        N, C, H, W = x.shape
        x = x.reshape(N, C, H//2, 2, W//2, 2)
        return x.max(axis=(3,5))

    def forward_features(x):
        h = conv2d(x, conv1_w, conv1_b, padding=1)
        h = relu(h)
        h = maxpool2d(h)
        h = conv2d(h, conv2_w, conv2_b, padding=1)
        h = relu(h)
        h = maxpool2d(h)
        return h.reshape(x.shape[0], -1)

    # Original data
    x_orig = f["x"].astype(np.float32)  # (2000,1,28,28)
    y_orig = f["y"].astype(np.int64)    # (2000,)

    # Build augmented images: original + 4 shifts + horizontal flip
    N = x_orig.shape[0]
    aug_list = [x_orig]
    # shift left
    aug = np.zeros_like(x_orig)
    aug[:,:,:,:-1] = x_orig[:,:,:,1:]
    aug_list.append(aug)
    # shift right
    aug = np.zeros_like(x_orig)
    aug[:,:,:,1:] = x_orig[:,:,:,:-1]
    aug_list.append(aug)
    # shift up
    aug = np.zeros_like(x_orig)
    aug[:,:,:-1,:] = x_orig[:,:,1:,:]
    aug_list.append(aug)
    # shift down
    aug = np.zeros_like(x_orig)
    aug[:,:,1:,:] = x_orig[:,:,:-1,:]
    aug_list.append(aug)
    # horizontal flip
    aug_list.append(x_orig[:,:,:,::-1].copy())

    x_aug = np.concatenate(aug_list, axis=0)  # (12000,1,28,28)
    y_aug = np.tile(y_orig, 6)                # (12000,)

    # Compute features for augmented data
    X_aug = forward_features(x_aug)  # (12000,1568)
    X_aug = X_aug.astype(np.float32)

    # Also compute original features for validation split
    X_orig = forward_features(x_orig)  # (2000,1568)
    X_orig = X_orig.astype(np.float32)

    # Simple hold-out: use first 1600 original for train, last 400 for val
    # But we want to use augmented data for training. We'll use augmented data from first 1600 original,
    # and validate on original last 400.
    train_idx = np.arange(1600)
    val_idx = np.arange(1600, 2000)

    # Build augmented training set from first 1600 original
    # We need to map augmented indices: each original image has 6 copies.
    # For original index i, augmented indices are i, i+N, i+2N, ... i+5N.
    # So for train_idx, we can gather all 6 copies.
    aug_train_idx = []
    for i in train_idx:
        for k in range(6):
            aug_train_idx.append(i + k * N)
    aug_train_idx = np.array(aug_train_idx)
    X_train = X_aug[aug_train_idx]
    y_train = y_aug[aug_train_idx]

    X_val = X_orig[val_idx]
    y_val = y_orig[val_idx]

    # Ridge regression with bias: augment X with ones
    def fit_ridge(X, y, lam):
        # X: (n, d), y: (n,)
        n, d = X.shape
        X_aug = np.hstack([X, np.ones((n,1), dtype=np.float32)])
        # One-hot y
        Y = np.zeros((n, 10), dtype=np.float32)
        Y[np.arange(n), y] = 1.0
        # Solve (X^T X + lam I) W = X^T Y
        A = X_aug.T @ X_aug + lam * np.eye(d+1, dtype=np.float32)
        B = X_aug.T @ Y
        W = np.linalg.solve(A, B)
        return W[:-1].T, W[-1]  # weight (10,d), bias (10,)

    # Try a few lambdas
    best_lam = 0.001
    best_acc = -1.0
    for lam in [0.0001, 0.001, 0.01, 0.1, 1.0]:
        W, b = fit_ridge(X_train, y_train, lam)
        logits = X_val @ W.T + b
        pred = np.argmax(logits, axis=1)
        acc = np.mean(pred == y_val)
        if acc > best_acc:
            best_acc = acc
            best_lam = lam

    # Final fit on all augmented data
    X_all = X_aug
    y_all = y_aug
    W_final, b_final = fit_ridge(X_all, y_all, best_lam)

    # Update head weights
    w["head.weight"] = W_final.astype(np.float32)
    w["head.bias"] = b_final.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lam),
        "val_accuracy": float(best_acc),
        "train_accuracy": float(np.mean(np.argmax(X_all @ W_final.T + b_final, axis=1) == y_all)),
        "head_weight_norm": float(np.linalg.norm(W_final)),
        "head_bias_norm": float(np.linalg.norm(b_final)),
        "augmented_samples": int(X_all.shape[0])
    }
    return {"weights": w, "diagnostics": diagnostics}
