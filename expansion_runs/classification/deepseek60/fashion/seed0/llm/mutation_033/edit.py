import numpy as np

def edit(w, f):
    # Extract current weights
    conv1_w = w["conv1.weight"].astype(np.float32)
    conv1_b = w["conv1.bias"].astype(np.float32)
    conv2_w = w["conv2.weight"].astype(np.float32)
    conv2_b = w["conv2.bias"].astype(np.float32)
    head_w = w["head.weight"].astype(np.float32)
    head_b = w["head.bias"].astype(np.float32)

    # Feedback data
    x = f["x"].astype(np.float32)          # (2000,1,28,28)
    y = f["y"].astype(np.int64)            # (2000,)
    X_orig = f["features"].astype(np.float32)  # (2000,1568)

    # Helper: conv2d with padding=1, stride=1
    def conv2d_batch(xb, weight, bias, padding=1):
        N, C, H, W = xb.shape
        out_channels, _, kH, kW = weight.shape
        x_pad = np.pad(xb, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant')
        patches = np.lib.stride_tricks.sliding_window_view(x_pad, (kH, kW), axis=(2,3))
        # patches: (N, C, H_out, W_out, kH, kW)
        patches = patches.transpose(0, 2, 3, 1, 4, 5)
        patches = patches.reshape(-1, C*kH*kW)
        weight_flat = weight.reshape(out_channels, -1)
        out = patches @ weight_flat.T + bias
        out = out.reshape(N, H, W, out_channels).transpose(0, 3, 1, 2)
        return out

    def maxpool2d(xb, kernel=2, stride=2):
        N, C, H, W = xb.shape
        H_out = H // kernel
        W_out = W // kernel
        xb = xb.reshape(N, C, H_out, kernel, W_out, kernel)
        return xb.max(axis=(3,5))

    def compute_features(xb, batch_size=500):
        feats = []
        for i in range(0, len(xb), batch_size):
            x_b = xb[i:i+batch_size]
            h1 = conv2d_batch(x_b, conv1_w, conv1_b, padding=1)
            h1 = np.maximum(h1, 0)
            h1 = maxpool2d(h1, 2, 2)
            h2 = conv2d_batch(h1, conv2_w, conv2_b, padding=1)
            h2 = np.maximum(h2, 0)
            h2 = maxpool2d(h2, 2, 2)
            feats.append(h2.reshape(h2.shape[0], -1))
        return np.vstack(feats)

    # Features for horizontally flipped images
    x_flip = x[:, :, :, ::-1]
    X_flip = compute_features(x_flip)

    # Augmented training set
    X = np.vstack([X_orig, X_flip])          # (4000,1568)
    y_aug = np.concatenate([y, y])           # (4000,)
    N = X.shape[0]

    # One-hot targets
    Y = np.zeros((N, 10), dtype=np.float32)
    Y[np.arange(N), y_aug] = 1.0

    # Standardize features
    mean_X = np.mean(X, axis=0)
    std_X = np.std(X, axis=0) + 1e-8
    X_scaled = (X - mean_X) / std_X

    # Center targets
    mean_Y = np.mean(Y, axis=0)
    Y_c = Y - mean_Y

    # Precompute full XTX and XTY for fast CV
    full_XTX = X_scaled.T @ X_scaled
    full_XTY = X_scaled.T @ Y_c

    # 5-fold CV to choose lambda
    n = X_scaled.shape[0]
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, 5)

    lambdas = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]
    best_acc = -1.0
    best_lambda = 1.0

    for lam in lambdas:
        accs = []
        for i in range(5):
            val_idx = folds[i]
            tr_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            X_val = X_scaled[val_idx]
            Y_val = Y_c[val_idx]
            XTX_val = X_val.T @ X_val
            XTY_val = X_val.T @ Y_val
            XTX_tr = full_XTX - XTX_val
            XTY_tr = full_XTY - XTY_val
            # Solve ridge
            W = np.linalg.solve(XTX_tr + lam * np.eye(XTX_tr.shape[0]), XTY_tr)
            # Predict on validation
            logits = X_val @ W
            preds = np.argmax(logits, axis=1)
            # True labels for validation: need original y for val indices
            # Since Y_c is centered, argmax of logits is same as argmax of Y? Actually logits = X_val @ W, and we want to compare to y_aug.
            # But we don't have y_aug directly for val_idx? We can reconstruct from Y_val + mean_Y.
            # Simpler: use y_aug[val_idx]
            true = y_aug[val_idx]
            acc = np.mean(preds == true)
            accs.append(acc)
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lambda = lam

    # Fit final head on full augmented data
    W_scaled = np.linalg.solve(full_XTX + best_lambda * np.eye(full_XTX.shape[0]), full_XTY)
    b_scaled = mean_Y

    # Transform back to original feature space
    W_orig = W_scaled / std_X[:, None]
    b_orig = b_scaled - mean_X @ W_orig

    # Set new head weights (shape [10, 1568])
    w["head.weight"] = W_orig.T.astype(np.float32)
    w["head.bias"] = b_orig.astype(np.float32)

    # Diagnostics
    diag = {
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_acc),
        "head_weight_mean": float(np.mean(w["head.weight"])),
        "head_weight_std": float(np.std(w["head.weight"])),
        "head_weight_max_abs": float(np.max(np.abs(w["head.weight"]))),
        "head_bias_mean": float(np.mean(w["head.bias"])),
        "head_bias_std": float(np.std(w["head.bias"])),
        "num_train": int(N),
    }
    return {"weights": w, "diagnostics": diag}
