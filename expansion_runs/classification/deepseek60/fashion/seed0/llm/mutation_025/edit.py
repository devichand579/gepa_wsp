import numpy as np

def edit(w, f):
    # Extract data
    x = f["x"]                     # (2000,1,28,28)
    y = f["y"]                     # (2000,)
    X_orig = f["features"]         # (2000,1568)

    # Fast conv2d helper using sliding_window_view
    def conv2d(x, weight, bias, padding=1):
        N, C_in, H, W = x.shape
        C_out, _, K, _ = weight.shape
        x_pad = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant')
        patches = np.lib.stride_tricks.sliding_window_view(x_pad, (K, K), axis=(2,3))
        patches = patches.transpose(0, 2, 3, 1, 4, 5).reshape(N, H, W, C_in*K*K)
        weight_flat = weight.reshape(C_out, -1)
        out = patches @ weight_flat.T + bias
        return out.transpose(0, 3, 1, 2)

    def extract_features(images, w):
        out1 = conv2d(images, w["conv1.weight"], w["conv1.bias"], padding=1)
        out1 = np.maximum(out1, 0)
        N, C, H, W = out1.shape
        out1 = out1.reshape(N, C, H//2, 2, W//2, 2).max(axis=(3,5))
        out2 = conv2d(out1, w["conv2.weight"], w["conv2.bias"], padding=1)
        out2 = np.maximum(out2, 0)
        N, C, H, W = out2.shape
        out2 = out2.reshape(N, C, H//2, 2, W//2, 2).max(axis=(3,5))
        return out2.reshape(N, -1)

    # Compute features for horizontally flipped images
    x_flip = x[:, :, :, ::-1].copy()
    X_flip = extract_features(x_flip, w)

    # Cross-validation to choose ridge lambda
    n_orig = X_orig.shape[0]
    indices = np.arange(n_orig)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, 5)

    lambdas = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    best_lambda = None
    best_acc = -1.0
    best_W = None

    for lam in lambdas:
        accs = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])

            X_train = np.concatenate([X_orig[train_idx], X_flip[train_idx]], axis=0)
            Y_train = np.concatenate([y[train_idx], y[train_idx]], axis=0)
            X_val = np.concatenate([X_orig[val_idx], X_flip[val_idx]], axis=0)
            Y_val = np.concatenate([y[val_idx], y[val_idx]], axis=0)

            Y_train_oh = np.zeros((Y_train.shape[0], 10), dtype=np.float32)
            Y_train_oh[np.arange(Y_train.shape[0]), Y_train] = 1.0
            Y_val_oh = np.zeros((Y_val.shape[0], 10), dtype=np.float32)
            Y_val_oh[np.arange(Y_val.shape[0]), Y_val] = 1.0

            X_train_aug = np.concatenate([X_train, np.ones((X_train.shape[0], 1), dtype=np.float32)], axis=1)
            X_val_aug = np.concatenate([X_val, np.ones((X_val.shape[0], 1), dtype=np.float32)], axis=1)

            if lam == 0.0:
                W, _, _, _ = np.linalg.lstsq(X_train_aug, Y_train_oh, rcond=None)
            else:
                A = X_train_aug.T @ X_train_aug + lam * np.eye(X_train_aug.shape[1])
                B = X_train_aug.T @ Y_train_oh
                W = np.linalg.solve(A, B)

            logits = X_val_aug @ W
            preds = np.argmax(logits, axis=1)
            true = np.argmax(Y_val_oh, axis=1)
            accs.append(np.mean(preds == true))

        mean_acc = float(np.mean(accs))
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lambda = lam

            X_full = np.concatenate([X_orig, X_flip], axis=0)
            Y_full = np.concatenate([y, y], axis=0)
            Y_full_oh = np.zeros((Y_full.shape[0], 10), dtype=np.float32)
            Y_full_oh[np.arange(Y_full.shape[0]), Y_full] = 1.0
            X_full_aug = np.concatenate([X_full, np.ones((X_full.shape[0], 1), dtype=np.float32)], axis=1)

            if lam == 0.0:
                W_full, _, _, _ = np.linalg.lstsq(X_full_aug, Y_full_oh, rcond=None)
            else:
                A = X_full_aug.T @ X_full_aug + lam * np.eye(X_full_aug.shape[1])
                B = X_full_aug.T @ Y_full_oh
                W_full = np.linalg.solve(A, B)
            best_W = W_full

    # best_W shape: (1569, 10). Last row is bias.
    W_head = best_W[:-1, :].T.astype(np.float32)   # (10,1568)
    b_head = best_W[-1, :].astype(np.float32)      # (10,)

    # Clip to allowed range
    W_head = np.clip(W_head, -1000.0, 1000.0)
    b_head = np.clip(b_head, -1000.0, 1000.0)

    w["head.weight"] = W_head
    w["head.bias"] = b_head

    diagnostics = {
        "best_lambda": best_lambda,
        "best_cv_accuracy": best_acc,
        "head_weight_mean": float(np.mean(W_head)),
        "head_weight_std": float(np.std(W_head)),
        "head_bias_mean": float(np.mean(b_head)),
        "head_bias_std": float(np.std(b_head)),
    }
    return {"weights": w, "diagnostics": diagnostics}
