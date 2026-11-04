import numpy as np

def edit(w, f):
    # Extract feedback data
    x = f["x"]                     # (2000,1,28,28)
    y = f["y"]                     # (2000,)
    features = f["features"]       # (2000,1568)

    # Helper: compute features for a batch of images using current conv weights
    def compute_features_batch(x_batch, w):
        N = x_batch.shape[0]
        w1 = w["conv1.weight"]     # (16,1,3,3)
        b1 = w["conv1.bias"]       # (16,)
        w2 = w["conv2.weight"]     # (32,16,3,3)
        b2 = w["conv2.bias"]       # (32,)

        # conv1 + relu + maxpool2d
        x_pad = np.pad(x_batch, ((0,0),(0,0),(1,1),(1,1)), mode='constant')
        patches = np.lib.stride_tricks.sliding_window_view(x_pad, (3,3), axis=(2,3))
        out1 = np.einsum('nchwkl,ockl->nohw', patches, w1)
        out1 += b1[None, :, None, None]
        out1 = np.maximum(out1, 0.0)
        out1 = out1.reshape(N, 16, 14, 2, 14, 2).max(axis=(3,5))

        # conv2 + relu + maxpool2d
        x_pad2 = np.pad(out1, ((0,0),(0,0),(1,1),(1,1)), mode='constant')
        patches2 = np.lib.stride_tricks.sliding_window_view(x_pad2, (3,3), axis=(2,3))
        out2 = np.einsum('nchwkl,ockl->nohw', patches2, w2)
        out2 += b2[None, :, None, None]
        out2 = np.maximum(out2, 0.0)
        out2 = out2.reshape(N, 32, 7, 2, 7, 2).max(axis=(3,5))
        return out2.reshape(N, -1)

    # Compute features for horizontally flipped images
    x_flipped = x[:, :, :, ::-1]
    features_flipped = compute_features_batch(x_flipped, w)

    # Augment training set
    X = np.vstack([features, features_flipped])          # (4000,1568)
    y_aug = np.concatenate([y, y])                       # (4000,)

    # One-hot targets
    N = X.shape[0]
    Y = np.zeros((N, 10), dtype=np.float64)
    Y[np.arange(N), y_aug] = 1.0

    # Augment with bias column
    X_aug = np.hstack([X, np.ones((N, 1), dtype=np.float64)])  # (4000,1569)

    # Ridge regression with bias unregularized
    lambda_reg = 1000.0
    XTX = X_aug.T @ X_aug
    XTY = X_aug.T @ Y
    reg = np.eye(X_aug.shape[1]) * lambda_reg
    reg[-1, -1] = 0.0
    W_aug = np.linalg.solve(XTX + reg, XTY)              # (1569,10)

    head_weight = W_aug[:-1, :].T                        # (10,1568)
    head_bias = W_aug[-1, :]                             # (10,)

    # Clip to safe range and cast to float32
    head_weight = np.clip(head_weight, -1000.0, 1000.0).astype(np.float32)
    head_bias = np.clip(head_bias, -1000.0, 1000.0).astype(np.float32)

    # Update weights
    w["head.weight"] = head_weight
    w["head.bias"] = head_bias

    # Diagnostics
    diagnostics = {
        "head_weight_mean": float(np.mean(head_weight)),
        "head_weight_std": float(np.std(head_weight)),
        "head_weight_max_abs": float(np.max(np.abs(head_weight))),
        "head_bias_mean": float(np.mean(head_bias)),
        "head_bias_std": float(np.std(head_bias)),
        "lambda": lambda_reg,
        "num_train": int(N),
    }

    return {"weights": w, "diagnostics": diagnostics}
