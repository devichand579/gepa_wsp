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
    x = f["x"]          # (2000, 1, 28, 28)
    y = f["y"]          # (2000,)
    orig_feats = f["features"]  # (2000, 1568)

    N = x.shape[0]

    # Helper: compute features for a batch of images
    def compute_features(x_batch):
        # x_batch: (B, 1, 28, 28)
        B = x_batch.shape[0]
        # conv1
        pad1 = np.pad(x_batch, ((0,0),(0,0),(1,1),(1,1)), mode='constant')
        patches1 = np.lib.stride_tricks.sliding_window_view(pad1, (3,3), axis=(2,3))  # (B,1,28,28,3,3)
        patches1 = patches1.transpose(0,2,3,1,4,5).reshape(B, 28*28, -1)  # (B,784,9)
        w1 = conv1_w.reshape(16, -1)  # (16,9)
        out1 = patches1 @ w1.T + conv1_b  # (B,784,16)
        out1 = out1.transpose(0,2,1).reshape(B,16,28,28)
        out1 = np.maximum(out1, 0)
        h1 = out1.reshape(B,16,14,2,14,2).max(axis=(3,5))  # (B,16,14,14)
        # conv2
        pad2 = np.pad(h1, ((0,0),(0,0),(1,1),(1,1)), mode='constant')
        patches2 = np.lib.stride_tricks.sliding_window_view(pad2, (3,3), axis=(2,3))  # (B,16,14,14,3,3)
        patches2 = patches2.transpose(0,2,3,1,4,5).reshape(B, 14*14, -1)  # (B,196,144)
        w2 = conv2_w.reshape(32, -1)  # (32,144)
        out2 = patches2 @ w2.T + conv2_b  # (B,196,32)
        out2 = out2.transpose(0,2,1).reshape(B,32,14,14)
        out2 = np.maximum(out2, 0)
        h2 = out2.reshape(B,32,7,2,7,2).max(axis=(3,5))  # (B,32,7,7)
        feats = h2.reshape(B, -1)  # (B,1568)
        return feats

    # Generate augmented features in batches
    batch_size = 200
    aug_feats_list = []
    # We'll create 5 augmentations: flip, shift right, shift left, shift down, shift up
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        xb = x[start:end]  # (B,1,28,28)
        B = xb.shape[0]
        # flip
        x_flip = xb[:, :, :, ::-1]
        # shift right
        x_sr = np.zeros_like(xb)
        x_sr[:, :, :, 1:] = xb[:, :, :, :-1]
        # shift left
        x_sl = np.zeros_like(xb)
        x_sl[:, :, :, :-1] = xb[:, :, :, 1:]
        # shift down
        x_sd = np.zeros_like(xb)
        x_sd[:, :, 1:, :] = xb[:, :, :-1, :]
        # shift up
        x_su = np.zeros_like(xb)
        x_su[:, :, :-1, :] = xb[:, :, 1:, :]

        for x_aug in [x_flip, x_sr, x_sl, x_sd, x_su]:
            feats = compute_features(x_aug)
            aug_feats_list.append(feats)

    aug_feats = np.vstack(aug_feats_list)  # (5*N, 1568)
    X_all = np.vstack([orig_feats, aug_feats])  # (6*N, 1568)
    y_all = np.tile(y, 6)  # (6*N,)

    # Standardize features
    X_mean = X_all.mean(axis=0)
    X_std = X_all.std(axis=0) + 1e-6
    X_norm = (X_all - X_mean) / X_std

    # One-hot labels
    Y = np.eye(10)[y_all]  # (6*N, 10)

    # Ridge regression with bias not regularized
    N_all = X_norm.shape[0]
    X_aug = np.hstack([X_norm, np.ones((N_all, 1))])  # (N_all, 1569)
    lambda_ = 1.0  # regularization strength
    I = np.eye(X_aug.shape[1])
    I[-1, -1] = 0.0  # do not regularize bias
    W = np.linalg.solve(X_aug.T @ X_aug + lambda_ * I, X_aug.T @ Y)  # (1569, 10)
    W = W.T  # (10, 1569)
    head_w_new = W[:, :1568]  # (10, 1568)
    head_b_new = W[:, 1568]   # (10,)

    # Transform back to original feature space
    head_w_orig = head_w_new / X_std  # (10, 1568)
    head_b_orig = head_b_new - head_w_orig @ X_mean

    # Ensure float32 and within bounds
    head_w_orig = np.clip(head_w_orig, -1000, 1000).astype(np.float32)
    head_b_orig = np.clip(head_b_orig, -1000, 1000).astype(np.float32)

    # Update weights
    w["head.weight"] = head_w_orig
    w["head.bias"] = head_b_orig

    # Diagnostics
    diagnostics = {
        "head_weight_norm": float(np.linalg.norm(head_w_orig)),
        "head_bias_norm": float(np.linalg.norm(head_b_orig)),
        "head_weight_max_abs": float(np.max(np.abs(head_w_orig))),
        "head_bias_max_abs": float(np.max(np.abs(head_b_orig))),
        "lambda": lambda_,
        "augmented_samples": int(N_all),
    }
    return {"weights": w, "diagnostics": diagnostics}
