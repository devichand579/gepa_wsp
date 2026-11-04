import numpy as np
import math

def edit(w, f):
    # Extract current conv weights/biases
    conv1_w = w["conv1.weight"]
    conv1_b = w["conv1.bias"]
    conv2_w = w["conv2.weight"]
    conv2_b = w["conv2.bias"]

    # Helper: conv2d with padding=1, stride=1
    def conv2d_batch(x, weight, bias, padding=1):
        N, C_in, H, W = x.shape
        C_out, _, kH, kW = weight.shape
        H_out = H + 2 * padding - kH + 1
        W_out = W + 2 * padding - kW + 1
        x_pad = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
        s = x_pad.strides
        shape = (N, C_in, kH, kW, H_out, W_out)
        strides = (s[0], s[1], s[2], s[3], s[2], s[3])
        x_col = np.lib.stride_tricks.as_strided(x_pad, shape=shape, strides=strides)
        x_col = x_col.reshape(N, C_in * kH * kW, H_out * W_out)
        w_col = weight.reshape(C_out, -1)
        out = np.einsum('ok,nkl->nol', w_col, x_col)
        out = out + bias.reshape(1, -1, 1)
        out = out.reshape(N, C_out, H_out, W_out)
        return out

    def maxpool2d(x):
        N, C, H, W = x.shape
        x = x.reshape(N, C, H // 2, 2, W // 2, 2)
        x = x.max(axis=(3, 5))
        return x

    def compute_features(images):
        # images: [N,1,28,28]
        x = conv2d_batch(images, conv1_w, conv1_b, padding=1)
        x = np.maximum(x, 0)
        x = maxpool2d(x)
        x = conv2d_batch(x, conv2_w, conv2_b, padding=1)
        x = np.maximum(x, 0)
        x = maxpool2d(x)
        N = x.shape[0]
        return x.reshape(N, -1)

    # Generate augmented images
    X_orig = f["x"]  # [2000,1,28,28]
    y_orig = f["y"]  # [2000]
    N = X_orig.shape[0]

    def shift_image(img, dx, dy):
        # img: [1,28,28]
        # dx, dy: shift amounts (positive right/down)
        # pad with zeros
        pad_x = max(0, dx)
        pad_y = max(0, dy)
        crop_x = max(0, -dx)
        crop_y = max(0, -dy)
        # We want to shift the image by (dx, dy). 
        # Equivalent to padding on one side and cropping on the other.
        # For dx > 0: pad left with dx, crop right by dx.
        # For dx < 0: pad right with -dx, crop left by -dx.
        # Let's do it simply: create a zero canvas and place the image at the offset.
        canvas = np.zeros_like(img)
        # source region
        src_x0 = max(0, -dx)
        src_x1 = min(28, 28 - dx)
        src_y0 = max(0, -dy)
        src_y1 = min(28, 28 - dy)
        dst_x0 = max(0, dx)
        dst_x1 = min(28, 28 + dx)
        dst_y0 = max(0, dy)
        dst_y1 = min(28, 28 + dy)
        if src_x0 < src_x1 and src_y0 < src_y1:
            canvas[:, dst_y0:dst_y1, dst_x0:dst_x1] = img[:, src_y0:src_y1, src_x0:src_x1]
        return canvas

    # Augmentations: original, flip, and shifts
    aug_images = []
    aug_labels = []
    # original
    aug_images.append(X_orig)
    aug_labels.append(y_orig)
    # horizontal flip
    aug_images.append(X_orig[:, :, :, ::-1])
    aug_labels.append(y_orig)
    # shifts: (-1,0), (1,0), (0,-1), (0,1) without flip
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dx, dy in shifts:
        shifted = np.zeros_like(X_orig)
        for i in range(N):
            shifted[i] = shift_image(X_orig[i], dx, dy)
        aug_images.append(shifted)
        aug_labels.append(y_orig)
    # shifts with flip
    for dx, dy in shifts:
        shifted = np.zeros_like(X_orig)
        for i in range(N):
            shifted[i] = shift_image(X_orig[i], dx, dy)
        aug_images.append(shifted[:, :, :, ::-1])
        aug_labels.append(y_orig)

    X_aug = np.concatenate(aug_images, axis=0)
    y_aug = np.concatenate(aug_labels, axis=0)
    N_aug = X_aug.shape[0]

    # Compute features in batches
    batch_size = 128
    features_list = []
    for i in range(0, N_aug, batch_size):
        batch = X_aug[i:i+batch_size]
        feats = compute_features(batch)
        features_list.append(feats)
    X_feat = np.concatenate(features_list, axis=0)  # [N_aug, 1568]

    # Prepare labels as one-hot
    num_classes = 10
    Y_onehot = np.zeros((N_aug, num_classes), dtype=np.float32)
    Y_onehot[np.arange(N_aug), y_aug] = 1.0

    # Center features and labels for ridge with bias
    X_mean = X_feat.mean(axis=0, keepdims=True)
    Y_mean = Y_onehot.mean(axis=0, keepdims=True)
    X_c = X_feat - X_mean
    Y_c = Y_onehot - Y_mean

    # Precompute full X^T X and X^T Y
    XTX = X_c.T @ X_c
    XTY = X_c.T @ Y_c

    # 5-fold cross-validation to select lambda
    n_folds = 5
    indices = np.arange(N_aug)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    lambdas = np.logspace(-3, 4, 20)  # 0.001 to 10000
    best_lambda = None
    best_acc = -1.0

    for lam in lambdas:
        acc_sum = 0.0
        for val_idx in folds:
            train_idx = np.setdiff1d(indices, val_idx, assume_unique=False)
            # Compute XTX_train and XTY_train by subtracting validation contributions
            X_val = X_c[val_idx]
            Y_val = Y_c[val_idx]
            XTX_val = X_val.T @ X_val
            XTY_val = X_val.T @ Y_val
            XTX_train = XTX - XTX_val
            XTY_train = XTY - XTY_val
            # Solve ridge
            A = XTX_train + lam * np.eye(XTX_train.shape[0], dtype=np.float32)
            W = np.linalg.solve(A, XTY_train)
            # Predict on validation
            pred_val = X_val @ W + Y_mean  # add back mean
            pred_labels = np.argmax(pred_val, axis=1)
            true_labels = y_aug[val_idx]
            acc = np.mean(pred_labels == true_labels)
            acc_sum += acc
        avg_acc = acc_sum / n_folds
        if avg_acc > best_acc:
            best_acc = avg_acc
            best_lambda = lam

    # Train on full augmented set with best lambda
    A = XTX + best_lambda * np.eye(XTX.shape[0], dtype=np.float32)
    W = np.linalg.solve(A, XTY)
    bias = Y_mean.flatten() - (X_mean @ W).flatten()

    # Update head weights and bias
    w["head.weight"] = W.T.astype(np.float32)  # shape [10, 1568]
    w["head.bias"] = bias.astype(np.float32)   # shape [10]

    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_acc),
        "head_weight_norm": float(np.linalg.norm(w["head.weight"])),
        "head_bias_norm": float(np.linalg.norm(w["head.bias"])),
        "head_weight_max_abs": float(np.max(np.abs(w["head.weight"]))),
        "head_bias_max_abs": float(np.max(np.abs(w["head.bias"]))),
        "num_augmented": int(N_aug),
    }

    return {"weights": w, "diagnostics": diagnostics}
