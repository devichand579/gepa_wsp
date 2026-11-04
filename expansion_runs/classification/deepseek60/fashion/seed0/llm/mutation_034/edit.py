import numpy as np
from numpy.lib.stride_tricks import as_strided

def edit(w, f):
    # ------------------------------------------------------------------
    # 0. Setup
    # ------------------------------------------------------------------
    X = f["x"].astype(np.float32)          # (2000, 1, 28, 28)
    y = f["y"].astype(np.int64)            # (2000,)
    N = X.shape[0]
    num_classes = 10

    # ------------------------------------------------------------------
    # 1. Learn conv1 filters via PCA on 3x3 patches (with padding=1)
    # ------------------------------------------------------------------
    X_pad = np.pad(X, ((0,0), (0,0), (1,1), (1,1)), mode='constant')
    sN, sC, sH, sW = X_pad.strides
    patches = as_strided(X_pad, shape=(N, 28, 28, 3, 3),
                         strides=(sN, sH, sW, sH, sW))
    patches_flat = patches.reshape(-1, 9)
    mean_patch = patches_flat.mean(axis=0)
    patches_centered = patches_flat - mean_patch
    cov = patches_centered.T @ patches_centered / patches_centered.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    filters1 = eigvecs[:, :16].T                     # (16, 9)
    conv1_weight = filters1.reshape(16, 1, 3, 3).astype(np.float32)
    conv1_bias = np.zeros(16, dtype=np.float32)

    # ------------------------------------------------------------------
    # 2. Helper functions for forward pass
    # ------------------------------------------------------------------
    def conv2d(x, weight, bias, padding=1, stride=1):
        N, C_in, H, W = x.shape
        C_out, _, K, _ = weight.shape
        x_pad = np.pad(x, ((0,0), (0,0), (padding, padding), (padding, padding)), mode='constant')
        H_pad, W_pad = x_pad.shape[2], x_pad.shape[3]
        H_out = (H_pad - K) // stride + 1
        W_out = (W_pad - K) // stride + 1
        sN, sC, sH, sW = x_pad.strides
        patches = as_strided(x_pad, shape=(N, C_in, K, K, H_out, W_out),
                             strides=(sN, sC, sH, sW, sH*stride, sW*stride))
        patches = patches.transpose(0, 4, 5, 1, 2, 3).reshape(N, H_out, W_out, C_in*K*K)
        weight_flat = weight.reshape(C_out, -1)
        out = patches @ weight_flat.T + bias.reshape(1, 1, 1, C_out)
        out = out.transpose(0, 3, 1, 2)
        return out

    def maxpool2d(x, kernel_size=2, stride=2):
        N, C, H, W = x.shape
        H_out = (H - kernel_size) // stride + 1
        W_out = (W - kernel_size) // stride + 1
        sN, sC, sH, sW = x.strides
        patches = as_strided(x, shape=(N, C, H_out, W_out, kernel_size, kernel_size),
                             strides=(sN, sC, sH*stride, sW*stride, sH, sW))
        return patches.max(axis=(4,5))

    # ------------------------------------------------------------------
    # 3. Compute conv1 features for all images
    # ------------------------------------------------------------------
    conv1_out = conv2d(X, conv1_weight, conv1_bias, padding=1, stride=1)
    conv1_relu = np.maximum(conv1_out, 0)
    conv1_pool = maxpool2d(conv1_relu, kernel_size=2, stride=2)  # (N, 16, 14, 14)

    # ------------------------------------------------------------------
    # 4. Learn conv2 filters via PCA on 3x3 patches from conv1_pool
    # ------------------------------------------------------------------
    conv1_pool_pad = np.pad(conv1_pool, ((0,0), (0,0), (1,1), (1,1)), mode='constant')
    sN, sC, sH, sW = conv1_pool_pad.strides
    patches2 = as_strided(conv1_pool_pad, shape=(N, 16, 3, 3, 14, 14),
                          strides=(sN, sC, sH, sW, sH, sW))
    # patches2: (N, 16, 3, 3, 14, 14) -> (N*14*14, 16*3*3)
    patches2_flat = patches2.transpose(0, 4, 5, 1, 2, 3).reshape(-1, 16*3*3)
    # Sample a subset to keep memory reasonable
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(patches2_flat.shape[0], size=min(100000, patches2_flat.shape[0]), replace=False)
    patches2_sample = patches2_flat[sample_idx]
    mean_patch2 = patches2_sample.mean(axis=0)
    patches2_centered = patches2_sample - mean_patch2
    cov2 = patches2_centered.T @ patches2_centered / patches2_centered.shape[0]
    eigvals2, eigvecs2 = np.linalg.eigh(cov2)
    idx2 = np.argsort(eigvals2)[::-1]
    eigvecs2 = eigvecs2[:, idx2]
    filters2 = eigvecs2[:, :32].T                     # (32, 144)
    conv2_weight = filters2.reshape(32, 16, 3, 3).astype(np.float32)
    conv2_bias = np.zeros(32, dtype=np.float32)

    # ------------------------------------------------------------------
    # 5. Compute final features for all images (and horizontal flips)
    # ------------------------------------------------------------------
    def compute_features(images):
        # images: (N, 1, 28, 28)
        c1 = conv2d(images, conv1_weight, conv1_bias, padding=1, stride=1)
        c1 = np.maximum(c1, 0)
        p1 = maxpool2d(c1, kernel_size=2, stride=2)
        c2 = conv2d(p1, conv2_weight, conv2_bias, padding=1, stride=1)
        c2 = np.maximum(c2, 0)
        p2 = maxpool2d(c2, kernel_size=2, stride=2)
        return p2.reshape(p2.shape[0], -1)  # (N, 1568)

    feats = compute_features(X)
    # Horizontal flips
    X_flip = X[:, :, :, ::-1].copy()
    feats_flip = compute_features(X_flip)
    # Combine
    X_train = np.vstack([feats, feats_flip])          # (4000, 1568)
    y_train = np.concatenate([y, y])                  # (4000,)

    # One-hot labels
    Y_train = np.zeros((X_train.shape[0], num_classes), dtype=np.float32)
    Y_train[np.arange(X_train.shape[0]), y_train] = 1.0

    # ------------------------------------------------------------------
    # 6. Fit ridge regression with cross-validation for lambda
    # ------------------------------------------------------------------
    # Center features? Ridge with intercept: we can add bias column or center.
    # We'll center X and Y, then fit weights, then compute bias.
    X_mean = X_train.mean(axis=0)
    Y_mean = Y_train.mean(axis=0)
    Xc = X_train - X_mean
    Yc = Y_train - Y_mean

    # Try a few lambdas
    lambdas = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
    best_lambda = None
    best_cv_acc = -1.0

    # 5-fold cross-validation
    n_folds = 5
    indices = np.arange(X_train.shape[0])
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    for lam in lambdas:
        cv_acc = 0.0
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            X_tr = Xc[train_idx]
            Y_tr = Yc[train_idx]
            X_val = Xc[val_idx]
            Y_val = Y_train[val_idx]
            # Solve (X^T X + lam I) W = X^T Y
            # Use np.linalg.solve for better stability
            A = X_tr.T @ X_tr + lam * np.eye(X_tr.shape[1])
            B = X_tr.T @ Y_tr
            W = np.linalg.solve(A, B)
            # Predict
            pred = X_val @ W + Y_mean
            pred_labels = np.argmax(pred, axis=1)
            true_labels = np.argmax(Y_val, axis=1)
            cv_acc += np.mean(pred_labels == true_labels)
        cv_acc /= n_folds
        if cv_acc > best_cv_acc:
            best_cv_acc = cv_acc
            best_lambda = lam

    # Fit final model on all data
    lam = best_lambda
    A = Xc.T @ Xc + lam * np.eye(Xc.shape[1])
    B = Xc.T @ Yc
    W = np.linalg.solve(A, B)
    head_weight = W.T.astype(np.float32)              # (10, 1568)
    head_bias = (Y_mean - X_mean @ W).astype(np.float32)  # (10,)

    # ------------------------------------------------------------------
    # 7. Update weights
    # ------------------------------------------------------------------
    w["conv1.weight"] = conv1_weight
    w["conv1.bias"] = conv1_bias
    w["conv2.weight"] = conv2_weight
    w["conv2.bias"] = conv2_bias
    w["head.weight"] = head_weight
    w["head.bias"] = head_bias

    # Ensure finite and within bounds
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)

    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_cv_acc),
        "head_weight_mean": float(head_weight.mean()),
        "head_weight_std": float(head_weight.std()),
        "head_weight_max_abs": float(np.abs(head_weight).max()),
        "head_bias_mean": float(head_bias.mean()),
        "head_bias_std": float(head_bias.std()),
        "num_train": int(X_train.shape[0]),
    }

    return {"weights": w, "diagnostics": diagnostics}
