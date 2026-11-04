import numpy as np
import math

def edit(w, f):
    # Extract data
    x = f["x"].astype(np.float32)          # (2000,1,28,28)
    y = f["y"].astype(np.int64)            # (2000,)
    N = x.shape[0]
    C = 10

    # Helper: im2col for 3x3 conv with padding=1, stride=1
    def im2col_pad1(x, kh=3, kw=3):
        # x: (N,C,H,W)
        N, C, H, W = x.shape
        pad = 1
        xp = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
        # sliding window view
        # shape: (N, C, H, W, kh, kw)
        s = xp.strides
        patches = np.lib.stride_tricks.as_strided(
            xp,
            shape=(N, C, H, W, kh, kw),
            strides=(s[0], s[1], s[2], s[3], s[2], s[3])
        )
        return patches

    # Helper: maxpool 2x2 stride 2
    def maxpool2(x):
        # x: (N,C,H,W)
        N, C, H, W = x.shape
        x = x.reshape(N, C, H//2, 2, W//2, 2)
        return x.max(axis=(3,5))

    # 1. PCA for conv1 filters
    # Extract 3x3 patches from original images (no padding for PCA)
    patches1 = []
    for i in range(N):
        img = x[i,0]  # (28,28)
        for r in range(26):
            for c in range(26):
                patches1.append(img[r:r+3, c:c+3].ravel())
    patches1 = np.array(patches1, dtype=np.float32)  # (N*26*26, 9)
    # Subsample if too many
    if patches1.shape[0] > 200000:
        idx = np.random.choice(patches1.shape[0], 200000, replace=False)
        patches1 = patches1[idx]
    mean1 = patches1.mean(axis=0, keepdims=True)
    patches1_centered = patches1 - mean1
    cov1 = patches1_centered.T @ patches1_centered / (patches1.shape[0]-1)
    eigvals1, eigvecs1 = np.linalg.eigh(cov1)
    # Sort descending
    idx1 = np.argsort(eigvals1)[::-1]
    eigvecs1 = eigvecs1[:, idx1]
    # Take top 16
    W1 = eigvecs1[:, :16].T  # (16,9)
    W1 = W1.reshape(16,1,3,3).astype(np.float32)
    b1 = np.zeros(16, dtype=np.float32)

    # 2. Compute conv1 output
    patches1_conv = im2col_pad1(x)  # (N,1,28,28,3,3)
    patches1_conv = patches1_conv.reshape(N*28*28, 9)
    conv1_out = patches1_conv @ W1.reshape(16,9).T + b1  # (N*28*28,16)
    conv1_out = conv1_out.reshape(N,28,28,16).transpose(0,3,1,2)  # (N,16,28,28)
    conv1_out = np.maximum(conv1_out, 0)  # ReLU
    conv1_out = maxpool2(conv1_out)  # (N,16,14,14)

    # 3. PCA for conv2 filters
    # Extract 3x3 patches across spatial dims from conv1_out
    patches2 = []
    for i in range(N):
        fm = conv1_out[i]  # (16,14,14)
        for r in range(12):
            for c in range(12):
                patches2.append(fm[:, r:r+3, c:c+3].ravel())
    patches2 = np.array(patches2, dtype=np.float32)  # (N*12*12, 144)
    if patches2.shape[0] > 100000:
        idx = np.random.choice(patches2.shape[0], 100000, replace=False)
        patches2 = patches2[idx]
    mean2 = patches2.mean(axis=0, keepdims=True)
    patches2_centered = patches2 - mean2
    cov2 = patches2_centered.T @ patches2_centered / (patches2.shape[0]-1)
    eigvals2, eigvecs2 = np.linalg.eigh(cov2)
    idx2 = np.argsort(eigvals2)[::-1]
    eigvecs2 = eigvecs2[:, idx2]
    W2 = eigvecs2[:, :32].T  # (32,144)
    W2 = W2.reshape(32,16,3,3).astype(np.float32)
    b2 = np.zeros(32, dtype=np.float32)

    # 4. Compute conv2 output
    patches2_conv = im2col_pad1(conv1_out)  # (N,16,14,14,3,3)
    patches2_conv = patches2_conv.reshape(N*14*14, 144)
    conv2_out = patches2_conv @ W2.reshape(32,144).T + b2  # (N*14*14,32)
    conv2_out = conv2_out.reshape(N,14,14,32).transpose(0,3,1,2)  # (N,32,14,14)
    conv2_out = np.maximum(conv2_out, 0)
    conv2_out = maxpool2(conv2_out)  # (N,32,7,7)
    features = conv2_out.reshape(N, -1)  # (N,1568)

    # 5. Train head with ridge regression + CV
    # Standardize features
    feat_mean = features.mean(axis=0, keepdims=True)
    feat_std = features.std(axis=0, keepdims=True) + 1e-6
    X = (features - feat_mean) / feat_std  # (N,1568)
    X_aug = np.hstack([X, np.ones((N,1), dtype=np.float32)])  # (N,1569)
    Y = np.zeros((N, C), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    # 5-fold CV to choose lambda
    lambdas = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
    best_lambda = 1e-2
    best_acc = 0.0
    indices = np.arange(N)
    np.random.shuffle(indices)
    folds = np.array_split(indices, 5)
    for lam in lambdas:
        accs = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            X_tr = X_aug[train_idx]
            Y_tr = Y[train_idx]
            X_val = X_aug[val_idx]
            Y_val = Y[val_idx]
            # Ridge closed-form: W = (X^T X + lam I)^{-1} X^T Y
            # Use lstsq for stability
            A = X_tr.T @ X_tr + lam * np.eye(X_tr.shape[1], dtype=np.float32)
            B = X_tr.T @ Y_tr
            W_ridge = np.linalg.solve(A, B)  # (1569,10)
            logits_val = X_val @ W_ridge
            preds = np.argmax(logits_val, axis=1)
            acc = np.mean(preds == y[val_idx])
            accs.append(acc)
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lambda = lam

    # Train on full data with best lambda
    A = X_aug.T @ X_aug + best_lambda * np.eye(X_aug.shape[1], dtype=np.float32)
    B = X_aug.T @ Y
    W_full = np.linalg.solve(A, B)  # (1569,10)

    # Convert back to original feature space
    W_std = W_full[:1568, :]  # (1568,10)
    b_std = W_full[1568, :]   # (10,)
    # logits = X_std @ W_std + b_std = ((X - mean)/std) @ W_std + b_std
    #        = X @ (W_std/std).T + (b_std - mean @ (W_std/std).T)
    head_weight = (W_std / feat_std.T).T  # (10,1568)
    head_bias = b_std - (feat_mean @ head_weight.T).ravel()  # (10,)

    # 6. Update weights
    w["conv1.weight"] = W1.astype(np.float32)
    w["conv1.bias"] = b1.astype(np.float32)
    w["conv2.weight"] = W2.astype(np.float32)
    w["conv2.bias"] = b2.astype(np.float32)
    w["head.weight"] = head_weight.astype(np.float32)
    w["head.bias"] = head_bias.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_acc),
        "train_accuracy": float(np.mean(np.argmax(X_aug @ W_full, axis=1) == y)),
        "head_weight_mean": float(w["head.weight"].mean()),
        "head_weight_std": float(w["head.weight"].std()),
        "head_bias": w["head.bias"].tolist(),
        "conv1_weight_std": float(w["conv1.weight"].std()),
        "conv2_weight_std": float(w["conv2.weight"].std()),
    }

    return {"weights": w, "diagnostics": diagnostics}
