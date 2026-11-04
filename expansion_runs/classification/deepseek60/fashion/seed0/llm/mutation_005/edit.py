import numpy as np

def edit(w, f):
    # Helper functions
    def conv2d_pad1(x, weight, bias):
        # x: (N, C_in, H, W), weight: (C_out, C_in, 3, 3), bias: (C_out,)
        N, C_in, H, W = x.shape
        C_out = weight.shape[0]
        x_pad = np.pad(x, ((0,0),(0,0),(1,1),(1,1)), mode='constant')
        # Extract patches: (N, H, W, C_in*3*3)
        patches = np.zeros((N, H, W, C_in*3*3), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                patches[:, :, :, i*C_in*3 + j*C_in : i*C_in*3 + (j+1)*C_in] = x_pad[:, :, i:i+H, j:j+W].transpose(0,2,3,1)
        # Reshape weight: (C_out, C_in*3*3)
        w_flat = weight.reshape(C_out, -1)
        # Compute output: (N, H, W, C_out)
        out = patches @ w_flat.T + bias.reshape(1,1,1,-1)
        # Transpose to (N, C_out, H, W)
        return out.transpose(0,3,1,2)

    def relu(x):
        return np.maximum(x, 0)

    def maxpool2d(x):
        # x: (N, C, H, W) -> (N, C, H//2, W//2)
        N, C, H, W = x.shape
        x = x.reshape(N, C, H//2, 2, W//2, 2)
        return x.max(axis=(3,5))

    def extract_patches_3x3(x):
        # x: (N, C, H, W) -> (N*H*W, C*9)
        N, C, H, W = x.shape
        patches = np.zeros((N, H, W, C, 3, 3), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                patches[:, :, :, :, i, j] = x[:, :, i:i+H, j:j+W].transpose(0,2,3,1)
        return patches.reshape(N*H*W, C*9)

    # 1. Compute PCA filters for conv1
    X = f["x"]  # (2000,1,28,28)
    N = X.shape[0]
    # Extract 3x3 patches from input
    patches1 = extract_patches_3x3(X)  # (2000*28*28, 9)
    mean1 = patches1.mean(axis=0)
    patches1_centered = patches1 - mean1
    cov1 = patches1_centered.T @ patches1_centered / patches1.shape[0]
    eigvals1, eigvecs1 = np.linalg.eigh(cov1)
    # Sort descending
    idx1 = np.argsort(eigvals1)[::-1]
    eigvecs1 = eigvecs1[:, idx1]
    # Take top 16
    filters1 = eigvecs1[:, :16].T  # (16, 9)
    conv1_weight = filters1.reshape(16, 1, 3, 3).astype(np.float32)
    conv1_bias = - (mean1 @ filters1.T).astype(np.float32)  # (16,)

    # 2. Compute conv1 output on all data
    conv1_out = conv2d_pad1(X, conv1_weight, conv1_bias)
    conv1_relu = relu(conv1_out)
    conv1_pool = maxpool2d(conv1_relu)  # (2000,16,14,14)

    # 3. Compute PCA filters for conv2
    patches2 = extract_patches_3x3(conv1_pool)  # (2000*14*14, 16*9=144)
    mean2 = patches2.mean(axis=0)
    patches2_centered = patches2 - mean2
    cov2 = patches2_centered.T @ patches2_centered / patches2.shape[0]
    eigvals2, eigvecs2 = np.linalg.eigh(cov2)
    idx2 = np.argsort(eigvals2)[::-1]
    eigvecs2 = eigvecs2[:, idx2]
    filters2 = eigvecs2[:, :32].T  # (32, 144)
    conv2_weight = filters2.reshape(32, 16, 3, 3).astype(np.float32)
    conv2_bias = - (mean2 @ filters2.T).astype(np.float32)  # (32,)

    # 4. Compute final features
    conv2_out = conv2d_pad1(conv1_pool, conv2_weight, conv2_bias)
    conv2_relu = relu(conv2_out)
    conv2_pool = maxpool2d(conv2_relu)  # (2000,32,7,7)
    features = conv2_pool.reshape(N, -1)  # (2000, 1568)

    # 5. Fit head with ridge regression using cross-validation to select lambda
    y = f["y"]
    Y = np.zeros((N, 10), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    # Augment features with bias
    X_aug = np.concatenate([features, np.ones((N, 1), dtype=np.float32)], axis=1)  # (2000, 1569)

    # 5-fold cross-validation to select lambda
    lambdas = [0.001, 0.01, 0.1, 1.0, 10.0]
    best_lambda = 0.001
    best_acc = 0.0
    n_folds = 5
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = N // n_folds

    for lam in lambdas:
        accs = []
        for fold in range(n_folds):
            val_idx = indices[fold*fold_size:(fold+1)*fold_size]
            train_idx = np.setdiff1d(indices, val_idx)
            X_train = X_aug[train_idx]
            Y_train = Y[train_idx]
            X_val = X_aug[val_idx]
            Y_val = Y[val_idx]
            # Solve ridge: (X^T X + lam I) W = X^T Y
            A = X_train.T @ X_train + lam * np.eye(X_train.shape[1], dtype=np.float32)
            B = X_train.T @ Y_train
            W = np.linalg.solve(A, B)
            # Predict
            logits = X_val @ W
            preds = np.argmax(logits, axis=1)
            true = np.argmax(Y_val, axis=1)
            acc = np.mean(preds == true)
            accs.append(acc)
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lambda = lam

    # Fit on full data with best lambda
    A = X_aug.T @ X_aug + best_lambda * np.eye(X_aug.shape[1], dtype=np.float32)
    B = X_aug.T @ Y
    W_full = np.linalg.solve(A, B)  # (1569, 10)

    head_weight = W_full[:1568, :].T.astype(np.float32)  # (10, 1568)
    head_bias = W_full[1568, :].astype(np.float32)  # (10,)

    # 6. Update weights
    w["conv1.weight"] = conv1_weight
    w["conv1.bias"] = conv1_bias
    w["conv2.weight"] = conv2_weight
    w["conv2.bias"] = conv2_bias
    w["head.weight"] = head_weight
    w["head.bias"] = head_bias

    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_acc),
        "conv1_weight_norm": float(np.linalg.norm(conv1_weight)),
        "conv2_weight_norm": float(np.linalg.norm(conv2_weight)),
        "head_weight_norm": float(np.linalg.norm(head_weight)),
        "head_bias_norm": float(np.linalg.norm(head_bias)),
    }
    return {"weights": w, "diagnostics": diagnostics}
