import numpy as np

def edit(w, f):
    # Extract training data
    x = f["x"]  # (2000,1,28,28)
    y = f["y"]  # (2000,)
    N = x.shape[0]
    
    # Helper: conv2d with padding=1, stride=1
    def conv2d(x, weight, bias, padding=1):
        # x: (N, C_in, H, W)
        # weight: (C_out, C_in, 3, 3)
        # bias: (C_out,)
        N, C_in, H, W = x.shape
        C_out = weight.shape[0]
        # Pad input
        x_pad = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant')
        # Extract patches: (N, H, W, C_in, 3, 3)
        patches = np.zeros((N, H, W, C_in, 3, 3), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                patches[:, :, :, :, i, j] = x_pad[:, :, i:i+H, j:j+W]
        # Reshape to (N*H*W, C_in*9)
        patches = patches.reshape(N*H*W, C_in*9)
        # weight reshape to (C_out, C_in*9)
        w_flat = weight.reshape(C_out, C_in*9)
        # Compute output
        out = patches @ w_flat.T  # (N*H*W, C_out)
        out = out.reshape(N, H, W, C_out)
        out = out.transpose(0, 3, 1, 2)  # (N, C_out, H, W)
        out += bias.reshape(1, C_out, 1, 1)
        return out

    # Helper: max_pool2d with kernel=2, stride=2
    def max_pool2d(x, kernel=2, stride=2):
        N, C, H, W = x.shape
        H_out = H // kernel
        W_out = W // kernel
        x = x.reshape(N, C, H_out, kernel, W_out, kernel)
        return x.max(axis=(3,5))

    # Helper: relu
    def relu(x):
        return np.maximum(x, 0)

    # --- Step 1: Learn conv1 filters via PCA on 3x3 patches ---
    # Extract 3x3 patches from all images (no padding)
    patches1 = []
    for n in range(N):
        img = x[n, 0]  # (28,28)
        for i in range(26):
            for j in range(26):
                patches1.append(img[i:i+3, j:j+3].flatten())
    patches1 = np.array(patches1, dtype=np.float32)  # (N*26*26, 9)
    # Subtract mean
    mean1 = patches1.mean(axis=0)
    patches1_centered = patches1 - mean1
    # Compute covariance (9x9)
    cov1 = patches1_centered.T @ patches1_centered / patches1_centered.shape[0]
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov1)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    # Take top 16
    filters1 = eigvecs[:, :16].T  # (16, 9)
    # Reshape to (16,1,3,3)
    conv1_weight = filters1.reshape(16, 1, 3, 3).astype(np.float32)
    conv1_bias = np.zeros(16, dtype=np.float32)

    # --- Step 2: Compute conv1 output on all images ---
    h1 = relu(conv2d(x, conv1_weight, conv1_bias, padding=1))
    h1 = max_pool2d(h1, 2, 2)  # (N,16,14,14)

    # --- Step 3: Learn conv2 filters via PCA on 3x3 patches from h1 ---
    # Extract 3x3 patches from h1 (16 channels)
    patches2 = []
    for n in range(N):
        for i in range(12):  # 14-3+1=12
            for j in range(12):
                patch = h1[n, :, i:i+3, j:j+3]  # (16,3,3)
                patches2.append(patch.flatten())
    patches2 = np.array(patches2, dtype=np.float32)  # (N*12*12, 144)
    mean2 = patches2.mean(axis=0)
    patches2_centered = patches2 - mean2
    # Compute covariance (144x144) - might be large but doable
    # To save memory, we can use SVD on centered data
    # patches2_centered: (M, 144). M = 2000*144 = 288000. 
    # We can compute covariance as (144x144) by matrix multiply.
    cov2 = patches2_centered.T @ patches2_centered / patches2_centered.shape[0]
    eigvals2, eigvecs2 = np.linalg.eigh(cov2)
    idx2 = np.argsort(eigvals2)[::-1]
    eigvecs2 = eigvecs2[:, idx2]
    filters2 = eigvecs2[:, :32].T  # (32, 144)
    conv2_weight = filters2.reshape(32, 16, 3, 3).astype(np.float32)
    conv2_bias = np.zeros(32, dtype=np.float32)

    # --- Step 4: Compute final features for training data ---
    h2 = relu(conv2d(h1, conv2_weight, conv2_bias, padding=1))
    h2 = max_pool2d(h2, 2, 2)  # (N,32,7,7)
    features = h2.reshape(N, -1)  # (N, 1568)

    # --- Step 5: Augment with horizontal flips ---
    x_flip = x[:, :, :, ::-1]  # (N,1,28,28)
    h1_flip = relu(conv2d(x_flip, conv1_weight, conv1_bias, padding=1))
    h1_flip = max_pool2d(h1_flip, 2, 2)
    h2_flip = relu(conv2d(h1_flip, conv2_weight, conv2_bias, padding=1))
    h2_flip = max_pool2d(h2_flip, 2, 2)
    features_flip = h2_flip.reshape(N, -1)

    X_train = np.vstack([features, features_flip])  # (2N, 1568)
    y_train = np.concatenate([y, y])  # (2N,)

    # One-hot encode labels
    Y_train = np.zeros((2*N, 10), dtype=np.float32)
    Y_train[np.arange(2*N), y_train] = 1.0

    # --- Step 6: Ridge regression with cross-validation to choose lambda ---
    # Center features and labels? Ridge with intercept: we can augment with bias term.
    # We'll use standard ridge: W = (X^T X + lambda I)^-1 X^T Y
    # Add bias term by augmenting X with a column of ones.
    X_aug = np.hstack([X_train, np.ones((2*N, 1), dtype=np.float32)])
    # We'll do 5-fold CV to select lambda
    lambdas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    best_lambda = None
    best_acc = -1
    # Shuffle data for CV
    rng = np.random.RandomState(42)
    indices = rng.permutation(2*N)
    X_aug = X_aug[indices]
    Y_train = Y_train[indices]
    fold_size = 2*N // 5
    for lam in lambdas:
        accs = []
        for fold in range(5):
            val_start = fold * fold_size
            val_end = (fold+1) * fold_size if fold < 4 else 2*N
            val_idx = np.arange(val_start, val_end)
            train_idx = np.setdiff1d(np.arange(2*N), val_idx)
            X_tr = X_aug[train_idx]
            Y_tr = Y_train[train_idx]
            X_val = X_aug[val_idx]
            Y_val = Y_train[val_idx]
            # Solve ridge: W = (X^T X + lam I)^-1 X^T Y
            # Use normal equations
            XtX = X_tr.T @ X_tr
            XtY = X_tr.T @ Y_tr
            # Add regularization (do not regularize bias? Usually regularize all)
            reg = lam * np.eye(XtX.shape[0], dtype=np.float32)
            W = np.linalg.solve(XtX + reg, XtY)
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

    # Train final model on all augmented data with best_lambda
    XtX = X_aug.T @ X_aug
    XtY = X_aug.T @ Y_train
    reg = best_lambda * np.eye(XtX.shape[0], dtype=np.float32)
    W_full = np.linalg.solve(XtX + reg, XtY)  # (1569, 10)

    # Extract head weights and bias
    head_weight = W_full[:-1, :].T  # (10, 1568)
    head_bias = W_full[-1, :]  # (10,)

    # Update w
    w["conv1.weight"] = conv1_weight
    w["conv1.bias"] = conv1_bias
    w["conv2.weight"] = conv2_weight
    w["conv2.bias"] = conv2_bias
    w["head.weight"] = head_weight.astype(np.float32)
    w["head.bias"] = head_bias.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_acc),
        "head_weight_mean": float(np.mean(head_weight)),
        "head_weight_std": float(np.std(head_weight)),
        "head_weight_max_abs": float(np.max(np.abs(head_weight))),
        "head_bias_mean": float(np.mean(head_bias)),
        "head_bias_std": float(np.std(head_bias)),
        "num_train": int(2*N)
    }
    return {"weights": w, "diagnostics": diagnostics}
