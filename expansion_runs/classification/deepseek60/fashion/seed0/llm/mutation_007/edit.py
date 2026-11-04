import numpy as np

def edit(w, f):
    # Extract training data
    x = f["x"].astype(np.float32)          # (2000,1,28,28)
    y = f["y"].astype(np.int64)            # (2000,)
    N = x.shape[0]
    num_classes = 10

    # Helper: conv2d with padding
    def conv2d(x, weight, bias, padding=1):
        N, C_in, H, W = x.shape
        C_out, _, kH, kW = weight.shape
        if padding > 0:
            x_pad = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant')
        else:
            x_pad = x
        windows = np.lib.stride_tricks.sliding_window_view(x_pad, (kH, kW), axis=(2,3))
        # windows: (N, C_in, H_out, W_out, kH, kW)
        windows = windows.transpose(0, 2, 3, 1, 4, 5)
        patches = windows.reshape(N * windows.shape[1] * windows.shape[2], -1)
        w_flat = weight.reshape(C_out, -1)
        out = patches @ w_flat.T + bias
        out = out.reshape(N, windows.shape[1], windows.shape[2], C_out)
        out = out.transpose(0, 3, 1, 2)
        return out

    def maxpool2d(x, kernel_size=2, stride=2):
        N, C, H, W = x.shape
        H_out = H // kernel_size
        W_out = W // kernel_size
        x = x.reshape(N, C, H_out, kernel_size, W_out, kernel_size)
        return x.max(axis=(3,5))

    def relu(x):
        return np.maximum(x, 0)

    # --- 1. Learn conv1 filters via PCA on 3x3 patches ---
    # Extract 3x3 patches from original images (no padding)
    windows = np.lib.stride_tricks.sliding_window_view(x, (3,3), axis=(2,3))
    # windows: (N,1,26,26,3,3)
    patches = windows.transpose(0,2,3,1,4,5).reshape(-1, 9)
    mean1 = patches.mean(axis=0)
    patches_centered = patches - mean1
    cov1 = patches_centered.T @ patches_centered / patches.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov1)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    w1 = eigvecs[:, :16].T.reshape(16, 1, 3, 3).astype(np.float32)
    b1 = (-mean1 @ w1.reshape(16, 9).T).astype(np.float32)

    w["conv1.weight"] = w1
    w["conv1.bias"] = b1

    # --- 2. Compute conv1 features for all images ---
    h1 = conv2d(x, w1, b1, padding=1)
    h1 = relu(h1)
    h1 = maxpool2d(h1)          # (N,16,14,14)

    # --- 3. Learn conv2 filters via PCA on 3x3 patches from h1 ---
    windows2 = np.lib.stride_tricks.sliding_window_view(h1, (3,3), axis=(2,3))
    # windows2: (N,16,12,12,3,3)
    patches2 = windows2.transpose(0,2,3,1,4,5).reshape(-1, 16*3*3)
    mean2 = patches2.mean(axis=0)
    patches2_centered = patches2 - mean2
    cov2 = patches2_centered.T @ patches2_centered / patches2.shape[0]
    eigvals2, eigvecs2 = np.linalg.eigh(cov2)
    idx2 = np.argsort(eigvals2)[::-1]
    eigvecs2 = eigvecs2[:, idx2]
    w2 = eigvecs2[:, :32].T.reshape(32, 16, 3, 3).astype(np.float32)
    b2 = (-mean2 @ w2.reshape(32, 144).T).astype(np.float32)

    w["conv2.weight"] = w2
    w["conv2.bias"] = b2

    # --- 4. Compute final features ---
    h2 = conv2d(h1, w2, b2, padding=1)
    h2 = relu(h2)
    h2 = maxpool2d(h2)          # (N,32,7,7)
    F = h2.reshape(N, -1)       # (N,1568)

    # --- 5. Fit ridge-regularized linear head with CV ---
    # One-hot labels
    Y = np.zeros((N, num_classes), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    # Standardize features
    F_mean = F.mean(axis=0)
    F_std = F.std(axis=0) + 1e-6
    F_std = np.where(F_std < 1e-6, 1.0, F_std)
    F_norm = (F - F_mean) / F_std

    # Ridge with intercept: center Y and F
    Y_mean = Y.mean(axis=0)
    Y_centered = Y - Y_mean

    # Candidate lambdas
    lambdas = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    best_lambda = None
    best_acc = -1.0

    # 5-fold CV
    n_folds = 5
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = N // n_folds

    for lam in lambdas:
        accs = []
        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < n_folds - 1 else N
            val_idx = indices[val_start:val_end]
            train_idx = np.setdiff1d(indices, val_idx, assume_unique=False)

            F_train = F_norm[train_idx]
            Y_train = Y_centered[train_idx]
            F_val = F_norm[val_idx]
            Y_val = Y[val_idx]

            # Solve (F^T F + lam I) W = F^T Y
            A = F_train.T @ F_train + lam * np.eye(F_train.shape[1], dtype=np.float32)
            B = F_train.T @ Y_train
            W = np.linalg.solve(A, B)

            # Predict
            logits_val = F_val @ W + Y_mean
            preds = np.argmax(logits_val, axis=1)
            acc = np.mean(preds == y[val_idx])
            accs.append(acc)
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lambda = lam

    # Final training on all data with best lambda
    A = F_norm.T @ F_norm + best_lambda * np.eye(F_norm.shape[1], dtype=np.float32)
    B = F_norm.T @ Y_centered
    W = np.linalg.solve(A, B)

    # Convert back to original feature scale
    W_orig = W / F_std[:, None]
    bias = Y_mean - F_mean @ W_orig

    w["head.weight"] = W_orig.T.astype(np.float32)   # (10,1568)
    w["head.bias"] = bias.astype(np.float32)         # (10,)

    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_acc),
        "conv1_weight_norm": float(np.linalg.norm(w["conv1.weight"])),
        "conv2_weight_norm": float(np.linalg.norm(w["conv2.weight"])),
        "head_weight_norm": float(np.linalg.norm(w["head.weight"])),
        "head_bias_norm": float(np.linalg.norm(w["head.bias"])),
    }

    return {"weights": w, "diagnostics": diagnostics}
