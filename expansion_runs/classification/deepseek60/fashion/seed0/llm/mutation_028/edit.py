import numpy as np

def edit(w, f):
    # Extract feedback data
    x = f["x"].astype(np.float32)  # (2000,1,28,28)
    y = f["y"].astype(np.int64)    # (2000,)
    N = x.shape[0]

    # Helper: conv2d with padding
    def conv2d(x, weight, bias, padding=1):
        N, C, H, W = x.shape
        O, C2, K, _ = weight.shape
        if padding > 0:
            x_pad = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant')
        else:
            x_pad = x
        H_out = H + 2*padding - K + 1
        W_out = W + 2*padding - K + 1
        windows = np.lib.stride_tricks.sliding_window_view(x_pad, (K, K), axis=(2,3))  # (N,C,H_out,W_out,K,K)
        windows = windows.transpose(0, 1, 4, 5, 2, 3)  # (N,C,K,K,H_out,W_out)
        windows = windows.reshape(N, C*K*K, H_out, W_out)
        w_flat = weight.reshape(O, C*K*K)
        out = np.einsum('ok,nkhw->nohw', w_flat, windows) + bias.reshape(1, O, 1, 1)
        return out

    # Helper: maxpool2d stride 2
    def maxpool2d(x, kernel=2, stride=2):
        N, C, H, W = x.shape
        windows = np.lib.stride_tricks.sliding_window_view(x, (kernel, kernel), axis=(2,3))
        windows = windows[:, :, ::stride, ::stride, :, :]
        return windows.max(axis=(4,5))

    # 1. PCA on input patches for conv1
    x_pad = np.pad(x, ((0,0),(0,0),(1,1),(1,1)), mode='constant')
    patches1 = np.lib.stride_tricks.sliding_window_view(x_pad, (3,3), axis=(2,3))  # (N,1,28,28,3,3)
    patches1 = patches1.reshape(-1, 9)  # (N*28*28, 9)
    mean1 = patches1.mean(axis=0)
    patches1_centered = patches1 - mean1
    cov1 = patches1_centered.T @ patches1_centered / (patches1.shape[0] - 1)
    eigvals1, eigvecs1 = np.linalg.eigh(cov1)
    idx1 = np.argsort(eigvals1)[::-1]
    comp1 = eigvecs1[:, idx1[:16]]  # (9,16)
    conv1_w = comp1.T.reshape(16, 1, 3, 3).astype(np.float32)
    conv1_b = (-comp1.T @ mean1).astype(np.float32)

    # 2. Compute conv1 output and pool
    h1 = conv2d(x, conv1_w, conv1_b, padding=1)
    h1 = np.maximum(h1, 0)
    h1_pool = maxpool2d(h1, 2, 2)  # (N,16,14,14)

    # 3. PCA on conv1 pooled patches for conv2
    h1_pad = np.pad(h1_pool, ((0,0),(0,0),(1,1),(1,1)), mode='constant')
    patches2 = np.lib.stride_tricks.sliding_window_view(h1_pad, (3,3), axis=(2,3))  # (N,16,14,14,3,3)
    patches2 = patches2.transpose(0,2,3,1,4,5).reshape(-1, 16*9)  # (N*14*14, 144)
    mean2 = patches2.mean(axis=0)
    patches2_centered = patches2 - mean2
    cov2 = patches2_centered.T @ patches2_centered / (patches2.shape[0] - 1)
    eigvals2, eigvecs2 = np.linalg.eigh(cov2)
    idx2 = np.argsort(eigvals2)[::-1]
    comp2 = eigvecs2[:, idx2[:32]]  # (144,32)
    conv2_w = comp2.T.reshape(32, 16, 3, 3).astype(np.float32)
    conv2_b = (-comp2.T @ mean2).astype(np.float32)

    # 4. Compute features for all x and flipped x
    def compute_features(x_in):
        h = conv2d(x_in, conv1_w, conv1_b, padding=1)
        h = np.maximum(h, 0)
        h = maxpool2d(h, 2, 2)
        h = conv2d(h, conv2_w, conv2_b, padding=1)
        h = np.maximum(h, 0)
        h = maxpool2d(h, 2, 2)
        return h.reshape(x_in.shape[0], -1)

    F = compute_features(x)
    F_flip = compute_features(x[:, :, :, ::-1])
    X = np.vstack([F, F_flip]).astype(np.float32)  # (4000,1568)
    Y = np.zeros((2*N, 10), dtype=np.float32)
    Y[np.arange(N), y] = 1.0
    Y[N + np.arange(N), y] = 1.0

    # 5. Ridge regression with validation to choose lambda
    rng = np.random.RandomState(42)
    perm = rng.permutation(2*N)
    val_size = 1000
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    best_lam = 1000.0
    best_acc = -1.0
    for lam in [1.0, 10.0, 100.0, 1000.0, 10000.0]:
        mean_X = X_train.mean(axis=0)
        mean_Y = Y_train.mean(axis=0)
        Xc = X_train - mean_X
        Yc = Y_train - mean_Y
        D = Xc.shape[1]
        W = np.linalg.solve(Xc.T @ Xc + lam * np.eye(D, dtype=np.float32), Xc.T @ Yc)
        b = mean_Y - mean_X @ W
        logits_val = X_val @ W + b
        pred = np.argmax(logits_val, axis=1)
        acc = np.mean(pred == np.argmax(Y_val, axis=1))
        if acc > best_acc:
            best_acc = acc
            best_lam = lam

    # Retrain on all data with best lambda
    mean_X = X.mean(axis=0)
    mean_Y = Y.mean(axis=0)
    Xc = X - mean_X
    Yc = Y - mean_Y
    D = Xc.shape[1]
    W = np.linalg.solve(Xc.T @ Xc + best_lam * np.eye(D, dtype=np.float32), Xc.T @ Yc)
    b = mean_Y - mean_X @ W

    head_w = W.T.astype(np.float32)  # (10,1568)
    head_b = b.astype(np.float32)    # (10,)

    # 6. Write new weights into w
    w["conv1.weight"] = conv1_w
    w["conv1.bias"] = conv1_b
    w["conv2.weight"] = conv2_w
    w["conv2.bias"] = conv2_b
    w["head.weight"] = head_w
    w["head.bias"] = head_b

    # Ensure finite and within bounds
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)

    diagnostics = {
        "best_lambda": float(best_lam),
        "val_accuracy": float(best_acc),
        "head_weight_std": float(np.std(head_w)),
        "head_bias_mean": float(np.mean(head_b)),
        "conv1_weight_std": float(np.std(conv1_w)),
        "conv2_weight_std": float(np.std(conv2_w)),
    }
    return {"weights": w, "diagnostics": diagnostics}
