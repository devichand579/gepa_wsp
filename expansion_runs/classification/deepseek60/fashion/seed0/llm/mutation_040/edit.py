import numpy as np

def edit(w, f):
    # Extract training data
    X = f["x"].astype(np.float32)  # (2000,1,28,28)
    y = f["y"].astype(np.int64)    # (2000,)
    N = X.shape[0]
    
    # Helper: conv2d with padding=1, stride=1
    def conv2d(x, weight, bias):
        # x: (N, C_in, H, W)
        # weight: (C_out, C_in, 3, 3)
        # bias: (C_out,)
        N, C_in, H, W = x.shape
        C_out = weight.shape[0]
        # Pad with 1
        x_pad = np.pad(x, ((0,0),(0,0),(1,1),(1,1)), mode='constant')
        # Extract patches: (N, C_in, H, W, 3, 3)
        patches = np.lib.stride_tricks.sliding_window_view(x_pad, (3,3), axis=(2,3))
        # patches shape: (N, C_in, H, W, 3, 3)
        # Reshape to (N*H*W, C_in*3*3)
        patches = patches.transpose(0,2,3,1,4,5).reshape(N*H*W, C_in*9)
        # weight reshape: (C_out, C_in*9)
        w_flat = weight.reshape(C_out, C_in*9)
        # Matmul
        out = patches @ w_flat.T + bias
        out = out.reshape(N, H, W, C_out).transpose(0,3,1,2)
        return out
    
    def maxpool2d(x):
        # x: (N, C, H, W) with H,W even
        N, C, H, W = x.shape
        x = x.reshape(N, C, H//2, 2, W//2, 2)
        return x.max(axis=(3,5))
    
    def relu(x):
        return np.maximum(x, 0)
    
    # --- Stage 1 PCA ---
    # Extract 3x3 patches from raw images (no padding)
    # X: (N,1,28,28) -> patches: (N, 26, 26, 3, 3) -> (N*26*26, 9)
    patches1 = np.lib.stride_tricks.sliding_window_view(X, (3,3), axis=(2,3))
    # patches1 shape: (N,1,26,26,3,3)
    patches1 = patches1.transpose(0,2,3,1,4,5).reshape(-1, 9)
    mean1 = patches1.mean(axis=0)
    patches1_centered = patches1 - mean1
    C1 = patches1_centered.T @ patches1_centered / patches1.shape[0]
    eigvals1, eigvecs1 = np.linalg.eigh(C1)
    # Take top 16 eigenvectors (largest eigenvalues)
    idx1 = np.argsort(eigvals1)[::-1][:16]
    filters1 = eigvecs1[:, idx1].T  # (16, 9)
    conv1_weight = filters1.reshape(16, 1, 3, 3).astype(np.float32)
    # Bias to center: - w @ mean1
    conv1_bias = -filters1 @ mean1
    conv1_bias = conv1_bias.astype(np.float32)
    
    # Apply conv1 + relu + maxpool
    H1 = conv2d(X, conv1_weight, conv1_bias)
    H1 = relu(H1)
    H1 = maxpool2d(H1)  # (N,16,14,14)
    
    # --- Stage 2 PCA ---
    # Extract 3x3 patches from H1 (no padding)
    # H1: (N,16,14,14) -> patches: (N, 12, 12, 16, 3, 3) -> (N*12*12, 144)
    patches2 = np.lib.stride_tricks.sliding_window_view(H1, (3,3), axis=(2,3))
    # patches2 shape: (N,16,12,12,3,3)
    patches2 = patches2.transpose(0,2,3,1,4,5).reshape(-1, 144)
    mean2 = patches2.mean(axis=0)
    patches2_centered = patches2 - mean2
    C2 = patches2_centered.T @ patches2_centered / patches2.shape[0]
    eigvals2, eigvecs2 = np.linalg.eigh(C2)
    idx2 = np.argsort(eigvals2)[::-1][:32]
    filters2 = eigvecs2[:, idx2].T  # (32, 144)
    conv2_weight = filters2.reshape(32, 16, 3, 3).astype(np.float32)
    conv2_bias = -filters2 @ mean2
    conv2_bias = conv2_bias.astype(np.float32)
    
    # Apply conv2 + relu + maxpool
    H2 = conv2d(H1, conv2_weight, conv2_bias)
    H2 = relu(H2)
    H2 = maxpool2d(H2)  # (N,32,7,7)
    F = H2.reshape(N, -1)  # (N,1568)
    
    # --- LDA Classifier ---
    F_mean = F.mean(axis=0)
    F_centered = F - F_mean
    
    # PCA reduction to 200 dims
    U, S, Vt = np.linalg.svd(F_centered, full_matrices=False)
    k = 200
    Vt_k = Vt[:k, :]  # (k, 1568)
    Z = F_centered @ Vt_k.T  # (N, k)
    
    # LDA
    classes = np.unique(y)
    n_classes = len(classes)
    mu = np.zeros((n_classes, k), dtype=np.float64)
    for i, c in enumerate(classes):
        mu[i] = Z[y == c].mean(axis=0)
    
    Sw = np.zeros((k, k), dtype=np.float64)
    for i in range(N):
        diff = Z[i] - mu[y[i]]
        Sw += np.outer(diff, diff)
    Sw /= N
    # Regularization
    reg = 1e-3 * np.trace(Sw) / k
    Sw += reg * np.eye(k)
    
    Sw_inv = np.linalg.inv(Sw)
    W_lda = Sw_inv @ mu.T  # (k, 10)
    b_lda = -0.5 * np.sum(mu.T * W_lda, axis=0) + np.log(np.bincount(y, minlength=n_classes) / N)
    
    head_weight = W_lda.T @ Vt_k  # (10, 1568)
    head_bias = b_lda - F_mean @ head_weight.T  # (10,)
    
    # Assign to w
    w["conv1.weight"] = conv1_weight
    w["conv1.bias"] = conv1_bias
    w["conv2.weight"] = conv2_weight
    w["conv2.bias"] = conv2_bias
    w["head.weight"] = head_weight.astype(np.float32)
    w["head.bias"] = head_bias.astype(np.float32)
    
    # Diagnostics
    diagnostics = {
        "conv1_weight_norm": float(np.linalg.norm(conv1_weight)),
        "conv2_weight_norm": float(np.linalg.norm(conv2_weight)),
        "head_weight_norm": float(np.linalg.norm(head_weight)),
        "head_bias_max_abs": float(np.max(np.abs(head_bias))),
        "feature_mean": float(F.mean()),
        "feature_std": float(F.std()),
        "pca_k": k,
        "reg": float(reg)
    }
    
    return {"weights": w, "diagnostics": diagnostics}
