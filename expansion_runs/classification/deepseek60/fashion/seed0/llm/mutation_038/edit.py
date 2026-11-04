import numpy as np

def edit(w, f):
    # Extract feedback data
    x = f["x"]  # (2000, 1, 28, 28)
    y = f["y"]  # (2000,)
    N = x.shape[0]
    
    # Helper: extract patches for convolution with padding
    def get_patches(x, K, pad):
        # x: (N, C, H, W)
        N, C, H, W = x.shape
        x_pad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
        # sliding_window_view on last two dims
        patches = np.lib.stride_tricks.sliding_window_view(x_pad, (K, K), axis=(2,3))
        # patches shape: (N, C, H, W, K, K)
        return patches
    
    # Helper: conv2d forward using patches and filters
    def conv_forward(x, weight, bias, pad=1):
        # x: (N, C, H, W)
        # weight: (F, C, K, K)
        # bias: (F,)
        N, C, H, W = x.shape
        F, _, K, _ = weight.shape
        patches = get_patches(x, K, pad)  # (N, C, H, W, K, K)
        # Rearrange to (N, H, W, C*K*K)
        patches = patches.transpose(0, 2, 3, 1, 4, 5).reshape(N, H, W, C*K*K)
        # weight_flat: (F, C*K*K)
        weight_flat = weight.reshape(F, -1)
        # out: (N, H, W, F)
        out = patches @ weight_flat.T + bias.reshape(1, 1, 1, F)
        # Transpose to (N, F, H, W)
        out = out.transpose(0, 3, 1, 2)
        return out
    
    def relu(x):
        return np.maximum(x, 0)
    
    def maxpool2d(x, size=2, stride=2):
        # x: (N, C, H, W)
        N, C, H, W = x.shape
        H_out = (H - size) // stride + 1
        W_out = (W - size) // stride + 1
        out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                out[:, :, i, j] = np.max(x[:, :, h_start:h_start+size, w_start:w_start+size], axis=(2,3))
        return out
    
    # --- Stage 1: PCA on 3x3 patches of input ---
    patches1 = get_patches(x, 3, 1)  # (N, 1, 28, 28, 3, 3)
    patches1 = patches1.transpose(0, 2, 3, 1, 4, 5).reshape(-1, 9)  # (N*28*28, 9)
    mean1 = patches1.mean(axis=0)
    patches1_centered = patches1 - mean1
    cov1 = patches1_centered.T @ patches1_centered / patches1.shape[0]
    eigvals1, eigvecs1 = np.linalg.eigh(cov1)
    # Sort descending
    idx1 = np.argsort(eigvals1)[::-1]
    eigvecs1 = eigvecs1[:, idx1]
    # Take top 9 (max possible), pad to 16 with zeros
    filters1 = np.zeros((16, 9), dtype=np.float32)
    filters1[:9] = eigvecs1[:, :9].T  # (9,9) -> each row is a filter
    # Reshape to (16, 1, 3, 3)
    conv1_weight = filters1.reshape(16, 1, 3, 3).astype(np.float32)
    conv1_bias = np.zeros(16, dtype=np.float32)
    
    # --- Stage 2: Compute conv1 output, pool, then PCA on 3x3 patches of 16-channel maps ---
    conv1_out = conv_forward(x, conv1_weight, conv1_bias, pad=1)
    conv1_out = relu(conv1_out)
    conv1_pool = maxpool2d(conv1_out, 2, 2)  # (N, 16, 14, 14)
    
    patches2 = get_patches(conv1_pool, 3, 1)  # (N, 16, 14, 14, 3, 3)
    patches2 = patches2.transpose(0, 2, 3, 1, 4, 5).reshape(-1, 16*9)  # (N*14*14, 144)
    mean2 = patches2.mean(axis=0)
    patches2_centered = patches2 - mean2
    cov2 = patches2_centered.T @ patches2_centered / patches2.shape[0]
    eigvals2, eigvecs2 = np.linalg.eigh(cov2)
    idx2 = np.argsort(eigvals2)[::-1]
    eigvecs2 = eigvecs2[:, idx2]
    # Take top 32
    filters2 = eigvecs2[:, :32].T  # (32, 144)
    conv2_weight = filters2.reshape(32, 16, 3, 3).astype(np.float32)
    conv2_bias = np.zeros(32, dtype=np.float32)
    
    # --- Stage 3: Compute final features ---
    conv2_out = conv_forward(conv1_pool, conv2_weight, conv2_bias, pad=1)
    conv2_out = relu(conv2_out)
    conv2_pool = maxpool2d(conv2_out, 2, 2)  # (N, 32, 7, 7)
    X = conv2_pool.reshape(N, -1)  # (N, 1568)
    
    # --- Stage 4: Train LDA head with regularization ---
    # Normalize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std
    
    # Compute class means and within-class scatter
    classes = np.unique(y)
    n_classes = len(classes)
    mu = np.zeros((n_classes, X_norm.shape[1]), dtype=np.float64)
    S_w = np.zeros((X_norm.shape[1], X_norm.shape[1]), dtype=np.float64)
    for c in classes:
        X_c = X_norm[y == c]
        mu_c = X_c.mean(axis=0)
        mu[c] = mu_c
        X_c_centered = X_c - mu_c
        S_w += X_c_centered.T @ X_c_centered
    
    # Regularization
    lambda_reg = 1e-2
    S_w_reg = S_w + lambda_reg * np.eye(S_w.shape[0])
    S_w_inv = np.linalg.inv(S_w_reg)
    
    # LDA weights and biases
    head_weight = np.zeros((n_classes, X_norm.shape[1]), dtype=np.float32)
    head_bias = np.zeros(n_classes, dtype=np.float32)
    for i, c in enumerate(classes):
        w_c = S_w_inv @ mu[i]
        head_weight[i] = w_c
        head_bias[i] = -0.5 * mu[i] @ w_c + np.log((y == c).sum() / N)
    
    # Convert back to original feature space
    # score = head_weight @ X_norm + head_bias
    # X_norm = (X - X_mean) / X_std
    # score = (head_weight / X_std) @ X + (head_bias - head_weight @ (X_mean / X_std))
    head_weight_orig = head_weight / X_std
    head_bias_orig = head_bias - head_weight_orig @ X_mean
    
    # Assign to w
    w["conv1.weight"] = conv1_weight
    w["conv1.bias"] = conv1_bias
    w["conv2.weight"] = conv2_weight
    w["conv2.bias"] = conv2_bias
    w["head.weight"] = head_weight_orig.astype(np.float32)
    w["head.bias"] = head_bias_orig.astype(np.float32)
    
    # Diagnostics
    diagnostics = {
        "conv1_weight_norm": float(np.linalg.norm(conv1_weight)),
        "conv2_weight_norm": float(np.linalg.norm(conv2_weight)),
        "head_weight_norm": float(np.linalg.norm(head_weight_orig)),
        "head_bias_max_abs": float(np.max(np.abs(head_bias_orig))),
        "feature_mean": float(X_mean.mean()),
        "feature_std": float(X_std.mean()),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
