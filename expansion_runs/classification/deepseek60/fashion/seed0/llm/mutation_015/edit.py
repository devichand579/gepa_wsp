import numpy as np
import math

def edit(w, f):
    # Extract feedback data
    x = f["x"].astype(np.float32)  # (2000,1,28,28)
    y = f["y"].astype(np.int64)    # (2000,)
    N = x.shape[0]
    
    # Helper: maxpool2d with stride 2, kernel 2
    def maxpool2d(a):
        # a: (N,C,H,W) with H,W even
        n, c, h, w = a.shape
        a = a.reshape(n, c, h//2, 2, w//2, 2)
        return a.max(axis=(3,5))
    
    # Helper: conv2d with padding=1, stride=1
    def conv2d(x, weight, bias):
        # x: (N, C_in, H, W)
        # weight: (C_out, C_in, 3, 3)
        # bias: (C_out,)
        n, c_in, h, w = x.shape
        c_out = weight.shape[0]
        # Pad with 1 zero on each side
        x_pad = np.pad(x, ((0,0),(0,0),(1,1),(1,1)), mode='constant')
        # Use sliding_window_view to get patches
        # x_pad: (N, C_in, H+2, W+2)
        # patches: (N, C_in, H, W, 3, 3)
        patches = np.lib.stride_tricks.sliding_window_view(x_pad, (3,3), axis=(2,3))
        # patches shape: (N, C_in, H, W, 3, 3)
        # weight shape: (C_out, C_in, 3, 3)
        # We want output: (N, C_out, H, W)
        # Compute via einsum: 'nchwkl,ockl->nohw' but careful with indices
        # patches: n, c, h, w, k, l
        # weight: o, c, k, l
        out = np.einsum('nchwkl,ockl->nohw', patches, weight)
        out += bias[None, :, None, None]
        return out
    
    # 1. Compute conv1 filters via PCA on 3x3 patches
    # Pad x with 1 zero
    x_pad = np.pad(x, ((0,0),(0,0),(1,1),(1,1)), mode='constant')
    # Extract 3x3 patches: shape (N, 1, 28, 28, 3, 3)
    patches1 = np.lib.stride_tricks.sliding_window_view(x_pad, (3,3), axis=(2,3))
    # Reshape to (N*28*28, 9)
    patches1 = patches1.reshape(-1, 9)
    # Compute mean and covariance
    mean1 = patches1.mean(axis=0)
    patches1_centered = patches1 - mean1
    cov1 = np.cov(patches1_centered, rowvar=False)
    # Eigen decomposition
    eigvals1, eigvecs1 = np.linalg.eigh(cov1)
    # Sort descending
    idx1 = np.argsort(eigvals1)[::-1]
    eigvecs1 = eigvecs1[:, idx1]
    # Take top 16
    filters1 = eigvecs1[:, :16].T  # (16, 9)
    # Reshape to (16, 1, 3, 3)
    conv1_weight = filters1.reshape(16, 1, 3, 3).astype(np.float32)
    conv1_bias = np.zeros(16, dtype=np.float32)
    
    # 2. Compute conv1 output on x
    conv1_out = conv2d(x, conv1_weight, conv1_bias)
    conv1_out = np.maximum(conv1_out, 0)  # ReLU
    conv1_out = maxpool2d(conv1_out)       # (N,16,14,14)
    
    # 3. Compute conv2 filters via PCA on 3x3 patches of conv1_out
    # Pad conv1_out with 1 zero
    c1_pad = np.pad(conv1_out, ((0,0),(0,0),(1,1),(1,1)), mode='constant')
    # Extract 3x3 patches: shape (N, 16, 14, 14, 3, 3)
    patches2 = np.lib.stride_tricks.sliding_window_view(c1_pad, (3,3), axis=(2,3))
    # Reshape to (N*14*14, 16*3*3=144)
    patches2 = patches2.reshape(-1, 144)
    mean2 = patches2.mean(axis=0)
    patches2_centered = patches2 - mean2
    cov2 = np.cov(patches2_centered, rowvar=False)
    eigvals2, eigvecs2 = np.linalg.eigh(cov2)
    idx2 = np.argsort(eigvals2)[::-1]
    eigvecs2 = eigvecs2[:, idx2]
    # Take top 32
    filters2 = eigvecs2[:, :32].T  # (32, 144)
    # Reshape to (32, 16, 3, 3)
    conv2_weight = filters2.reshape(32, 16, 3, 3).astype(np.float32)
    conv2_bias = np.zeros(32, dtype=np.float32)
    
    # 4. Compute conv2 output
    conv2_out = conv2d(conv1_out, conv2_weight, conv2_bias)
    conv2_out = np.maximum(conv2_out, 0)
    conv2_out = maxpool2d(conv2_out)       # (N,32,7,7)
    
    # 5. Flatten to features
    X = conv2_out.reshape(N, -1)           # (N, 1568)
    
    # 6. Train ridge regression classifier
    # One-hot encode y
    Y = np.zeros((N, 10), dtype=np.float32)
    Y[np.arange(N), y] = 1.0
    
    # Add bias column
    X_aug = np.hstack([X, np.ones((N, 1), dtype=np.float32)])
    
    # Ridge regression: W = (X^T X + lambda I)^-1 X^T Y
    # Use a small lambda for regularization
    lam = 1e-2
    XtX = X_aug.T @ X_aug
    XtX += lam * np.eye(XtX.shape[0], dtype=np.float32)
    XtY = X_aug.T @ Y
    W = np.linalg.solve(XtX, XtY)          # (1569, 10)
    
    # Extract head weights and bias
    head_weight = W[:-1, :].T.astype(np.float32)  # (10, 1568)
    head_bias = W[-1, :].astype(np.float32)       # (10,)
    
    # Clip to [-1000, 1000] to satisfy constraints
    conv1_weight = np.clip(conv1_weight, -1000, 1000)
    conv1_bias = np.clip(conv1_bias, -1000, 1000)
    conv2_weight = np.clip(conv2_weight, -1000, 1000)
    conv2_bias = np.clip(conv2_bias, -1000, 1000)
    head_weight = np.clip(head_weight, -1000, 1000)
    head_bias = np.clip(head_bias, -1000, 1000)
    
    # Update weights
    w["conv1.weight"] = conv1_weight
    w["conv1.bias"] = conv1_bias
    w["conv2.weight"] = conv2_weight
    w["conv2.bias"] = conv2_bias
    w["head.weight"] = head_weight
    w["head.bias"] = head_bias
    
    # Diagnostics
    diagnostics = {
        "conv1_weight_norm": float(np.linalg.norm(conv1_weight)),
        "conv2_weight_norm": float(np.linalg.norm(conv2_weight)),
        "head_weight_norm": float(np.linalg.norm(head_weight)),
        "head_bias_norm": float(np.linalg.norm(head_bias)),
        "ridge_lambda": lam,
        "max_abs_conv1": float(np.max(np.abs(conv1_weight))),
        "max_abs_conv2": float(np.max(np.abs(conv2_weight))),
        "max_abs_head": float(np.max(np.abs(head_weight))),
        "max_abs_head_bias": float(np.max(np.abs(head_bias))),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
