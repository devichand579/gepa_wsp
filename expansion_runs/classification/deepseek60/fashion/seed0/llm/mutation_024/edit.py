import numpy as np

def edit(w, f):
    # Extract training data
    X = f["x"]  # (2000,1,28,28)
    y = f["y"]  # (2000,)
    N = X.shape[0]
    
    # --- 1. Learn conv1 filters via PCA on 3x3 patches ---
    # Extract all 3x3 patches from all images (no padding, stride 1)
    patches = []
    for i in range(N):
        img = X[i, 0]  # (28,28)
        for r in range(26):
            for c in range(26):
                patches.append(img[r:r+3, c:c+3].flatten())
    patches = np.array(patches, dtype=np.float32)  # (N*676, 9)
    
    # Center patches
    mean_patch = patches.mean(axis=0)
    patches_centered = patches - mean_patch
    
    # Compute covariance and eigenvectors
    cov = patches_centered.T @ patches_centered / patches.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    
    # Take top 16 components as filters
    n_filters = 16
    filters = eigvecs[:, :n_filters].T  # (16, 9)
    # Reshape to (16,1,3,3)
    conv1_w = filters.reshape(16, 1, 3, 3).astype(np.float32)
    conv1_b = np.zeros(16, dtype=np.float32)
    
    w["conv1.weight"] = conv1_w
    w["conv1.bias"] = conv1_b
    
    # --- 2. Set conv2 as identity pass-through (first 16 channels) ---
    conv2_w = np.zeros((32, 16, 3, 3), dtype=np.float32)
    for i in range(16):
        conv2_w[i, i, 1, 1] = 1.0  # center weight
    conv2_b = np.zeros(32, dtype=np.float32)
    w["conv2.weight"] = conv2_w
    w["conv2.bias"] = conv2_b
    
    # --- 3. Compute features after conv1->relu->pool->conv2->relu->pool ---
    def conv2d_pad1(x, weight, bias):
        # x: (N, C_in, H, W), weight: (C_out, C_in, 3, 3), bias: (C_out,)
        N, C_in, H, W = x.shape
        C_out = weight.shape[0]
        out = np.zeros((N, C_out, H, W), dtype=np.float32)
        x_pad = np.pad(x, ((0,0), (0,0), (1,1), (1,1)), mode='constant')
        for n in range(N):
            for co in range(C_out):
                acc = np.zeros((H, W), dtype=np.float32)
                for ci in range(C_in):
                    for kr in range(3):
                        for kc in range(3):
                            acc += weight[co, ci, kr, kc] * x_pad[n, ci, kr:kr+H, kc:kc+W]
                out[n, co] = acc + bias[co]
        return out
    
    def relu(x):
        return np.maximum(x, 0)
    
    def maxpool2d(x):
        # x: (N, C, H, W) -> (N, C, H//2, W//2)
        N, C, H, W = x.shape
        x = x.reshape(N, C, H//2, 2, W//2, 2)
        return x.max(axis=(3,5))
    
    # Compute features for all training images
    # To save memory, process in batches
    batch_size = 200
    features_list = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        xb = X[start:end]  # (B,1,28,28)
        # conv1
        h = conv2d_pad1(xb, conv1_w, conv1_b)
        h = relu(h)
        h = maxpool2d(h)  # (B,16,14,14)
        # conv2
        h = conv2d_pad1(h, conv2_w, conv2_b)
        h = relu(h)
        h = maxpool2d(h)  # (B,32,7,7)
        # flatten
        h = h.reshape(h.shape[0], -1)  # (B, 1568)
        features_list.append(h)
    features = np.concatenate(features_list, axis=0)  # (N, 1568)
    
    # --- 4. Fit ridge regression classifier ---
    # One-hot encode labels
    Y = np.zeros((N, 10), dtype=np.float32)
    Y[np.arange(N), y] = 1.0
    
    # Augment features with bias column
    X_aug = np.hstack([features, np.ones((N, 1), dtype=np.float32)])  # (N, 1569)
    
    # Ridge regression: W = (X^T X + lambda I)^{-1} X^T Y
    lambda_reg = 1e-3
    XtX = X_aug.T @ X_aug
    XtX += lambda_reg * np.eye(XtX.shape[0], dtype=np.float32)
    XtY = X_aug.T @ Y
    W = np.linalg.solve(XtX, XtY)  # (1569, 10)
    
    # Extract head weights and bias
    head_w = W[:-1, :].T  # (10, 1568)
    head_b = W[-1, :]     # (10,)
    
    # Scale to reasonable magnitude (optional, for numerical stability)
    # The ridge solution is already small; we can scale up to make logits larger.
    # But scaling does not change argmax. We'll scale by a factor to make max abs ~100.
    max_abs = max(np.abs(head_w).max(), np.abs(head_b).max())
    if max_abs > 0:
        scale = 100.0 / max_abs
        head_w *= scale
        head_b *= scale
    
    w["head.weight"] = head_w.astype(np.float32)
    w["head.bias"] = head_b.astype(np.float32)
    
    # Diagnostics
    diag = {
        "conv1_filters_norm": float(np.linalg.norm(conv1_w)),
        "head_weight_max": float(np.abs(head_w).max()),
        "head_bias_max": float(np.abs(head_b).max()),
        "ridge_lambda": lambda_reg,
        "num_train": N,
        "feature_dim": features.shape[1],
    }
    
    return {"weights": w, "diagnostics": diag}
