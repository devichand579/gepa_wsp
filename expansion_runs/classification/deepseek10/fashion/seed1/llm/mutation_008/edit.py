import numpy as np

def edit(w, f):
    # Extract data
    x = f["x"]          # (2000,1,28,28)
    y = f["y"]          # (2000,)
    N = x.shape[0]
    C = 10

    # Helper: extract 3x3 patches with padding=1
    def extract_patches(images):
        # images: (N, C, H, W)
        # returns: (N*H*W, C*9)
        N, C, H, W = images.shape
        padded = np.pad(images, ((0,0),(0,0),(1,1),(1,1)), mode='constant')
        patches = np.zeros((N, H, W, C, 3, 3), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                patches[:, :, :, :, i, j] = padded[:, :, i:i+H, j:j+W]
        # reshape to (N*H*W, C*9)
        patches = patches.reshape(N*H*W, C*9)
        return patches

    # ---------- Learn conv1 filters via LDA on input patches ----------
    patches1 = extract_patches(x)  # (2000*784, 9)
    # Compute class means
    mu_c1 = np.zeros((C, 9), dtype=np.float64)
    counts = np.zeros(C, dtype=np.int64)
    for c in range(C):
        idx = np.where(y == c)[0]
        counts[c] = len(idx)
        # For each image, we have 784 patches. We can compute mean patch per class.
        # To save memory, we can compute sum of patches per class.
        # patches1 is (N*784, 9). We need to map image index to patch index.
        # Each image has 784 patches. So image i patches are patches1[i*784:(i+1)*784].
        # We can compute sum over these.
        # Better: reshape patches1 to (N, 784, 9)
        patches1_reshaped = patches1.reshape(N, 784, 9)
        mu_c1[c] = patches1_reshaped[idx].mean(axis=(0,1))
    mu1 = patches1.mean(axis=0)

    # Compute within-class scatter S_w1 (9x9)
    S_w1 = np.zeros((9, 9), dtype=np.float64)
    patches1_reshaped = patches1.reshape(N, 784, 9)
    for c in range(C):
        idx = np.where(y == c)[0]
        # centered patches for class c
        centered = patches1_reshaped[idx] - mu_c1[c]  # (N_c, 784, 9)
        # sum of outer products: for each patch, centered^T centered
        # We can compute sum over N_c*784 patches
        # Reshape to (N_c*784, 9)
        centered_flat = centered.reshape(-1, 9)
        S_w1 += centered_flat.T @ centered_flat

    # Between-class scatter S_b1
    S_b1 = np.zeros((9, 9), dtype=np.float64)
    for c in range(C):
        diff = mu_c1[c] - mu1
        S_b1 += counts[c] * np.outer(diff, diff)

    # Add small ridge to S_w1 for numerical stability
    S_w1 += np.eye(9) * 1e-3 * np.trace(S_w1) / 9

    # Solve generalized eigenvalue problem: S_b1 v = lambda S_w1 v
    # Equivalent to inv(S_w1) @ S_b1 v = lambda v
    M1 = np.linalg.solve(S_w1, S_b1)
    eigvals1, eigvecs1 = np.linalg.eig(M1)
    # Sort by eigenvalue descending
    idx1 = np.argsort(eigvals1)[::-1]
    eigvecs1 = eigvecs1[:, idx1]
    # Take top 16 eigenvectors (real part)
    filters1 = np.real(eigvecs1[:, :16]).T  # (16, 9)
    # Normalize filters to unit norm
    filters1 = filters1 / np.linalg.norm(filters1, axis=1, keepdims=True)
    # Reshape to (16,1,3,3)
    conv1_weight = filters1.reshape(16, 1, 3, 3).astype(np.float32)
    conv1_bias = np.zeros(16, dtype=np.float32)

    # ---------- Compute conv1 output for all training data ----------
    # We need to apply conv1 with padding=1, then ReLU, then maxpool2d
    # We can use a simple convolution implementation.
    def conv2d(x, weight, bias, padding=1):
        # x: (N, C_in, H, W)
        # weight: (C_out, C_in, kH, kW)
        # bias: (C_out,)
        N, C_in, H, W = x.shape
        C_out, _, kH, kW = weight.shape
        # Pad input
        x_pad = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant')
        # Output dimensions
        H_out = H
        W_out = W
        out = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)
        for i in range(kH):
            for j in range(kW):
                # Extract sliding window
                window = x_pad[:, :, i:i+H_out, j:j+W_out]  # (N, C_in, H_out, W_out)
                # For each output channel, sum over input channels
                # weight[:, :, i, j] is (C_out, C_in)
                out += np.tensordot(window, weight[:, :, i, j], axes=([1], [1])).transpose(0, 3, 1, 2)
        out += bias.reshape(1, C_out, 1, 1)
        return out

    def relu(x):
        return np.maximum(x, 0)

    def max_pool2d(x, kernel_size=2, stride=2):
        N, C, H, W = x.shape
        H_out = H // stride
        W_out = W // stride
        x_reshaped = x.reshape(N, C, H_out, stride, W_out, stride)
        return x_reshaped.max(axis=(3, 5))

    # Compute conv1 output
    conv1_out = conv2d(x, conv1_weight, conv1_bias, padding=1)
    conv1_out = relu(conv1_out)
    pool1 = max_pool2d(conv1_out, 2, 2)  # (2000, 16, 14, 14)

    # ---------- Learn conv2 filters via LDA on pooled conv1 patches ----------
    # Extract 3x3 patches from pool1 (16 channels)
    # pool1 shape: (2000, 16, 14, 14)
    # We need patches of size 3x3x16 = 144
    patches2 = extract_patches(pool1)  # (2000*196, 144)
    # Compute class means
    mu_c2 = np.zeros((C, 144), dtype=np.float64)
    for c in range(C):
        idx = np.where(y == c)[0]
        patches2_reshaped = patches2.reshape(N, 196, 144)
        mu_c2[c] = patches2_reshaped[idx].mean(axis=(0,1))
    mu2 = patches2.mean(axis=0)

    # Compute within-class scatter S_w2 (144x144)
    S_w2 = np.zeros((144, 144), dtype=np.float64)
    patches2_reshaped = patches2.reshape(N, 196, 144)
    for c in range(C):
        idx = np.where(y == c)[0]
        centered = patches2_reshaped[idx] - mu_c2[c]
        centered_flat = centered.reshape(-1, 144)
        S_w2 += centered_flat.T @ centered_flat

    # Between-class scatter S_b2
    S_b2 = np.zeros((144, 144), dtype=np.float64)
    for c in range(C):
        diff = mu_c2[c] - mu2
        S_b2 += counts[c] * np.outer(diff, diff)

    # Add ridge
    S_w2 += np.eye(144) * 1e-3 * np.trace(S_w2) / 144

    # Solve generalized eigenvalue problem
    M2 = np.linalg.solve(S_w2, S_b2)
    eigvals2, eigvecs2 = np.linalg.eig(M2)
    idx2 = np.argsort(eigvals2)[::-1]
    eigvecs2 = eigvecs2[:, idx2]
    # Take top 32 eigenvectors
    filters2 = np.real(eigvecs2[:, :32]).T  # (32, 144)
    # Normalize
    filters2 = filters2 / np.linalg.norm(filters2, axis=1, keepdims=True)
    # Reshape to (32, 16, 3, 3)
    conv2_weight = filters2.reshape(32, 16, 3, 3).astype(np.float32)
    conv2_bias = np.zeros(32, dtype=np.float32)

    # ---------- Compute new features and fit head ----------
    # Apply conv2
    conv2_out = conv2d(pool1, conv2_weight, conv2_bias, padding=1)
    conv2_out = relu(conv2_out)
    pool2 = max_pool2d(conv2_out, 2, 2)  # (2000, 32, 7, 7)
    features = pool2.reshape(N, -1)      # (2000, 1568)

    # Fit head using least squares on one-hot targets
    # We can use ridge for stability
    Y_onehot = np.zeros((N, C), dtype=np.float64)
    Y_onehot[np.arange(N), y] = 1.0

    # Add small ridge to X^T X
    X = features.astype(np.float64)
    XtX = X.T @ X
    XtX += np.eye(XtX.shape[0]) * 1e-3 * np.trace(XtX) / XtX.shape[0]
    XtY = X.T @ Y_onehot
    head_weight = np.linalg.solve(XtX, XtY).T  # (10, 1568)
    head_bias = np.zeros(C, dtype=np.float64)

    # Convert to float32
    head_weight = head_weight.astype(np.float32)
    head_bias = head_bias.astype(np.float32)

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
        "head_weight_std": float(np.std(head_weight)),
        "train_accuracy": float(np.mean(np.argmax(X @ head_weight.T + head_bias, axis=1) == y))
    }

    return {"weights": w, "diagnostics": diagnostics}
