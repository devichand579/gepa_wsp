import numpy as np

def edit(w, f):
    # Extract conv weights
    conv1_w = w["conv1.weight"]
    conv1_b = w["conv1.bias"]
    conv2_w = w["conv2.weight"]
    conv2_b = w["conv2.bias"]
    
    # Helper functions
    def conv2d(x, weight, bias, padding=1, stride=1):
        N, C_in, H, W = x.shape
        C_out, _, kH, kW = weight.shape
        H_out = (H + 2*padding - kH) // stride + 1
        W_out = (W + 2*padding - kW) // stride + 1
        x_pad = np.pad(x, ((0,0), (0,0), (padding, padding), (padding, padding)), mode='constant')
        out = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)
        for i in range(kH):
            for j in range(kW):
                x_slice = x_pad[:, :, i:i+H_out*stride:stride, j:j+W_out*stride:stride]
                w_slice = weight[:, :, i, j]
                out += np.einsum('nchw,oc->nohw', x_slice, w_slice)
        out += bias[None, :, None, None]
        return out
    
    def maxpool2d(x):
        N, C, H, W = x.shape
        H_out = H // 2
        W_out = W // 2
        x_reshaped = x.reshape(N, C, H_out, 2, W_out, 2)
        return x_reshaped.max(axis=(3,5))
    
    def relu(x):
        return np.maximum(x, 0)
    
    # Compute features for flipped images
    x = f["x"]  # (2000, 1, 28, 28)
    x_flip = x[:, :, :, ::-1]  # horizontal flip
    
    # Forward pass for flipped
    h = conv2d(x_flip, conv1_w, conv1_b, padding=1, stride=1)
    h = relu(h)
    h = maxpool2d(h)
    h = conv2d(h, conv2_w, conv2_b, padding=1, stride=1)
    h = relu(h)
    h = maxpool2d(h)
    features_flip = h.reshape(h.shape[0], -1)  # (2000, 1568)
    
    # Original features from f
    features_orig = f["features"]  # (2000, 1568)
    
    # Combine
    X = np.vstack([features_orig, features_flip]).astype(np.float32)
    y = f["y"]
    Y = np.zeros((X.shape[0], 10), dtype=np.float32)
    Y[:len(y), y] = 1.0
    Y[len(y):, y] = 1.0
    
    # Standardize
    mean_X = X.mean(axis=0)
    std_X = X.std(axis=0) + 1e-6
    X_std = (X - mean_X) / std_X
    mean_Y = Y.mean(axis=0)
    Y_centered = Y - mean_Y
    
    # Ridge regression
    lam = 100.0
    A = X_std.T @ X_std + lam * np.eye(X_std.shape[1], dtype=np.float32)
    B = X_std.T @ Y_centered
    W = np.linalg.solve(A, B)  # (1568, 10)
    bias = mean_Y - mean_X @ W  # (10,)
    
    head_weight = W.T  # (10, 1568)
    head_bias = bias
    
    # Scale to max abs 10
    max_abs = max(np.max(np.abs(head_weight)), np.max(np.abs(head_bias)))
    scale = 10.0 / max_abs if max_abs > 0 else 1.0
    head_weight *= scale
    head_bias *= scale
    
    # Update w
    w["head.weight"] = head_weight.astype(np.float32)
    w["head.bias"] = head_bias.astype(np.float32)
    
    # Diagnostics
    diagnostics = {
        "lambda": lam,
        "scale": float(scale),
        "max_abs_head_weight": float(np.max(np.abs(w["head.weight"]))),
        "max_abs_head_bias": float(np.max(np.abs(w["head.bias"]))),
        "feature_mean_std": float(np.mean(std_X)),
        "num_train": int(X.shape[0])
    }
    
    return {"weights": w, "diagnostics": diagnostics}
