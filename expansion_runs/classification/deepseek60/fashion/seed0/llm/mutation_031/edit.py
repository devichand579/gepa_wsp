import numpy as np

def edit(w, f):
    # Read current weights
    conv1_w = w["conv1.weight"].copy()  # [16,1,3,3]
    conv1_b = w["conv1.bias"].copy()    # [16]
    conv2_w = w["conv2.weight"].copy()  # [32,16,3,3]
    conv2_b = w["conv2.bias"].copy()    # [32]
    head_w = w["head.weight"].copy()    # [10,1568]
    head_b = w["head.bias"].copy()      # [10]

    # Feedback data
    x = f["x"]          # [2000,1,28,28]
    y = f["y"]          # [2000]
    features_orig = f["features"]  # [2000,1568]

    # Helper: conv2d with padding=1, stride=1
    def conv2d(x, weight, bias, padding=1):
        N, C, H, W = x.shape
        O, _, kH, kW = weight.shape
        x_pad = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant')
        out = np.zeros((N, O, H, W), dtype=np.float32)
        for i in range(kH):
            for j in range(kW):
                x_slice = x_pad[:, :, i:i+H, j:j+W]  # [N,C,H,W]
                w_slice = weight[:, :, i, j]          # [O,C]
                out += np.einsum('nchw,oc->nohw', x_slice, w_slice, optimize=True)
        out += bias[None, :, None, None]
        return out

    def relu(x):
        return np.maximum(x, 0)

    def maxpool2d(x, kernel=2, stride=2):
        N, C, H, W = x.shape
        H2, W2 = H // kernel, W // kernel
        x_reshaped = x.reshape(N, C, H2, kernel, W2, kernel)
        return x_reshaped.max(axis=(3, 5))

    # Compute features for flipped images
    x_flip = x[:, :, :, ::-1].copy()
    h1 = conv2d(x_flip, conv1_w, conv1_b, padding=1)
    h1 = relu(h1)
    h1 = maxpool2d(h1, 2, 2)
    h2 = conv2d(h1, conv2_w, conv2_b, padding=1)
    h2 = relu(h2)
    h2 = maxpool2d(h2, 2, 2)
    features_flip = h2.reshape(x.shape[0], -1)

    # Combine original and flipped features
    X = np.vstack([features_orig, features_flip]).astype(np.float32)  # [4000,1568]
    y_aug = np.concatenate([y, y]).astype(np.int64)                   # [4000]

    # Standardize features
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    X_std = (X - mean) / std

    # Add bias column
    X_aug = np.hstack([X_std, np.ones((X.shape[0], 1), dtype=np.float32)])

    # One-hot targets
    Y = np.zeros((X.shape[0], 10), dtype=np.float32)
    Y[np.arange(X.shape[0]), y_aug] = 1.0

    # Ridge regression (closed-form)
    # Do not regularize bias term
    n_features = X_aug.shape[1]
    reg = np.eye(n_features, dtype=np.float32) * 1e-3
    reg[-1, -1] = 0.0
    A = X_aug.T @ X_aug + reg
    B = X_aug.T @ Y
    W = np.linalg.solve(A, B)  # [n_features, 10]

    # Convert back to raw feature space
    W_std = W[:-1, :]          # [1568,10]
    b_std = W[-1, :]           # [10]
    head_w_new = (W_std * std[:, None]).T   # [10,1568]
    head_b_new = mean @ W_std + b_std       # [10]

    # Clip to safe range
    head_w_new = np.clip(head_w_new, -1000.0, 1000.0).astype(np.float32)
    head_b_new = np.clip(head_b_new, -1000.0, 1000.0).astype(np.float32)

    # Update weights
    w["head.weight"] = head_w_new
    w["head.bias"] = head_b_new

    # Diagnostics
    diagnostics = {
        "head_weight_mean": float(head_w_new.mean()),
        "head_weight_std": float(head_w_new.std()),
        "head_weight_max_abs": float(np.abs(head_w_new).max()),
        "head_bias_mean": float(head_b_new.mean()),
        "head_bias_std": float(head_b_new.std()),
        "lambda": 1e-3,
        "num_train": int(X.shape[0])
    }

    return {"weights": w, "diagnostics": diagnostics}
