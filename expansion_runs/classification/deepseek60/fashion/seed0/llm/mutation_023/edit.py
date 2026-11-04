import numpy as np

def edit(w, f):
    # Helper: 2D convolution with padding=1
    def conv2d(x, weight, bias, padding=1):
        N, C_in, H, W = x.shape
        C_out, _, K, _ = weight.shape
        x_pad = np.pad(x, ((0,0), (0,0), (padding, padding), (padding, padding)), mode='constant')
        H_out = H + 2*padding - K + 1
        W_out = W + 2*padding - K + 1
        windows = np.lib.stride_tricks.sliding_window_view(x_pad, (K, K), axis=(2,3))
        out = np.einsum('nchwkl,ockl->nohw', windows, weight)
        out += bias.reshape(1, C_out, 1, 1)
        return out

    # Helper: max pooling with kernel=2, stride=2
    def maxpool2d(x, kernel_size=2):
        N, C, H, W = x.shape
        H_out = H // kernel_size
        W_out = W // kernel_size
        x_reshaped = x.reshape(N, C, H_out, kernel_size, W_out, kernel_size)
        return x_reshaped.max(axis=(3,5))

    # Compute features for a batch of images using current conv weights
    def compute_features(images):
        conv1 = conv2d(images, w["conv1.weight"], w["conv1.bias"], padding=1)
        relu1 = np.maximum(conv1, 0)
        pool1 = maxpool2d(relu1, 2)
        conv2 = conv2d(pool1, w["conv2.weight"], w["conv2.bias"], padding=1)
        relu2 = np.maximum(conv2, 0)
        pool2 = maxpool2d(relu2, 2)
        return pool2.reshape(images.shape[0], -1)

    # Original features and labels
    X_orig = f["features"].astype(np.float64)  # (2000, 1568)
    y_orig = f["y"].astype(np.int64)           # (2000,)

    # Create horizontally flipped images
    x_imgs = f["x"].astype(np.float32)         # (2000, 1, 28, 28)
    x_flipped = x_imgs[:, :, :, ::-1].copy()   # flip columns

    # Compute features for flipped images
    X_flip = compute_features(x_flipped).astype(np.float64)

    # Combine original and flipped features
    X = np.vstack([X_orig, X_flip])            # (4000, 1568)
    y = np.concatenate([y_orig, y_orig])       # (4000,)

    N = X.shape[0]
    D = X.shape[1]

    # Add bias column
    X_aug = np.hstack([X, np.ones((N, 1))])    # (4000, 1569)

    # One-hot labels
    Y = np.zeros((N, 10), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Minimum-norm least squares solution
    W_aug_T, residuals, rank, s = np.linalg.lstsq(X_aug, Y, rcond=None)
    W_aug = W_aug_T.T                          # (10, 1569)

    W = W_aug[:, :D]                           # (10, 1568)
    b = W_aug[:, D]                            # (10,)

    # Scale to make logits confident, keeping max abs <= 1000
    max_abs = max(np.abs(W).max(), np.abs(b).max())
    if max_abs > 0:
        K = 1000.0 / max_abs
    else:
        K = 1.0
    W_scaled = W * K
    b_scaled = b * K

    # Ensure within bounds
    W_scaled = np.clip(W_scaled, -1000.0, 1000.0)
    b_scaled = np.clip(b_scaled, -1000.0, 1000.0)

    # Update head weights
    w["head.weight"] = W_scaled.astype(np.float32)
    w["head.bias"] = b_scaled.astype(np.float32)

    diagnostics = {
        "least_squares_rank": int(rank),
        "max_abs_W": float(np.abs(W_scaled).max()),
        "max_abs_b": float(np.abs(b_scaled).max()),
        "scale_factor": float(K),
        "residual_norm": float(np.linalg.norm(X_aug @ W_aug_T - Y)),
        "num_train": int(N),
    }
    return {"weights": w, "diagnostics": diagnostics}
