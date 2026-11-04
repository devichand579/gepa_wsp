import numpy as np

def edit(w, f):
    # Extract full training features and targets
    features = f['features']          # (2000, 64, 16, 16) float32
    y = f['y']                        # (2000, 1, 16, 16) float32
    N, C, H, W = features.shape
    n_cols = C * 9 + 1                # 64*9 + 1 = 577

    # Accumulate normal equations in float64 for numerical stability
    XtX = np.zeros((n_cols, n_cols), dtype=np.float64)
    Xty = np.zeros(n_cols, dtype=np.float64)

    batch_size = 100
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B = end - start
        feat_batch = features[start:end]          # (B, 64, 16, 16)
        y_batch = y[start:end]                    # (B, 1, 16, 16)

        # Zero-pad to (B, 64, 18, 18)
        padded = np.pad(feat_batch, ((0,0),(0,0),(1,1),(1,1)), mode='constant')

        # Build design matrix for the 3x3 convolution
        slices = []
        for di in range(3):
            for dj in range(3):
                sl = padded[:, :, di:di+H, dj:dj+W]      # (B, 64, 16, 16)
                sl = sl.reshape(B, C, H*W).transpose(0, 2, 1)  # (B, 256, 64)
                slices.append(sl)
        X_batch = np.concatenate(slices, axis=2)          # (B, 256, 576)
        ones = np.ones((B, H*W, 1), dtype=np.float64)
        X_batch = np.concatenate([X_batch, ones], axis=2) # (B, 256, 577)

        X_batch = X_batch.reshape(-1, n_cols)             # (B*256, 577)
        y_flat = y_batch.reshape(-1)                      # (B*256,)

        XtX += X_batch.T @ X_batch
        Xty += X_batch.T @ y_flat

    # Solve least-squares (minimum-norm solution via lstsq on normal equations)
    w_out = np.linalg.lstsq(XtX, Xty, rcond=None)[0]      # (577,)

    # Reorder columns to match PyTorch weight layout (c, di, dj)
    perm = np.zeros(C*9, dtype=int)
    for c in range(C):
        for di in range(3):
            for dj in range(3):
                weight_idx = c*9 + di*3 + dj
                col_idx = (di*3 + dj)*C + c
                perm[weight_idx] = col_idx
    w_ordered = w_out[:C*9][perm].reshape(C, 3, 3)
    weight = w_ordered.reshape(1, C, 3, 3)
    bias = w_out[C*9]

    # Update the output layer weights
    w['out.weight'] = weight.astype(np.float32)
    w['out.bias'] = np.array([bias], dtype=np.float32)

    # Diagnostics
    mse_before = float(np.mean((f['prediction'] - f['y'])**2))
    total_ss = float(np.sum(y.astype(np.float64)**2))
    rss = total_ss - float(w_out @ Xty)
    mse_after = rss / (N * H * W)

    diagnostics = {
        'mse_before': mse_before,
        'mse_after_estimate': mse_after,
        'out_weight_norm': float(np.linalg.norm(weight)),
        'out_bias': float(bias),
        'n_params': int(n_cols),
        'ridge': 0.0
    }

    return {'weights': w, 'diagnostics': diagnostics}
