import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def edit(w, f):
    # Extract features (input to the final linear layer) and targets
    features = f['features']  # (2000, 64, 16, 16)
    y = f['y']                # (2000, 1, 16, 16)
    N = features.shape[0]

    # We will fit out.weight (1,64,3,3) and out.bias (1) via ridge regression.
    # Design matrix: for each example and each pixel (i,j), the 576-dim patch
    # of the 3x3 neighborhood across all 64 channels, plus a bias column.
    n_params = 64 * 3 * 3 + 1  # 577
    XTX = np.zeros((n_params, n_params), dtype=np.float64)
    XTy = np.zeros(n_params, dtype=np.float64)

    chunk_size = 200
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        F = features[start:end]  # (B, 64, 16, 16)
        Y = y[start:end]         # (B, 1, 16, 16)
        B = F.shape[0]

        # Pad to 18x18 to allow 3x3 windows at all 16x16 positions
        F_pad = np.pad(F, ((0,0), (0,0), (1,1), (1,1)), mode='constant')
        # Extract 3x3 patches: (B, 64, 16, 16, 3, 3)
        patches = sliding_window_view(F_pad, (3, 3), axis=(2, 3))
        # Rearrange to (B, 16, 16, 64, 3, 3) and flatten to (B*256, 576)
        patches = patches.transpose(0, 2, 3, 1, 4, 5)
        X = patches.reshape(B * 256, 64 * 9)

        # Add bias column
        X = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)

        # Target: (B, 1, 16, 16) -> (B*256,)
        Y_flat = Y.transpose(0, 2, 3, 1).reshape(-1)

        # Accumulate normal equations (use float64 for stability)
        X = X.astype(np.float64)
        Y_flat = Y_flat.astype(np.float64)
        XTX += X.T @ X
        XTy += X.T @ Y_flat

    # Small ridge for numerical stability
    ridge = 1e-6
    XTX += ridge * np.eye(n_params)

    # Solve for optimal weights and bias
    w_flat = np.linalg.solve(XTX, XTy)

    # Reshape and cast back to float32
    w_conv = w_flat[:-1].reshape(1, 64, 3, 3).astype(np.float32)
    b = np.array([w_flat[-1]], dtype=np.float32)

    # Clip to safe range (should not be needed, but ensures finite bounds)
    w_conv = np.clip(w_conv, -1000.0, 1000.0)
    b = np.clip(b, -1000.0, 1000.0)

    # Update the output layer only; earlier layers remain unchanged so cached
    # features stay valid.
    w['out.weight'] = w_conv
    w['out.bias'] = b

    diagnostics = {
        'out_weight_max': float(np.max(np.abs(w_conv))),
        'out_bias': float(b[0]),
        'ridge': ridge,
        'n_samples': N,
    }

    return {"weights": w, "diagnostics": diagnostics}
