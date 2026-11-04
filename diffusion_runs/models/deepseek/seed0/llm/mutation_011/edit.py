import numpy as np
import math

def edit(w, f):
    # Extract arrays
    features = f['features']          # (2000, 64, 16, 16)
    y = f['y']                        # (2000, 1, 16, 16)
    N, C, H, W = features.shape
    assert C == 64 and H == 16 and W == 16
    assert y.shape == (N, 1, H, W)

    # Pad features spatially by 1 (for 3x3 conv with padding=1)
    F_pad = np.pad(features, ((0,0),(0,0),(1,1),(1,1)), mode='constant')
    # F_pad shape: (N, 64, 18, 18)

    # Build design matrix X for the output convolution.
    # For each output pixel (i,j), the input is the 3x3 neighborhood across all 64 channels.
    # Use sliding_window_view to get patches of shape (N, 64, 16, 16, 3, 3)
    from numpy.lib.stride_tricks import sliding_window_view
    patches = sliding_window_view(F_pad, (3, 3), axis=(2, 3))  # (N, 64, 16, 16, 3, 3)
    # Transpose to (N, 16, 16, 64, 3, 3) then reshape to (N*16*16, 64*9)
    X = patches.transpose(0, 2, 3, 1, 4, 5).reshape(N * H * W, C * 9)
    # Add bias column
    X = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)
    # X shape: (N*H*W, 577)

    # Target: flatten y to (N*H*W, 1)
    Y = y.transpose(0, 2, 3, 1).reshape(-1, 1)

    # Solve least squares with a small ridge for numerical stability.
    # Use normal equations: (X^T X + ridge I) w = X^T Y
    ridge = 1e-6
    XTX = X.T @ X
    XTY = X.T @ Y
    # Add ridge to diagonal (except bias? we can add to all for simplicity)
    XTX += ridge * np.eye(XTX.shape[0], dtype=np.float32)
    # Solve
    w_out = np.linalg.solve(XTX, XTY)  # shape (577, 1)

    # Extract weights and bias
    out_weight = w_out[:C*9].reshape(1, C, 3, 3).astype(np.float32)
    out_bias = w_out[C*9:].reshape(1).astype(np.float32)

    # Update w
    w['out.weight'] = out_weight
    w['out.bias'] = out_bias

    # Compute diagnostics
    # Training MSE after update (on the same data)
    pred = X @ w_out  # (N*H*W, 1)
    mse = float(np.mean((pred - Y) ** 2))
    diagnostics = {
        "mse_before": 0.3059,
        "mse_after_estimate": mse,
        "ridge": ridge,
        "out_weight_norm": float(np.linalg.norm(out_weight)),
        "out_bias": float(out_bias[0]),
        "n_params": int(C*9 + 1)
    }

    return {"weights": w, "diagnostics": diagnostics}
