import numpy as np
import math

def edit(w, f):
    # Extract full training features and targets
    features = f['features']  # (2000, 64, 16, 16)
    y = f['y']                # (2000, 1, 16, 16)
    
    N, C, H, W = features.shape
    assert C == 64 and H == 16 and W == 16
    
    # Pad features spatially by 1 (for 3x3 conv with padding=1)
    pad = 1
    features_padded = np.pad(features, ((0,0), (0,0), (pad, pad), (pad, pad)), mode='constant')
    
    # Create sliding windows of shape (N, C, H, W, 3, 3)
    windows = np.lib.stride_tricks.sliding_window_view(features_padded, (3, 3), axis=(2, 3))
    # Transpose to (N, H, W, C, 3, 3) to match weight layout [out, in, kr, kc]
    windows = windows.transpose(0, 2, 3, 1, 4, 5)
    # Reshape to design matrix A: (N*H*W, C*9)
    A = windows.reshape(N * H * W, C * 9)
    
    # Add bias column
    A = np.concatenate([A, np.ones((N * H * W, 1), dtype=np.float32)], axis=1)
    
    # Flatten targets: (N*H*W, 1)
    y_flat = y.transpose(0, 2, 3, 1).reshape(N * H * W, 1)
    
    # Convert to float64 for numerical stability
    A64 = A.astype(np.float64)
    y64 = y_flat.astype(np.float64)
    
    # Solve normal equations: (A^T A) w = A^T y
    G = A64.T @ A64
    b = A64.T @ y64
    
    # Add tiny ridge for numerical stability (negligible effect)
    ridge = 1e-8
    G += ridge * np.eye(G.shape[0])
    
    w_sol = np.linalg.solve(G, b)
    
    # Extract weights and bias
    out_weight = w_sol[:C * 9].reshape(1, C, 3, 3).astype(np.float32)
    out_bias = w_sol[C * 9:].reshape(1).astype(np.float32)
    
    # Update the weights dictionary
    w['out.weight'] = out_weight
    w['out.bias'] = out_bias
    
    # Compute diagnostics
    pred = A64 @ w_sol
    mse = np.mean((pred - y64) ** 2)
    diagnostics = {
        "new_mse": float(mse),
        "out_weight_max": float(np.max(np.abs(out_weight))),
        "out_bias": float(out_bias[0]),
        "ridge": ridge
    }
    
    return {"weights": w, "diagnostics": diagnostics}
