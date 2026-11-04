import numpy as np
import math
from numpy.lib.stride_tricks import sliding_window_view

def edit(w, f):
    # Extract feedback arrays
    features = f['features']          # (2000, 64, 16, 16)
    y = f['y']                        # (2000, 1, 16, 16)
    
    N, C, H, W = features.shape
    assert C == 64 and H == 16 and W == 16
    assert y.shape == (N, 1, H, W)
    
    # We will solve for out.weight (1,64,3,3) and out.bias (1)
    # Parameters: 64*3*3 = 576 weights + 1 bias = 577
    n_params = 576 + 1
    
    # Accumulate normal equations: A = X^T X, b = X^T y
    A = np.zeros((n_params, n_params), dtype=np.float64)
    b_vec = np.zeros(n_params, dtype=np.float64)
    
    # Process in chunks to keep memory low and use BLAS efficiently
    chunk_size = 200
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        F_batch = features[start:end]   # (B, 64, 16, 16)
        y_batch = y[start:end]          # (B, 1, 16, 16)
        B = F_batch.shape[0]
        
        # Pad features to (B, 64, 18, 18)
        F_pad = np.pad(F_batch, ((0,0), (0,0), (1,1), (1,1)), mode='constant')
        
        # Extract 3x3 patches: sliding_window_view on axes (2,3)
        # F_pad shape: (B, 64, 18, 18)
        windows = sliding_window_view(F_pad, (3, 3), axis=(2, 3))
        # windows shape: (B, 64, 16, 16, 3, 3)
        # Transpose to (B, 16, 16, 64, 3, 3) then reshape to (B*256, 576)
        P = windows.transpose(0, 2, 3, 1, 4, 5).reshape(B * 256, 576)
        
        # Add bias column
        P_aug = np.concatenate([P, np.ones((B * 256, 1), dtype=np.float32)], axis=1)  # (B*256, 577)
        
        # Flatten y
        y_flat = y_batch.reshape(-1).astype(np.float64)  # (B*256,)
        
        # Accumulate
        A += P_aug.T.astype(np.float64) @ P_aug.astype(np.float64)
        b_vec += P_aug.T.astype(np.float64) @ y_flat
    
    # Ridge regularization
    ridge = 1e-4
    A += ridge * np.eye(n_params, dtype=np.float64)
    
    # Solve linear system
    theta = np.linalg.solve(A, b_vec)
    
    # Extract weights and bias
    out_weight = theta[:576].reshape(1, 64, 3, 3).astype(np.float32)
    out_bias = np.array([theta[576]], dtype=np.float32)
    
    # Update weights
    w['out.weight'] = out_weight
    w['out.bias'] = out_bias
    
    # Diagnostics
    diagnostics = {
        "out_weight_max": float(np.max(np.abs(out_weight))),
        "out_weight_mean": float(np.mean(out_weight)),
        "out_bias": float(out_bias[0]),
        "ridge": ridge,
        "n_samples": int(N),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
