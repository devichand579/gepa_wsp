import numpy as np
import math
from numpy.lib.stride_tricks import sliding_window_view

def edit(w, f):
    # Extract features and targets
    features = f['features']  # (2000,64,16,16)
    y = f['y']                # (2000,1,16,16)
    N = features.shape[0]
    C = features.shape[1]
    H = features.shape[2]
    W = features.shape[3]
    K = 3
    P = C * K * K + 1
    
    # Accumulators for normal equations (float64 for stability)
    XtX = np.zeros((P, P), dtype=np.float64)
    Xty = np.zeros((P, 1), dtype=np.float64)
    
    batch_size = 100
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        fb = features[i:end]  # (B,C,H,W)
        yb = y[i:end]         # (B,1,H,W)
        B = fb.shape[0]
        
        # Pad features with zeros (padding=1 for 3x3 conv)
        fb_pad = np.pad(fb, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant')
        # Extract 3x3 patches: (B,C,H,W,K,K)
        patches = sliding_window_view(fb_pad, (K, K), axis=(2, 3))
        # Transpose to (B,H,W,C,K,K) and reshape to (B*H*W, C*K*K)
        patches = patches.transpose(0, 2, 3, 1, 4, 5)
        X = patches.reshape(B * H * W, C * K * K)
        # Add bias column
        X = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)
        # Flatten target
        y_flat = yb.reshape(B * H * W, 1)
        
        # Accumulate normal equations
        XtX += X.T @ X
        Xty += X.T @ y_flat
    
    # Solve least squares (no ridge needed: 512k outputs vs 577 params)
    w_out = np.linalg.solve(XtX, Xty).ravel()
    out_weight = w_out[:C * K * K].reshape(1, C, K, K)
    out_bias = w_out[C * K * K:]
    
    # Copy and update weights
    w_new = {k: v.copy() for k, v in w.items()}
    w_new['out.weight'] = out_weight.astype(np.float32)
    w_new['out.bias'] = out_bias.astype(np.float32)
    
    # Diagnostics
    diagnostics = {
        'out_weight_max': float(np.max(np.abs(out_weight))),
        'out_bias': float(out_bias[0]),
        'ridge': 0.0
    }
    return {'weights': w_new, 'diagnostics': diagnostics}
