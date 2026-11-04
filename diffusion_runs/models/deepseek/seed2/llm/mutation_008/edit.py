import numpy as np
import math

def edit(w, f):
    # Extract features and target
    features = f['features']  # (2000, 64, 16, 16)
    y = f['y']                # (2000, 1, 16, 16)
    
    N, C, H, W = features.shape
    assert C == 64 and H == 16 and W == 16, "Unexpected feature shape"
    
    # Pad features to (N, C, H+2, W+2)
    features_pad = np.pad(features, ((0,0), (0,0), (1,1), (1,1)), mode='constant')
    
    # Create strided view for 3x3 patches: (N, H, W, C, 3, 3)
    itemsize = features_pad.itemsize
    strides = features_pad.strides
    # features_pad shape: (N, C, H+2, W+2)
    # strides: (C*(H+2)*(W+2)*itemsize, (H+2)*(W+2)*itemsize, (W+2)*itemsize, itemsize)
    shape = (N, H, W, C, 3, 3)
    strides_patch = (strides[0], strides[2], strides[3], strides[1], strides[2], strides[3])
    patches = np.lib.stride_tricks.as_strided(features_pad, shape=shape, strides=strides_patch)
    
    # Reshape to (N*H*W, C*3*3) = (512000, 576)
    X = patches.reshape(N * H * W, C * 3 * 3)
    
    # Target: y shape (N, 1, H, W) -> flatten to (N*H*W, 1)
    y_flat = y[:, 0, :, :].reshape(-1, 1)
    
    # Add bias column
    X_bias = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)
    
    # Normal equations: A = X_bias^T X_bias, b = X_bias^T y_flat
    A = X_bias.T @ X_bias
    b = X_bias.T @ y_flat
    
    # Small ridge for numerical stability
    ridge = 1e-4
    A += ridge * np.eye(A.shape[0], dtype=np.float32)
    
    # Solve linear system
    try:
        theta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Extract output weights and bias
    w_out = theta[:C * 3 * 3].reshape(1, C, 3, 3).astype(np.float32)
    b_out = theta[C * 3 * 3].astype(np.float32)
    
    # Clip to safe range
    w_out = np.clip(w_out, -1000.0, 1000.0)
    b_out = np.clip(b_out, -1000.0, 1000.0)
    
    # Update weights
    w['out.weight'] = w_out
    w['out.bias'] = np.array([b_out], dtype=np.float32)
    
    # Diagnostics
    diagnostics = {
        'out_weight_mean': float(np.mean(w_out)),
        'out_weight_std': float(np.std(w_out)),
        'out_weight_max_abs': float(np.max(np.abs(w_out))),
        'out_bias': float(b_out),
        'ridge': ridge,
        'num_samples': int(X.shape[0])
    }
    
    return {"weights": w, "diagnostics": diagnostics}
