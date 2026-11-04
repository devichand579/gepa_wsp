import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def edit(w, f):
    # Extract full training features and targets
    features = f['features']          # (2000, 64, 16, 16)
    y = f['y']                        # (2000, 1, 16, 16)
    
    N, C, H, W = features.shape
    # Pad spatially by 1 to handle 3x3 convolution with zero padding
    feat_pad = np.pad(features, ((0,0),(0,0),(1,1),(1,1)), mode='constant')
    
    # Number of parameters: C*3*3 weights + 1 bias
    n_params = C * 9 + 1
    XtX = np.zeros((n_params, n_params), dtype=np.float64)
    Xty = np.zeros((n_params, 1), dtype=np.float64)
    
    # Process in batches to keep memory low
    batch_size = 200
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        fb = feat_pad[start:end]                     # (B, C, H+2, W+2)
        # Extract 3x3 patches: (B, C, H, W, 3, 3)
        patches = sliding_window_view(fb, (3,3), axis=(2,3))
        # Rearrange to (B, H, W, C, 3, 3)
        patches = patches.transpose(0, 2, 3, 1, 4, 5)
        # Flatten to (B*H*W, C*9)
        X_batch = patches.reshape(-1, C*9)
        # Add bias column
        X_batch = np.concatenate([X_batch, np.ones((X_batch.shape[0], 1), dtype=X_batch.dtype)], axis=1)
        
        yb = y[start:end].reshape(-1, 1)
        
        XtX += X_batch.T @ X_batch
        Xty += X_batch.T @ yb
    
    # Ridge regularization for numerical stability
    ridge = 1e-6 * np.trace(XtX) / n_params
    XtX += ridge * np.eye(n_params)
    
    # Solve normal equations
    w_new = np.linalg.solve(XtX, Xty).flatten()
    
    # Extract weights and bias
    out_weight = w_new[:C*9].reshape(C, 3, 3)
    out_weight = out_weight.reshape(1, C, 3, 3)
    out_bias = w_new[C*9:C*9+1]
    
    # Update only the output layer
    w['out.weight'] = out_weight.astype(np.float32)
    w['out.bias'] = out_bias.astype(np.float32)
    
    # Diagnostics
    diagnostics = {
        'out_weight_norm': float(np.linalg.norm(out_weight)),
        'out_bias': float(out_bias[0]),
        'ridge': float(ridge),
        'n_params': n_params
    }
    
    return {"weights": w, "diagnostics": diagnostics}
