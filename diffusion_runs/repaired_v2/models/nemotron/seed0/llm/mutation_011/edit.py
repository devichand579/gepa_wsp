import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def edit(w, f):
    # Extract permitted feedback arrays
    features = f['features']                # [2000, 64, 16, 16]
    prediction = f['prediction']            # [2000, 1, 16, 16]
    y = f['y']                              # [2000, 1, 16, 16]
    
    # Current output layer parameters
    out_weight = w['out.weight'].copy()    # [1, 64, 3, 3]
    out_bias = float(w['out.bias'].copy()) # scalar
    
    # Compute per-pixel error
    error = prediction - y                  # [2000, 1, 16, 16]
    
    N, C, H, W = features.shape             # 2000, 64, 16, 16
    
    # Pad features with zeros to simulate padding=1 behavior
    # of the convolution (effective input becomes 18×18)
    features_padded = np.pad(features, ((0,0), (0,0), (1,1), (1,1)),
                             mode='constant', constant_values=0.0)
    # Shape: [2000, 64, 18, 18]
    
    # Build design matrix and target vector
    # Each row = flattened 3×3×64 patch at one (batch, h, w) position
    # Total rows: 2000 × 16 × 16 = 512,000
    X_rows = N * H * W                        # 512000
    X = np.zeros((X_rows, C * 3 * 3))       # [512000, 576]
    y_target = np.zeros(X_rows)
    
    idx = 0
    for b in range(N):
        for h in range(H):
            for w in range(W):
                # Extract 3×3 patch from padded features at column h, row w
                # (output position (h,w) uses padded input h:h+3, w:w+3)
                patch = features_padded[b, :, h:h+3, w:w+3]  # [64, 3, 3]
                X[idx, :] = patch.reshape(-1)                # [576]
                y_target[idx] = y[b, 0, h, w]               # target epsilon value
                idx += 1
    
    # Include bias term by adding column of ones
    X_with_bias = np.hstack([X, np.ones((X_rows, 1))])  # [512000, 577]
    
    # Closed-form least-squares: min ||X @ params - y||²
    # Normal equation: params = (XᵀX)⁻¹Xᵀy
    XtX = X_with_bias.T @ X_with_bias                    # [577, 577]
    Xty = X_with_bias.T @ y_target                       # [577,]
    params = np.linalg.solve(XtX, Xty)                   # [577,]
    
    # Split into weight delta and bias delta
    delta_weight_flat = params[:576]                     # [576,]
    delta_bias = params[576]                             # scalar
    
    # Reshape weight delta to [1, 64, 3, 3]
    delta_weight = delta_weight_flat.reshape(1, 64, 3, 3)
    
    # Update output layer weights (keep earlier layers fixed)
    new_out_weight = out_weight + delta_weight
    new_out_bias = out_bias + delta_bias
    
    # Preserve finite float32, clip to |val| <= 1000
    new_out_weight = np.clip(new_out_weight.astype(np.float32), -1000, 1000)
    new_out_bias = float(np.clip(np.float32(new_out_bias), -1000, 1000))
    
    # Construct updated weights dict (preserve all keys/shapes)
    w_updated = {k: v.copy() for k, v in w.items()}
    w_updated['out.weight'] = new_out_weight
    w_updated['out.bias'] = np.float32(new_out_bias)
    
    # Compute diagnostics
    # Predicted MSE improvement on the feedback batch
    # (using original features with new output weights)
    pred_new = np.zeros_like(prediction)
    for b in range(N):
        # Manual forward through output layer only
        # Pad features
        f_pad = np.pad(features[b:b+1], ((0,0), (0,0), (1,1), (1,1)),
                       mode='constant', constant_values=0.0)
        for h in range(H):
            for w in range(W):
                patch = f_pad[0, :, h:h+3, w:w+3]  # [64, 3, 3]
                pred_new[b, 0, h, w] = np.sum(new_out_weight[0] * patch) + new_out_bias
    
    # Per-batch MSE before and after
    mse_before = np.mean((prediction - y) ** 2)
    mse_after = np.mean((pred_new - y) ** 2)
    
    diagnostics = {
        'mse_before': float(mse_before),
        'mse_after': float(mse_after),
        'mse_reduction': float(mse_before - mse_after),
        'weight_delta_mean': float(np.mean(np.abs(delta_weight))),
        'bias_delta': float(delta_bias),
        'samples_used': int(X_rows)
    }
    
    return w_updated, {"weights": w_updated, "diagnostics": diagnostics}
