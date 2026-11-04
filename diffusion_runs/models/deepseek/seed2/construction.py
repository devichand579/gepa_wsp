import numpy as np

def edit(w, f):
    # Hypothesis: The current final-layer weights are random and produce near-zero
    # epsilon predictions (MSE ~ 1.0, which equals the variance of the target).
    # The final hidden features f['features'] are cached from the parent network.
    # Since the output layer is a linear 3x3 convolution, we can solve exactly for
    # its weights and bias by least squares on all 2000 training examples.
    # This is a closed-form linear regression with 577 parameters and 512000 equations.
    # Predicted effect: MSE on the training set will drop substantially (likely to < 0.5),
    # because the optimal linear readout of the existing features is fitted directly.
    
    F = f['features']          # (2000, 64, 16, 16)
    y = f['y']                 # (2000, 1, 16, 16)
    N, C, H, W = F.shape
    
    # Pad features by 1 on each spatial side for 3x3 convolution with padding=1
    F_pad = np.pad(F, ((0,0), (0,0), (1,1), (1,1)), mode='constant')
    
    # Build design matrix: for each output pixel, concatenate the 3x3 neighbourhood
    # across all 64 channels.  Order of offsets: (di, dj) in {-1,0,1} x {-1,0,1}.
    X_parts = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            # shifted feature map: (N, C, H, W)
            F_shift = F_pad[:, :, 1+di:1+di+H, 1+dj:1+dj+W]
            # flatten to (N*H*W, C)
            F_shift_flat = F_shift.transpose(0, 2, 3, 1).reshape(-1, C)
            X_parts.append(F_shift_flat)
    
    X = np.concatenate(X_parts, axis=1)          # (N*H*W, 576)
    X = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)  # (N*H*W, 577)
    
    y_flat = y.transpose(0, 2, 3, 1).reshape(-1)  # (N*H*W,)
    
    # Normal equations with tiny ridge for numerical stability
    XTX = X.T @ X
    XTy = X.T @ y_flat
    ridge = 1e-6 * np.eye(XTX.shape[0], dtype=np.float32)
    w_sol = np.linalg.solve(XTX + ridge, XTy)
    
    # Extract out.weight (1,64,3,3) and out.bias (1,)
    out_weight = np.zeros((1, C, 3, 3), dtype=np.float32)
    idx = 0
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            out_weight[0, :, di+1, dj+1] = w_sol[idx:idx+C]
            idx += C
    out_bias = np.array([w_sol[-1]], dtype=np.float32)
    
    # Update weights in-place
    w['out.weight'] = out_weight.astype(np.float32)
    w['out.bias'] = out_bias
    
    # Diagnostics
    pred = X @ w_sol
    mse_new = np.mean((pred - y_flat) ** 2)
    mse_old = np.mean((f['prediction'].reshape(-1) - y_flat) ** 2)
    
    return {
        "weights": w,
        "diagnostics": {
            "mse_old": float(mse_old),
            "mse_new": float(mse_new),
            "improvement": float(mse_old - mse_new),
            "out_weight_norm": float(np.linalg.norm(out_weight)),
            "out_bias": float(out_bias[0])
        }
    }
