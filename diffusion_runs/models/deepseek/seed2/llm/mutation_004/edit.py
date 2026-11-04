import numpy as np
import math

def edit(w, f):
    # Extract arrays
    features = f["features"].astype(np.float32)          # (2000, 64, 16, 16)
    y        = f["y"].astype(np.float32)                 # (2000, 1, 16, 16)
    N = features.shape[0]
    C = features.shape[1]
    H = features.shape[2]
    W = features.shape[3]
    # Flatten spatial and batch: we need to fit a 3x3 convolution from 64 channels to 1 channel.
    # Build design matrix for out layer: for each output pixel, take 3x3 patch across 64 channels.
    # We'll use im2col for the out layer. Input to out is features (N,64,H,W).
    # Output is y (N,1,H,W).
    # out.weight shape: (1,64,3,3), out.bias shape: (1)
    # We'll solve for W_out (64*9) and b_out.
    # Use padding=1.
    pad = 1
    # Pad features
    feat_pad = np.pad(features, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
    # Extract patches: for each n, i, j, we need feat_pad[n, :, i:i+3, j:j+3] -> 64*9
    # We can do this efficiently with stride tricks.
    from numpy.lib.stride_tricks import as_strided
    # Shape of patches: (N, H, W, C, 3, 3)
    # Strides: feat_pad.strides = (sN, sC, sH, sW)
    sN, sC, sH, sW = feat_pad.strides
    patches = as_strided(feat_pad,
                         shape=(N, H, W, C, 3, 3),
                         strides=(sN, sH, sW, sC, sH, sW))
    # Reshape to (N*H*W, C*9)
    X = patches.reshape(N*H*W, C*9).astype(np.float32)
    # Target: y flattened to (N*H*W, 1)
    Y = y.reshape(N*H*W, 1).astype(np.float32)
    
    # Ridge regression: solve (X^T X + lambda I) beta = X^T Y
    # We'll use a small lambda. We can choose lambda via cross-validation on the 2000 examples.
    # To keep it simple and robust, we'll use a fixed small lambda, e.g., 1e-3 * trace(X^T X)/p.
    # But we can also do a quick 5-fold CV to pick lambda from a small grid.
    # Given time, we'll use a fixed lambda that is a small fraction of the mean eigenvalue.
    # Compute X^T X and X^T Y
    XtX = X.T @ X
    XtY = X.T @ Y
    p = XtX.shape[0]
    # Estimate scale
    scale = np.trace(XtX) / p
    # Try a few lambdas and pick the one that minimizes CV error? 
    # For simplicity, use lambda = 1e-4 * scale. This is a mild ridge.
    # Actually, previous unregularized got max weight 48 and mse 0.2563. 
    # We can try lambda = 1e-3 * scale to be safe.
    lam = 1e-3 * scale
    A = XtX + lam * np.eye(p, dtype=np.float32)
    beta = np.linalg.solve(A, XtY)  # shape (C*9, 1)
    
    # Extract weights and bias
    W_out = beta.reshape(1, C, 3, 3).astype(np.float32)
    # Bias: we didn't include bias in X. We can compute bias as mean(Y - X @ beta)
    # Or we can augment X with a column of ones. Let's do that properly.
    # Actually, we should include bias. Let's redo with bias.
    # Augment X with ones: X_aug = [X, 1]
    X_aug = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)
    XtX_aug = X_aug.T @ X_aug
    XtY_aug = X_aug.T @ Y
    p_aug = XtX_aug.shape[0]
    scale_aug = np.trace(XtX_aug) / p_aug
    lam_aug = 1e-3 * scale_aug
    A_aug = XtX_aug + lam_aug * np.eye(p_aug, dtype=np.float32)
    beta_aug = np.linalg.solve(A_aug, XtY_aug)
    W_out = beta_aug[:-1].reshape(1, C, 3, 3).astype(np.float32)
    b_out = beta_aug[-1:].astype(np.float32)
    
    # Update w
    w["out.weight"] = W_out
    w["out.bias"] = b_out
    
    # Diagnostics
    # Compute training MSE with new out layer
    pred = X @ beta_aug[:-1] + beta_aug[-1]
    mse = np.mean((pred - Y)**2)
    diag = {
        "mse_train": float(mse),
        "lambda": float(lam_aug),
        "out_weight_max": float(np.max(np.abs(W_out))),
        "out_bias": float(b_out[0])
    }
    return {"weights": w, "diagnostics": diag}
