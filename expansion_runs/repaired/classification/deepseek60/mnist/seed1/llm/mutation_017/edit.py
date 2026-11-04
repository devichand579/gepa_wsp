import numpy as np

def edit(w, f):
    # Extract current fc2 parameters
    W0 = w["fc2.weight"].astype(np.float64)  # [10, 128]
    b0 = w["fc2.bias"].astype(np.float64)    # [10]
    
    # Get full feedback features and labels
    X = f["features"].astype(np.float64)     # [2000, 128]
    y = f["y"].astype(np.int64)              # [2000]
    
    # One-hot encode labels
    N = X.shape[0]
    Y = np.zeros((N, 10), dtype=np.float64)
    Y[np.arange(N), y] = 1.0
    
    # Augment features with bias column
    X_aug = np.concatenate([X, np.ones((N, 1))], axis=1)  # [2000, 129]
    
    # Current augmented weights: [10, 129]
    W0_aug = np.concatenate([W0, b0[:, None]], axis=1)  # [10, 129]
    
    # Ridge update with strong prior centered at current weights.
    # Objective: ||X_aug W^T - Y||^2 + lambda ||W - W0_aug||^2
    # Solution: W = (X_aug^T X_aug + lambda I)^{-1} (X_aug^T Y + lambda W0_aug)
    # Choose lambda to balance prior and data. With N=2000 and 128 features,
    # a lambda around 1000 gives the prior comparable weight to the data.
    lam = 1000.0
    
    XtX = X_aug.T @ X_aug          # [129, 129]
    XtY = X_aug.T @ Y              # [129, 10]
    
    # Solve for each class weight vector
    A = XtX + lam * np.eye(X_aug.shape[1])
    B = XtY + lam * W0_aug.T       # [129, 10]
    W_new = np.linalg.solve(A, B).T  # [10, 129]
    
    # Split back into weight and bias
    w_new = W_new[:, :128].astype(np.float32)
    b_new = W_new[:, 128].astype(np.float32)
    
    # Clip to safe range
    w_new = np.clip(w_new, -1000.0, 1000.0)
    b_new = np.clip(b_new, -1000.0, 1000.0)
    
    # Update weights
    w["fc2.weight"] = w_new
    w["fc2.bias"] = b_new
    
    # Diagnostics
    diagnostics = {
        "lambda": lam,
        "fc2_weight_mean": float(np.mean(w_new)),
        "fc2_weight_std": float(np.std(w_new)),
        "fc2_bias_mean": float(np.mean(b_new)),
        "fc2_bias_std": float(np.std(b_new)),
        "fc2_weight_change_norm": float(np.linalg.norm(w_new - W0.astype(np.float32))),
        "fc2_bias_change_norm": float(np.linalg.norm(b_new - b0.astype(np.float32))),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
