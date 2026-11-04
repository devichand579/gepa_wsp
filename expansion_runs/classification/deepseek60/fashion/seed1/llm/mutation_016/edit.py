import numpy as np

def edit(w, f):
    # Extract data
    X = f["features"].astype(np.float64)          # (2000, 1568)
    y = f["y"].astype(np.int64)                   # (2000,)
    W = w["head.weight"].astype(np.float64)       # (10, 1568)
    b = w["head.bias"].astype(np.float64)         # (10,)
    
    N, D = X.shape
    C = 10
    
    # Current logits and probabilities
    Z = X @ W.T + b                               # (N, 10)
    # Softmax
    Z_max = np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Z - Z_max)
    P = expZ / np.sum(expZ, axis=1, keepdims=True)  # (N, 10)
    
    # One-hot labels
    Y = np.zeros((N, C))
    Y[np.arange(N), y] = 1.0
    
    # Gradient of CE w.r.t. logits: g = P - Y
    G = P - Y                                     # (N, 10)
    
    # Compute Newton step for each sample: delta_z = -H^{-1} g
    # H = diag(P) - P P^T, which is singular (rank 9). Use pseudo-inverse.
    delta_z = np.zeros((N, C))
    for i in range(N):
        p = P[i]
        H = np.diag(p) - np.outer(p, p)
        # Solve H delta = -g using least squares (pseudo-inverse)
        # Since H is symmetric positive semi-definite, we can use lstsq
        delta_z[i] = -np.linalg.lstsq(H, G[i], rcond=None)[0]
    
    # Now we want to find delta_W, delta_b such that X @ delta_W.T + delta_b ≈ delta_z
    # Augment X with bias column
    X_aug = np.hstack([X, np.ones((N, 1))])       # (N, D+1)
    
    # Ridge regression: solve for Theta (C, D+1)
    # Theta = (X_aug^T X_aug + alpha I)^{-1} X_aug^T delta_z
    # Use a small alpha to regularize. Since delta_z is small, alpha can be small.
    alpha = 1e-3
    A = X_aug.T @ X_aug + alpha * np.eye(D + 1)
    B = X_aug.T @ delta_z
    Theta = np.linalg.solve(A, B).T               # (C, D+1)
    
    delta_W = Theta[:, :D]
    delta_b = Theta[:, D]
    
    # Update weights
    W_new = W + delta_W
    b_new = b + delta_b
    
    # Clip to ensure finite and within bounds
    W_new = np.clip(W_new, -1000, 1000).astype(np.float32)
    b_new = np.clip(b_new, -1000, 1000).astype(np.float32)
    
    w["head.weight"] = W_new
    w["head.bias"] = b_new
    
    # Diagnostics
    Z_new = X @ W_new.T + b_new
    Z_new_max = np.max(Z_new, axis=1, keepdims=True)
    expZ_new = np.exp(Z_new - Z_new_max)
    P_new = expZ_new / np.sum(expZ_new, axis=1, keepdims=True)
    acc_new = np.mean(np.argmax(Z_new, axis=1) == y)
    ce_new = -np.mean(np.log(P_new[np.arange(N), y] + 1e-12))
    
    diagnostics = {
        "acc_before": float(np.mean(np.argmax(Z, axis=1) == y)),
        "ce_before": float(-np.mean(np.log(P[np.arange(N), y] + 1e-12))),
        "acc_after": float(acc_new),
        "ce_after": float(ce_new),
        "delta_W_norm": float(np.linalg.norm(delta_W)),
        "delta_b_norm": float(np.linalg.norm(delta_b)),
        "alpha": alpha
    }
    
    return {"weights": w, "diagnostics": diagnostics}
