import numpy as np

def edit(w, f):
    # Extract current head parameters
    W0 = w["head.weight"].astype(np.float64)   # (10, 1568)
    b0 = w["head.bias"].astype(np.float64)     # (10,)
    
    # Full feedback pool features and labels
    X = f["features"].astype(np.float64)       # (2000, 1568)
    y = f["y"].astype(np.int64)                # (2000,)
    N = X.shape[0]
    C = 10
    
    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0
    
    # Augment features with a bias column
    X_aug = np.concatenate([X, np.ones((N, 1), dtype=np.float64)], axis=1)  # (2000, 1569)
    
    # Prior parameters augmented
    W0_aug = np.concatenate([W0, b0[:, None]], axis=1)  # (10, 1569)
    
    # Ridge coefficient (selected from prior reflection)
    lam = 100.0
    
    # Solve normal equations: W_aug^T = (X_aug^T X_aug + lam I)^{-1} (X_aug^T Y + lam W0_aug^T)
    A = X_aug.T @ X_aug + lam * np.eye(X_aug.shape[1])
    B = X_aug.T @ Y + lam * W0_aug.T
    W_aug_T = np.linalg.solve(A, B)          # (1569, 10)
    W_aug = W_aug_T.T                        # (10, 1569)
    
    # Split back into weights and biases
    W_new = W_aug[:, :-1]
    b_new = W_aug[:, -1]
    
    # Write back to w, preserving float32 and shape
    w["head.weight"] = W_new.astype(np.float32)
    w["head.bias"] = b_new.astype(np.float32)
    
    # Optional diagnostics
    diagnostics = {
        "lambda": lam,
        "delta_W_norm": float(np.linalg.norm(W_new - W0)),
        "delta_b_norm": float(np.linalg.norm(b_new - b0)),
        "W_new_norm": float(np.linalg.norm(W_new)),
        "b_new_norm": float(np.linalg.norm(b_new)),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
