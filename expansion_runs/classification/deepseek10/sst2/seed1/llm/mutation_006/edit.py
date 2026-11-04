import numpy as np

def edit(w, f):
    # Extract features and labels from the full feedback pool
    X = f["features"].astype(np.float64)  # [4000, 32]
    y = f["y"].astype(np.int64)           # [4000]
    
    # Build one-hot targets for ridge regression (class 0 and class 1)
    N = X.shape[0]
    Y = np.zeros((N, 2), dtype=np.float64)
    Y[np.arange(N), y] = 1.0
    
    # Center features and targets to fit bias separately
    X_mean = X.mean(axis=0, keepdims=True)
    Y_mean = Y.mean(axis=0, keepdims=True)
    Xc = X - X_mean
    Yc = Y - Y_mean
    
    # Ridge regression: solve (Xc^T Xc + lambda I) W = Xc^T Yc
    # Use a small lambda for numerical stability; features are already well-scaled.
    lam = 1e-3
    XtX = Xc.T @ Xc
    XtY = Xc.T @ Yc
    W = np.linalg.solve(XtX + lam * np.eye(XtX.shape[0]), XtY)  # [32, 2]
    
    # Bias: b = Y_mean - X_mean @ W
    b = Y_mean - X_mean @ W  # [1, 2]
    
    # The head computes logits = features @ head.weight.T + head.bias
    # So head.weight should be W.T (shape [2, 32]) and head.bias should be b[0] (shape [2])
    w["head.weight"] = W.T.astype(np.float32)
    w["head.bias"] = b[0].astype(np.float32)
    
    # Optional: ensure other parameters remain unchanged (they are already zero/identity)
    # We do not modify any other weights.
    
    # Diagnostics
    diagnostics = {
        "ridge_lambda": lam,
        "head_weight_norm": float(np.linalg.norm(w["head.weight"])),
        "head_bias": w["head.bias"].tolist(),
        "train_accuracy": float(np.mean(np.argmax(X @ W + b, axis=1) == y)),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
