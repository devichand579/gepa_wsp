import numpy as np

def edit(w, f):
    # Extract existing final features and labels for the entire feedback pool
    X = f["features"].astype(np.float64)          # [4000, 32]
    y = f["y"].astype(np.int64)                   # [4000]
    
    # Build one-hot targets: class 0 -> [1,0], class 1 -> [0,1]
    Y = np.zeros((len(y), 2), dtype=np.float64)
    Y[np.arange(len(y)), y] = 1.0
    
    # Add a bias column to X for closed-form ridge regression
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)  # [4000, 33]
    
    # Ridge regression: solve (Xb^T Xb + lambda I) W = Xb^T Y
    # Use a small regularization to avoid numerical issues
    lam = 1e-3
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    B = Xb.T @ Y
    W = np.linalg.solve(A, B)                     # [33, 2]
    
    # Extract head weights and bias
    head_w = W[:-1, :].T.astype(np.float32)       # [2, 32]
    head_b = W[-1, :].astype(np.float32)          # [2]
    
    # Update the head parameters in w
    w["head.weight"] = head_w
    w["head.bias"] = head_b
    
    # Optionally, we can also zero out the transformer blocks to ensure the features
    # are not altered by any later changes? But we are not changing earlier layers,
    # so the features remain exactly as provided. However, to be safe and avoid any
    # unintended interactions, we can leave the other weights unchanged.
    
    # Diagnostics
    # Compute training accuracy on the feedback pool
    logits = Xb @ W
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == y)
    
    diagnostics = {
        "method": "ridge head on existing features",
        "lambda": lam,
        "train_accuracy": float(acc),
        "head_weight_norm": float(np.linalg.norm(head_w)),
        "head_bias": head_b.tolist(),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
