def edit(w, f):
    X = f["features"].astype(np.float64)  # (N, 1568)
    y = f["y"].astype(np.int64)           # (N,)
    N, D = X.shape
    C = 10
    
    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0
    
    # Center features and targets
    X_mean = X.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    Xc = X - X_mean
    Yc = Y - Y_mean
    
    # Ridge regression: solve (Xc^T Xc + lam I) W = Xc^T Yc
    lam = 1.0
    A = Xc.T @ Xc + lam * np.eye(D)
    B = Xc.T @ Yc
    W = np.linalg.solve(A, B)  # (D, C)
    b = Y_mean - X_mean @ W   # (C,)
    
    # Clip for safety
    W = np.clip(W, -1000.0, 1000.0)
    b = np.clip(b, -1000.0, 1000.0)
    
    # Update head
    w["head.weight"] = W.T.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)
    
    # Diagnostics
    logits = X @ w["head.weight"].T + w["head.bias"]
    pred = np.argmax(logits, axis=1)
    acc = float(np.mean(pred == y))
    diag = {
        "train_acc": acc,
        "lam": lam,
        "head_weight_abs_mean": float(np.mean(np.abs(w["head.weight"]))),
        "head_weight_max": float(np.max(np.abs(w["head.weight"]))),
        "head_bias_abs_mean": float(np.mean(np.abs(w["head.bias"])))
    }
    return {"weights": w, "diagnostics": diag}
