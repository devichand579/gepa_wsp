def edit(w, f):
    # Extract features and labels from the full feedback pool
    X = f["features"].astype(np.float64)          # (2000, 1568)
    y = f["y"].astype(np.int64)                   # (2000,)
    N, D = X.shape

    # One-hot encode labels
    Y = np.eye(10, dtype=np.float64)[y]           # (2000, 10)

    # Standardize features (helps conditioning and ridge regularization)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    X_norm = (X - mean) / std

    # Augment with a bias column
    X_aug = np.hstack([X_norm, np.ones((N, 1), dtype=np.float64)])  # (2000, 1569)

    # Ridge regression closed-form solution
    lam = 1e-3
    XTX = X_aug.T @ X_aug
    XTY = X_aug.T @ Y
    W_aug = np.linalg.solve(XTX + lam * np.eye(D + 1), XTY)         # (1569, 10)

    W_norm = W_aug[:D, :].T          # (10, 1568)
    b_norm = W_aug[D, :]             # (10,)

    # Convert back to original feature scale
    W = W_norm / std                 # (10, 1568)
    b = b_norm - (mean / std) @ W_norm.T  # (10,)

    # Update head weights and biases
    w["head.weight"] = np.clip(W.astype(np.float32), -1000, 1000)
    w["head.bias"] = np.clip(b.astype(np.float32), -1000, 1000)

    # Diagnostics
    logits = X @ w["head.weight"].T + w["head.bias"]
    pred = np.argmax(logits, axis=1)
    acc = np.mean(pred == y)
    diag = {
        "train_accuracy": float(acc),
        "head_weight_norm": float(np.linalg.norm(w["head.weight"])),
        "head_bias_norm": float(np.linalg.norm(w["head.bias"])),
        "lambda": lam
    }
    return {"weights": w, "diagnostics": diag}
