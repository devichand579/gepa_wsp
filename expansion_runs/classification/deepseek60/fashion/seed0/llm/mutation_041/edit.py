def edit(w, f):
    X = f["features"]          # (2000, 1568)
    y = f["y"]                 # (2000,)
    n, d = X.shape

    # One-hot encode labels
    Y = np.zeros((n, 10), dtype=np.float32)
    Y[np.arange(n), y] = 1.0

    # Standardize features (add small epsilon to avoid division by zero)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-5
    X_scaled = (X - X_mean) / X_std

    # Center targets
    Y_mean = Y.mean(axis=0)
    Y_centered = Y - Y_mean

    # Ridge regression on standardized features
    lambda_reg = 10.0
    XtX = X_scaled.T @ X_scaled
    XtY = X_scaled.T @ Y_centered
    W_scaled = np.linalg.solve(XtX + lambda_reg * np.eye(d), XtY)

    # Convert weights back to original feature space
    W = W_scaled / X_std[:, None]
    bias = Y_mean - X_mean @ W

    # Update only the head
    w["head.weight"] = W.T.astype(np.float32)
    w["head.bias"] = bias.astype(np.float32)

    diagnostics = {
        "head_weight_norm": float(np.linalg.norm(w["head.weight"])),
        "head_bias_norm": float(np.linalg.norm(w["head.bias"])),
        "lambda": lambda_reg,
        "feature_mean": float(X_mean.mean()),
        "feature_std": float(X_std.mean())
    }
    return {"weights": w, "diagnostics": diagnostics}
