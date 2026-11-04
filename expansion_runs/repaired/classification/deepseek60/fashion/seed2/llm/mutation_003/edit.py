def edit(w, f):
    import numpy as np

    # Extract frozen features and labels from the full feedback pool
    X = f["features"].astype(np.float64)          # (2000, 1568)
    y = f["y"].astype(np.int64)                   # (2000,)
    n, d = X.shape

    # Augment features with a bias column
    Xb = np.concatenate([X, np.ones((n, 1), dtype=np.float64)], axis=1)  # (2000, 1569)

    # One-hot targets
    Y = np.zeros((n, 10), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    # Precompute full normal equations
    XtX = Xb.T @ Xb          # (1569, 1569)
    XtY = Xb.T @ Y           # (1569, 10)

    # Stratified 5-fold cross-validation to select ridge lambda
    n_folds = 5
    rng = np.random.default_rng(42)
    val_indices = [[] for _ in range(n_folds)]
    for c in range(10):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        splits = np.array_split(idx, n_folds)
        for i in range(n_folds):
            val_indices[i].append(splits[i])
    val_indices = [np.concatenate(v) for v in val_indices]

    lambdas = [0.1, 1.0, 10.0, 100.0, 1000.0]
    best_acc = -1.0
    best_lambda = 100.0

    for lam in lambdas:
        accs = []
        for i in range(n_folds):
            val_idx = val_indices[i]
            X_val = Xb[val_idx]
            Y_val = Y[val_idx]

            # Training normal equations = full - validation part
            XtX_train = XtX - X_val.T @ X_val
            XtY_train = XtY - X_val.T @ Y_val

            reg = lam * np.eye(d + 1)
            reg[-1, -1] = 1e-6  # do not regularize bias heavily

            W = np.linalg.solve(XtX_train + reg, XtY_train)
            pred = X_val @ W
            pred_class = np.argmax(pred, axis=1)
            acc = np.mean(pred_class == y[val_idx])
            accs.append(acc)

        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lambda = lam

    # Final closed-form ridge fit on all feedback data
    reg = best_lambda * np.eye(d + 1)
    reg[-1, -1] = 1e-6
    W_full = np.linalg.solve(XtX + reg, XtY)   # (1569, 10)

    # Update only the head; conv layers remain untouched
    w["head.weight"] = W_full[:-1].T.astype(np.float32)   # (10, 1568)
    w["head.bias"]   = W_full[-1].astype(np.float32)      # (10,)

    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_acc": float(best_acc),
        "head_weight_std": float(np.std(w["head.weight"])),
        "head_bias_mean": float(np.mean(w["head.bias"]))
    }
    return {"weights": w, "diagnostics": diagnostics}
