import numpy as np

def edit(w, f):
    # Extract features and labels from the full feedback pool
    X = f["features"].astype(np.float64)          # (2000, 1568)
    y = f["y"].astype(np.int64)                   # (2000,)
    N, D = X.shape
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Add bias column
    X_aug = np.hstack([X, np.ones((N, 1), dtype=np.float64)])  # (2000, 1569)
    D_aug = X_aug.shape[1]

    # Full normal equations
    XtX = X_aug.T @ X_aug
    XtY = X_aug.T @ Y

    # Cross-validation to select ridge lambda
    rng = np.random.default_rng(42)
    indices = rng.permutation(N)
    n_folds = 5
    folds = np.array_split(indices, n_folds)

    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    best_lambda = 1e-2
    best_acc = -1.0

    for lam in lambdas:
        acc_sum = 0.0
        for val_idx in folds:
            # Compute validation contributions
            X_val = X_aug[val_idx]
            Y_val = Y[val_idx]
            XtX_val = X_val.T @ X_val
            XtY_val = X_val.T @ Y_val

            # Training normal equations
            XtX_train = XtX - XtX_val
            XtY_train = XtY - XtY_val

            # Ridge solve
            A = XtX_train + lam * np.eye(D_aug)
            W = np.linalg.solve(A, XtY_train)

            # Evaluate on validation
            logits_val = X_val @ W
            pred = np.argmax(logits_val, axis=1)
            acc_sum += np.mean(pred == y[val_idx])

        avg_acc = acc_sum / n_folds
        if avg_acc > best_acc:
            best_acc = avg_acc
            best_lambda = lam

    # Final fit on all data with best lambda
    A = XtX + best_lambda * np.eye(D_aug)
    W_full = np.linalg.solve(A, XtY)

    # Extract weight and bias
    new_head_weight = W_full[:-1, :].T   # (10, 1568)
    new_head_bias = W_full[-1, :]        # (10,)

    # Optional: scale to keep values within [-1000, 1000]
    max_abs = max(np.max(np.abs(new_head_weight)), np.max(np.abs(new_head_bias)))
    if max_abs > 1000.0:
        scale = 1000.0 / max_abs
        new_head_weight *= scale
        new_head_bias *= scale

    # Update weights
    w["head.weight"] = new_head_weight.astype(np.float32)
    w["head.bias"] = new_head_bias.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_acc),
        "head_weight_mean": float(np.mean(new_head_weight)),
        "head_weight_std": float(np.std(new_head_weight)),
        "head_weight_min": float(np.min(new_head_weight)),
        "head_weight_max": float(np.max(new_head_weight)),
        "head_bias_mean": float(np.mean(new_head_bias)),
        "head_bias_std": float(np.std(new_head_bias)),
    }

    return {"weights": w, "diagnostics": diagnostics}
