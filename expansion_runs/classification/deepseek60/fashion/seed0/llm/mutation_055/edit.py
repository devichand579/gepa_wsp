import numpy as np

def edit(w, f):
    # Extract full feedback features and labels
    X = f["features"].astype(np.float64)  # (2000, 1568)
    y = f["y"].astype(np.int64)           # (2000,)
    n, d = X.shape
    num_classes = 10

    # Standardize features (zero mean, unit variance)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    X_std = (X - mean) / std

    # Add bias term
    X_aug = np.hstack([X_std, np.ones((n, 1))])
    d_aug = d + 1

    # One-hot encode labels
    Y = np.zeros((n, num_classes))
    Y[np.arange(n), y] = 1.0

    # Cross-validation to select ridge lambda
    lambdas = np.logspace(-4, 4, 17)  # 1e-4 to 1e4
    best_lambda = None
    best_acc = -1.0

    n_folds = 5
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // n_folds

    for lam in lambdas:
        accs = []
        for fold in range(n_folds):
            val_idx = indices[fold*fold_size:(fold+1)*fold_size]
            train_idx = np.setdiff1d(indices, val_idx)

            X_train = X_aug[train_idx]
            Y_train = Y[train_idx]
            X_val = X_aug[val_idx]
            y_val = y[val_idx]

            # Ridge closed-form solution
            A = X_train.T @ X_train + lam * np.eye(d_aug)
            B = X_train.T @ Y_train
            W = np.linalg.solve(A, B)

            logits = X_val @ W
            preds = np.argmax(logits, axis=1)
            acc = np.mean(preds == y_val)
            accs.append(acc)

        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lambda = lam

    # Fit on all data with best lambda
    A = X_aug.T @ X_aug + best_lambda * np.eye(d_aug)
    B = X_aug.T @ Y
    W = np.linalg.solve(A, B)

    # Extract head weights and bias
    head_weight = W[:-1, :].T   # (10, 1568)
    head_bias = W[-1, :]        # (10,)

    # Update the head parameters
    w["head.weight"] = head_weight.astype(np.float32)
    w["head.bias"] = head_bias.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_acc),
        "head_weight_norm": float(np.linalg.norm(head_weight)),
        "head_bias_norm": float(np.linalg.norm(head_bias)),
        "head_weight_max_abs": float(np.max(np.abs(head_weight))),
        "head_bias_max_abs": float(np.max(np.abs(head_bias))),
    }

    return {"weights": w, "diagnostics": diagnostics}
