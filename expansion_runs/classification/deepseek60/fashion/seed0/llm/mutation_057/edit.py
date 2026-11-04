import numpy as np

def edit(w, f):
    # Extract features and labels from the full feedback pool
    X = f["features"].astype(np.float64)          # [2000, 1568]
    y = f["y"].astype(np.int64)                   # [2000]
    n, d = X.shape
    K = 10

    # One-hot labels
    Y = np.zeros((n, K), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    # Standardize features (zero mean, unit variance)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    Xs = (X - mu) / sigma

    # 5-fold cross-validation to select ridge lambda
    n_folds = 5
    indices = np.arange(n)
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    lambdas = np.logspace(-4, 4, 17)
    best_lambda = None
    best_cv_acc = -1.0

    for lam in lambdas:
        accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            X_tr = Xs[train_idx]
            Y_tr = Y[train_idx]
            X_va = Xs[val_idx]
            y_va = y[val_idx]

            # Ridge regression: solve (X^T X + lam I) W = X^T Y
            # Add bias by augmenting with a column of ones
            X_tr_aug = np.hstack([X_tr, np.ones((len(train_idx), 1))])
            X_va_aug = np.hstack([X_va, np.ones((len(val_idx), 1))])
            d_aug = X_tr_aug.shape[1]

            # Use normal equations with regularization
            A = X_tr_aug.T @ X_tr_aug + lam * np.eye(d_aug)
            B = X_tr_aug.T @ Y_tr
            W_aug = np.linalg.solve(A, B)   # [d_aug, K]

            # Predict
            scores = X_va_aug @ W_aug
            preds = np.argmax(scores, axis=1)
            acc = np.mean(preds == y_va)
            accs.append(acc)

        mean_acc = np.mean(accs)
        if mean_acc > best_cv_acc:
            best_cv_acc = mean_acc
            best_lambda = lam

    # Fit final model on all data using best_lambda
    X_aug = np.hstack([Xs, np.ones((n, 1))])
    d_aug = X_aug.shape[1]
    A = X_aug.T @ X_aug + best_lambda * np.eye(d_aug)
    B = X_aug.T @ Y
    W_aug = np.linalg.solve(A, B)   # [d_aug, K]

    # Unstandardize: W_orig = W_std / sigma, b_orig = b_std - sum(W_std * mu / sigma)
    W_std = W_aug[:-1, :]           # [d, K]
    b_std = W_aug[-1, :]            # [K]

    W_orig = W_std / sigma[:, None]                 # [d, K]
    b_orig = b_std - (W_std * mu[:, None] / sigma[:, None]).sum(axis=0)  # [K]

    # Transpose to match head.weight shape [K, d]
    head_weight = W_orig.T.astype(np.float32)
    head_bias = b_orig.astype(np.float32)

    # Clip to safe range
    head_weight = np.clip(head_weight, -1000.0, 1000.0)
    head_bias = np.clip(head_bias, -1000.0, 1000.0)

    # Update weights
    w["head.weight"] = head_weight
    w["head.bias"] = head_bias

    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_cv_acc),
        "head_weight_norm": float(np.linalg.norm(head_weight)),
        "head_bias_norm": float(np.linalg.norm(head_bias)),
        "head_weight_max_abs": float(np.max(np.abs(head_weight))),
        "head_bias_max_abs": float(np.max(np.abs(head_bias))),
    }
    return {"weights": w, "diagnostics": diagnostics}
