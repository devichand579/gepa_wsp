import numpy as np

def edit(w, f):
    # Extract feedback data
    X = f["features"].astype(np.float64)  # [2000, 128]
    y = f["y"].astype(np.int64)           # [2000]
    n, d = X.shape
    n_classes = 10

    # Current fc2
    W_old = w["fc2.weight"].astype(np.float64)  # [10, 128]
    b_old = w["fc2.bias"].astype(np.float64)    # [10]

    # Compute current accuracy on the full feedback pool
    logits_old = X @ W_old.T + b_old
    pred_old = np.argmax(logits_old, axis=1)
    acc_old = np.mean(pred_old == y)

    # Prepare one-hot targets
    Y = np.zeros((n, n_classes), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    # Standardize features (fit on all data for simplicity; CV will use training folds)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    X_std = (X - mu) / sigma

    # Candidate ridge lambdas
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]

    # 5-fold cross-validation to select lambda
    n_folds = 5
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    best_lambda = None
    best_cv_acc = -1.0
    best_W = None
    best_b = None

    for lam in lambdas:
        cv_accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            X_train = X_std[train_idx]
            Y_train = Y[train_idx]
            X_val = X_std[val_idx]
            y_val = y[val_idx]

            # Ridge regression: W = (X^T X + lam I)^-1 X^T Y
            # Add bias by augmenting with a column of ones
            X_train_aug = np.hstack([X_train, np.ones((len(train_idx), 1))])
            X_val_aug = np.hstack([X_val, np.ones((len(val_idx), 1))])
            d_aug = X_train_aug.shape[1]

            A = X_train_aug.T @ X_train_aug + lam * np.eye(d_aug)
            B = X_train_aug.T @ Y_train
            W_aug = np.linalg.solve(A, B)  # [d_aug, 10]

            logits_val = X_val_aug @ W_aug
            pred_val = np.argmax(logits_val, axis=1)
            acc = np.mean(pred_val == y_val)
            cv_accs.append(acc)

        mean_cv_acc = np.mean(cv_accs)
        if mean_cv_acc > best_cv_acc:
            best_cv_acc = mean_cv_acc
            best_lambda = lam

    # Fit final model on all data with best lambda
    X_aug = np.hstack([X_std, np.ones((n, 1))])
    d_aug = X_aug.shape[1]
    A = X_aug.T @ X_aug + best_lambda * np.eye(d_aug)
    B = X_aug.T @ Y
    W_aug = np.linalg.solve(A, B)  # [d_aug, 10]

    # Convert back to original feature space: logits = X_std @ W_std + b_std
    # X_std = (X - mu) / sigma
    # logits = ((X - mu)/sigma) @ W_std + b_std
    #        = X @ (W_std / sigma) + (b_std - mu @ (W_std / sigma))
    W_std = W_aug[:-1, :].T  # [10, 128]
    b_std = W_aug[-1, :]     # [10]

    W_new = W_std / sigma    # [10, 128]
    b_new = b_std - mu @ W_new.T  # [10]

    # Evaluate new classifier on full pool
    logits_new = X @ W_new.T + b_new
    pred_new = np.argmax(logits_new, axis=1)
    acc_new = np.mean(pred_new == y)

    # Only replace if new classifier is better on the full pool
    if acc_new > acc_old:
        w["fc2.weight"] = W_new.astype(np.float32)
        w["fc2.bias"] = b_new.astype(np.float32)
        method = "ridge_std_cv"
    else:
        method = "keep_original"

    # Diagnostics
    diagnostics = {
        "method": method,
        "acc_old": float(acc_old),
        "acc_new": float(acc_new),
        "best_lambda": float(best_lambda) if best_lambda is not None else None,
        "best_cv_acc": float(best_cv_acc) if best_cv_acc is not None else None,
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
    }

    return {"weights": w, "diagnostics": diagnostics}
