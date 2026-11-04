import numpy as np

def edit(w, f):
    # Extract current head parameters
    W_old = w["head.weight"].astype(np.float64)  # [10, 1568]
    b_old = w["head.bias"].astype(np.float64)    # [10]

    # Feedback data
    X = f["features"].astype(np.float64)         # [2000, 1568]
    y = f["y"].astype(np.int64)                  # [2000]
    N = X.shape[0]

    # One-hot targets
    Y = np.zeros((N, 10), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Augment features with bias column
    X_aug = np.hstack([X, np.ones((N, 1), dtype=np.float64)])  # [2000, 1569]
    W_aug_old = np.hstack([W_old, b_old[:, None]])             # [10, 1569]

    # Precompute full X_aug^T X_aug and X_aug^T Y
    XtX_full = X_aug.T @ X_aug          # [1569, 1569]
    XtY_full = X_aug.T @ Y              # [1569, 10]

    # 5-fold cross-validation to select lambda
    n_folds = 5
    fold_size = N // n_folds
    indices = np.arange(N)
    # Use a fixed seed for reproducibility
    rng = np.random.RandomState(42)
    rng.shuffle(indices)

    lambdas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    best_lambda = None
    best_acc = -1.0

    for lam in lambdas:
        acc_sum = 0.0
        for k in range(n_folds):
            val_idx = indices[k*fold_size:(k+1)*fold_size]
            train_idx = np.setdiff1d(indices, val_idx, assume_unique=False)
            # Compute train XtX and XtY by subtracting validation contribution
            XtX_val = X_aug[val_idx].T @ X_aug[val_idx]
            XtY_val = X_aug[val_idx].T @ Y[val_idx]
            XtX_train = XtX_full - XtX_val
            XtY_train = XtY_full - XtY_val

            # Solve ridge: (XtX + lam I) W_aug^T = XtY + lam W_aug_old^T
            A = XtX_train + lam * np.eye(X_aug.shape[1])
            B = XtY_train + lam * W_aug_old.T
            W_aug_new = np.linalg.solve(A, B).T  # [10, 1569]

            # Evaluate on validation fold
            logits_val = X_aug[val_idx] @ W_aug_new.T
            preds = np.argmax(logits_val, axis=1)
            acc = np.mean(preds == y[val_idx])
            acc_sum += acc
        avg_acc = acc_sum / n_folds
        if avg_acc > best_acc:
            best_acc = avg_acc
            best_lambda = lam

    # If no lambda improved, fall back to a tiny shrink
    if best_lambda is None:
        best_lambda = 0.001

    # Final fit on full data with best lambda
    A = XtX_full + best_lambda * np.eye(X_aug.shape[1])
    B = XtY_full + best_lambda * W_aug_old.T
    W_aug_new = np.linalg.solve(A, B).T  # [10, 1569]

    # Extract new head weights and bias
    W_new = W_aug_new[:, :1568].astype(np.float32)
    b_new = W_aug_new[:, 1568].astype(np.float32)

    # Clip to safe range
    W_new = np.clip(W_new, -1000.0, 1000.0)
    b_new = np.clip(b_new, -1000.0, 1000.0)

    # Update weights
    w["head.weight"] = W_new
    w["head.bias"] = b_new

    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lambda),
        "best_cv_accuracy": float(best_acc),
        "head_weight_norm": float(np.linalg.norm(W_new)),
        "head_bias_norm": float(np.linalg.norm(b_new)),
        "delta_W_norm": float(np.linalg.norm(W_new - W_old)),
        "delta_b_norm": float(np.linalg.norm(b_new - b_old)),
    }

    return {"weights": w, "diagnostics": diagnostics}
