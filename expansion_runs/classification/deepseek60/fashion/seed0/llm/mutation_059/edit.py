import numpy as np

def edit(w, f):
    # Extract features and labels
    X = f["features"].astype(np.float64)          # (2000, 1568)
    y = f["y"].astype(np.int64)                   # (2000,)
    N, D = X.shape
    C = 10

    # Add bias column
    X_aug = np.hstack([X, np.ones((N, 1))])       # (2000, 1569)

    # One-hot labels
    Y = np.zeros((N, C))
    Y[np.arange(N), y] = 1.0

    # Current head parameters as prior
    W0 = np.vstack([w["head.weight"].T, w["head.bias"][None, :]])  # (1569, 10)

    # Precompute for efficiency
    XtX = X_aug.T @ X_aug
    XtY = X_aug.T @ Y

    # Candidate lambdas (log-spaced)
    lambdas = [0.01, 0.1, 1.0, 10.0, 100.0, 316.0, 1000.0, 10000.0]

    # 5-fold cross-validation
    n_folds = 5
    fold_size = N // n_folds
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)

    best_lambda = None
    best_cv_acc = -1.0

    for lam in lambdas:
        accs = []
        for fold in range(n_folds):
            val_idx = indices[fold * fold_size:(fold + 1) * fold_size]
            train_idx = np.setdiff1d(indices, val_idx)

            X_train = X_aug[train_idx]
            Y_train = Y[train_idx]
            X_val = X_aug[val_idx]
            Y_val = Y[val_idx]

            # Solve ridge with prior W0
            A = X_train.T @ X_train + lam * np.eye(D + 1)
            b = X_train.T @ Y_train + lam * W0
            W_fold = np.linalg.solve(A, b)

            logits_val = X_val @ W_fold
            preds = np.argmax(logits_val, axis=1)
            true = np.argmax(Y_val, axis=1)
            acc = np.mean(preds == true)
            accs.append(acc)

        cv_acc = np.mean(accs)
        if cv_acc > best_cv_acc:
            best_cv_acc = cv_acc
            best_lambda = lam

    # Fit on full data with best lambda
    lam = best_lambda
    A = XtX + lam * np.eye(D + 1)
    b = XtY + lam * W0
    W_final = np.linalg.solve(A, b)

    # Update head
    w["head.weight"] = W_final[:D, :].T.astype(np.float32)
    w["head.bias"] = W_final[D, :].astype(np.float32)

    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_cv_acc),
        "head_weight_norm": float(np.linalg.norm(w["head.weight"])),
        "head_bias_norm": float(np.linalg.norm(w["head.bias"])),
        "head_weight_max_abs": float(np.max(np.abs(w["head.weight"]))),
        "head_bias_max_abs": float(np.max(np.abs(w["head.bias"]))),
    }

    return {"weights": w, "diagnostics": diagnostics}
