import numpy as np

def edit(w, f):
    # Extract arrays
    F = f["features"].astype(np.float64)      # [2000, 128]
    y = f["y"].astype(np.int64)               # [2000]
    logits = f["logits"].astype(np.float64)   # [2000, 10]
    N, D = F.shape
    C = 10

    # Helper: compute ridge correction given training indices and hyperparameters
    def compute_correction(train_idx, margin, lambda_reg):
        F_train = F[train_idx]
        y_train = y[train_idx]
        logits_train = logits[train_idx]
        N_train = len(train_idx)

        # Build target logit changes dL
        dL = np.zeros((N_train, C))
        for i in range(N_train):
            true = y_train[i]
            L = logits_train[i]
            other = L.copy()
            other[true] = -np.inf
            max_other = np.max(other)
            current_margin = L[true] - max_other
            if current_margin < margin:
                inc = margin - current_margin
                dL[i, true] = inc
                j_max = np.argmax(other)
                dL[i, j_max] = -inc / 2.0

        # Design matrix with intercept
        X = np.hstack([F_train, np.ones((N_train, 1))])
        # Uniform weights (could weight errors higher, but target already handles them)
        weights = np.ones(N_train)
        XtW = X.T * weights
        A = XtW @ X + lambda_reg * np.eye(D + 1)
        B = XtW @ dL
        W_sol = np.linalg.solve(A, B)          # [D+1, C]
        dW = W_sol[:D, :].T                    # [C, D]
        db = W_sol[D, :]                       # [C]
        return dW, db

    # 5-fold cross-validation to select margin and lambda
    np.random.seed(0)
    indices = np.random.permutation(N)
    n_folds = 5
    fold_size = N // n_folds
    margins = [0.5, 1.0, 1.5, 2.0]
    lambdas = [0.1, 0.5, 1.0, 2.0, 5.0]
    best_acc = -1.0
    best_params = (1.0, 1.0)

    for margin in margins:
        for lam in lambdas:
            accs = []
            for fold in range(n_folds):
                val_start = fold * fold_size
                val_end = (fold + 1) * fold_size if fold < n_folds - 1 else N
                val_idx = indices[val_start:val_end]
                train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
                dW, db = compute_correction(train_idx, margin, lam)
                F_val = F[val_idx]
                logits_val = logits[val_idx]
                y_val = y[val_idx]
                new_logits = logits_val + F_val @ dW.T + db
                pred_val = np.argmax(new_logits, axis=1)
                accs.append(np.mean(pred_val == y_val))
            mean_acc = np.mean(accs)
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_params = (margin, lam)

    # Fit on full pool with best hyperparameters
    margin, lam = best_params
    dW, db = compute_correction(np.arange(N), margin, lam)

    # Apply correction to fc2
    w["fc2.weight"] = (w["fc2.weight"].astype(np.float64) + dW).astype(np.float32)
    w["fc2.bias"] = (w["fc2.bias"].astype(np.float64) + db).astype(np.float32)

    # Diagnostics
    new_logits = logits + F @ dW.T + db
    new_pred = np.argmax(new_logits, axis=1)
    acc_before = np.mean(np.argmax(logits, axis=1) == y)
    acc_after = np.mean(new_pred == y)
    diagnostics = {
        "best_margin": float(margin),
        "best_lambda": float(lam),
        "cv_accuracy": float(best_acc),
        "full_accuracy_before": float(acc_before),
        "full_accuracy_after": float(acc_after),
        "dW_norm": float(np.linalg.norm(dW)),
        "db_norm": float(np.linalg.norm(db))
    }

    return {"weights": w, "diagnostics": diagnostics}
