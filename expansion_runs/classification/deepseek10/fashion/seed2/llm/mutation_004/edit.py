import numpy as np

def edit(w, f):
    # Extract current head parameters
    W_old = w["head.weight"].astype(np.float64)  # [10, 1568]
    b_old = w["head.bias"].astype(np.float64)    # [10]

    # Get full feedback pool
    X = f["features"].astype(np.float64)         # [2000, 1568]
    y = f["y"].astype(np.int64)                  # [2000]
    N = X.shape[0]
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Current logits (should match f["logits"] but we recompute for consistency)
    L = X @ W_old.T + b_old                      # [2000, 10]

    # We will fit an affine transform: L_new = L @ A.T + c
    # Augment L with a column of ones for the bias c
    L_aug = np.hstack([L, np.ones((N, 1))])      # [2000, 11]

    # 5-fold cross-validation to select ridge lambda
    n_folds = 5
    fold_size = N // n_folds
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)

    lambdas = [0.0, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0]
    best_lambda = 0.0
    best_cv_acc = -1.0
    best_cv_ce = np.inf

    # Precompute L_aug^T L_aug and L_aug^T Y for the full set
    LtL_full = L_aug.T @ L_aug
    LtY_full = L_aug.T @ Y

    for lam in lambdas:
        cv_acc = 0.0
        cv_ce = 0.0
        for fold in range(n_folds):
            val_idx = indices[fold * fold_size:(fold + 1) * fold_size]
            train_idx = np.setdiff1d(indices, val_idx)

            # Compute train matrices by subtracting validation contributions
            L_val = L_aug[val_idx]
            Y_val = Y[val_idx]
            LtL_train = LtL_full - L_val.T @ L_val
            LtY_train = LtY_full - L_val.T @ Y_val

            # Ridge solution: A_aug = (LtL + lam * I)^-1 LtY
            # Note: we do not penalize the bias term (last column) by setting its diagonal to 0
            reg = lam * np.eye(L_aug.shape[1])
            reg[-1, -1] = 0.0
            A_aug = np.linalg.solve(LtL_train + reg, LtY_train)  # [11, 10]

            # Predict on validation fold
            L_val_pred = L_val @ A_aug  # [val_size, 10]
            # Accuracy
            preds = np.argmax(L_val_pred, axis=1)
            cv_acc += np.mean(preds == y[val_idx])
            # Cross-entropy
            # Softmax
            logits_max = np.max(L_val_pred, axis=1, keepdims=True)
            exp_logits = np.exp(L_val_pred - logits_max)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            # Clip to avoid log(0)
            probs = np.clip(probs, 1e-15, 1.0)
            ce = -np.mean(np.log(probs[np.arange(len(val_idx)), y[val_idx]]))
            cv_ce += ce

        cv_acc /= n_folds
        cv_ce /= n_folds

        if cv_acc > best_cv_acc or (cv_acc == best_cv_acc and cv_ce < best_cv_ce):
            best_cv_acc = cv_acc
            best_cv_ce = cv_ce
            best_lambda = lam

    # Fit final affine transform on the full pool using the best lambda
    reg = best_lambda * np.eye(L_aug.shape[1])
    reg[-1, -1] = 0.0
    A_aug = np.linalg.solve(LtL_full + reg, LtY_full)  # [11, 10]
    A = A_aug[:10, :].T   # [10, 10] because L_new = L @ A.T + c
    c = A_aug[10, :]      # [10]

    # New head weights: W_new = A @ W_old, b_new = A @ b_old + c
    W_new = A @ W_old
    b_new = A @ b_old + c

    # Update weights in place (preserve dtype float32)
    w["head.weight"] = W_new.astype(np.float32)
    w["head.bias"] = b_new.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_cv_acc),
        "cv_cross_entropy": float(best_cv_ce),
        "A_norm": float(np.linalg.norm(A)),
        "b_new_norm": float(np.linalg.norm(b_new)),
        "W_new_norm": float(np.linalg.norm(W_new)),
        "delta_W_norm": float(np.linalg.norm(W_new - W_old)),
        "delta_b_norm": float(np.linalg.norm(b_new - b_old)),
    }

    return {"weights": w, "diagnostics": diagnostics}
