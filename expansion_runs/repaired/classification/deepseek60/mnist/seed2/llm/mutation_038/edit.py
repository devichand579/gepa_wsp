import numpy as np

def edit(w, f):
    # Extract features and labels from the full feedback pool
    X = f["features"].astype(np.float64)  # (2000, 128)
    y = f["y"].astype(np.int64)           # (2000,)
    n, d = X.shape
    num_classes = 10

    # Standardize features (zero mean, unit variance)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    Xs = (X - mu) / sigma

    # One-hot targets
    Y = np.zeros((n, num_classes), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    # Augment with bias column
    X_aug = np.hstack([Xs, np.ones((n, 1))])

    # Ridge regression with cross-validation to choose lambda
    # Use a small grid of lambdas
    lambdas = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
    best_lambda = 1.0
    best_cv_acc = -1.0

    # 5-fold cross-validation
    n_folds = 5
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    for lam in lambdas:
        cv_acc = 0.0
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            X_train = X_aug[train_idx]
            Y_train = Y[train_idx]
            X_val = X_aug[val_idx]
            y_val = y[val_idx]

            # Solve ridge: (X^T X + lam * I) W = X^T Y
            # Use normal equations
            A = X_train.T @ X_train + lam * np.eye(d + 1)
            B = X_train.T @ Y_train
            W = np.linalg.solve(A, B)

            # Predict on validation
            logits_val = X_val @ W
            preds = np.argmax(logits_val, axis=1)
            cv_acc += np.mean(preds == y_val)
        cv_acc /= n_folds
        if cv_acc > best_cv_acc:
            best_cv_acc = cv_acc
            best_lambda = lam

    # Refit on full data with best lambda
    A = X_aug.T @ X_aug + best_lambda * np.eye(d + 1)
    B = X_aug.T @ Y
    W_full = np.linalg.solve(A, B)

    # Extract weights and bias in standardized space
    W_std = W_full[:d, :].T  # (10, 128)
    b_std = W_full[d, :]     # (10,)

    # Convert back to original feature space:
    # logits = (Xs) @ W_std.T + b_std
    #        = ((X - mu)/sigma) @ W_std.T + b_std
    #        = X @ (W_std / sigma).T + (b_std - (mu/sigma) @ W_std.T)
    W_orig = W_std / sigma[None, :]  # (10, 128)
    b_orig = b_std - (mu / sigma) @ W_std.T  # (10,)

    # Rescale logits to match current logit standard deviation
    # Current logits: f["logits"] (2000, 10)
    current_logit_std = np.std(f["logits"])
    new_logits = X @ W_orig.T + b_orig
    new_logit_std = np.std(new_logits)
    if new_logit_std > 1e-8:
        scale = current_logit_std / new_logit_std
    else:
        scale = 1.0

    W_orig *= scale
    b_orig *= scale

    # Update weights in w
    w["fc2.weight"] = W_orig.astype(np.float32)
    w["fc2.bias"] = b_orig.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_cv_acc),
        "current_logit_std": float(current_logit_std),
        "new_logit_std_before_scale": float(new_logit_std),
        "scale": float(scale),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
        "fc2_weight_min": float(np.min(w["fc2.weight"])),
        "fc2_weight_max": float(np.max(w["fc2.weight"])),
        "fc2_bias_min": float(np.min(w["fc2.bias"])),
        "fc2_bias_max": float(np.max(w["fc2.bias"])),
    }

    return {"weights": w, "diagnostics": diagnostics}
