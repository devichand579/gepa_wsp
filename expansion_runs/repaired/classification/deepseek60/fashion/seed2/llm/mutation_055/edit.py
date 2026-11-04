import numpy as np

def edit(w, f):
    # Extract features and labels from the full feedback pool
    X = f["features"].astype(np.float64)          # (2000, 1568)
    y = f["y"].astype(np.int64)                   # (2000,)
    n_samples, n_features = X.shape
    n_classes = 10

    # One-hot encode labels
    Y = np.zeros((n_samples, n_classes), dtype=np.float64)
    Y[np.arange(n_samples), y] = 1.0

    # Add bias column
    X_aug = np.hstack([X, np.ones((n_samples, 1), dtype=np.float64)])  # (2000, 1569)

    # Candidate regularization strengths
    lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]

    # 5-fold cross-validation to select lambda
    n_folds = 5
    fold_size = n_samples // n_folds
    indices = np.arange(n_samples)
    np.random.seed(42)  # for reproducibility
    np.random.shuffle(indices)

    best_lambda = None
    best_cv_acc = -1.0

    for lam in lambdas:
        cv_acc = 0.0
        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n_samples
            val_idx = indices[val_start:val_end]
            train_idx = np.setdiff1d(indices, val_idx)

            X_train = X_aug[train_idx]
            Y_train = Y[train_idx]
            X_val = X_aug[val_idx]
            y_val = y[val_idx]

            # Ridge solution: W = (X^T X + lambda I)^-1 X^T Y
            # Use solve for numerical stability
            A = X_train.T @ X_train + lam * np.eye(n_features + 1)
            B = X_train.T @ Y_train
            W = np.linalg.solve(A, B)  # (1569, 10)

            # Predict on validation
            logits_val = X_val @ W
            preds = np.argmax(logits_val, axis=1)
            cv_acc += np.mean(preds == y_val)

        cv_acc /= n_folds
        if cv_acc > best_cv_acc:
            best_cv_acc = cv_acc
            best_lambda = lam

    # If no lambda improved over default, fall back to a small lambda
    if best_lambda is None:
        best_lambda = 1e-4

    # Fit final head on the entire feedback pool using the best lambda
    A = X_aug.T @ X_aug + best_lambda * np.eye(n_features + 1)
    B = X_aug.T @ Y
    W_final = np.linalg.solve(A, B)  # (1569, 10)

    # Extract weights and bias
    head_weight = W_final[:n_features, :].T.astype(np.float32)  # (10, 1568)
    head_bias = W_final[n_features, :].astype(np.float32)       # (10,)

    # Clip to ensure finite and within bounds
    head_weight = np.clip(head_weight, -1000.0, 1000.0)
    head_bias = np.clip(head_bias, -1000.0, 1000.0)

    # Update weights
    w["head.weight"] = head_weight
    w["head.bias"] = head_bias

    # Diagnostics
    train_logits = X @ head_weight.T + head_bias
    train_preds = np.argmax(train_logits, axis=1)
    train_acc = np.mean(train_preds == y)
    max_abs_weight = np.max(np.abs(head_weight))
    max_abs_bias = np.max(np.abs(head_bias))

    diagnostics = {
        "best_lambda": best_lambda,
        "cv_accuracy": float(best_cv_acc),
        "train_accuracy": float(train_acc),
        "max_abs_head_weight": float(max_abs_weight),
        "max_abs_head_bias": float(max_abs_bias),
    }

    return {"weights": w, "diagnostics": diagnostics}
