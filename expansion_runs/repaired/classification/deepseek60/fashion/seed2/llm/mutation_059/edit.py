import numpy as np

def edit(w, f):
    # Extract features and labels from the full feedback pool
    X = f["features"].astype(np.float64)          # [2000, 1568]
    y = f["y"].astype(np.int64)                   # [2000]
    n_samples, n_features = X.shape
    n_classes = 10

    # One-hot targets
    Y = np.zeros((n_samples, n_classes), dtype=np.float64)
    Y[np.arange(n_samples), y] = 1.0

    # Center features to improve conditioning (optional but helpful)
    X_mean = X.mean(axis=0, keepdims=True)
    X_centered = X - X_mean

    # Candidate ridge lambdas (log-spaced)
    lambdas = np.logspace(-6, 2, 25)

    # 5-fold cross-validation to select lambda
    n_folds = 5
    fold_size = n_samples // n_folds
    # Shuffle indices deterministically
    rng = np.random.RandomState(42)
    indices = rng.permutation(n_samples)

    best_lambda = None
    best_cv_acc = -1.0

    for lam in lambdas:
        cv_acc = 0.0
        for fold in range(n_folds):
            val_idx = indices[fold * fold_size:(fold + 1) * fold_size]
            train_idx = np.setdiff1d(indices, val_idx, assume_unique=False)
            X_train = X_centered[train_idx]
            Y_train = Y[train_idx]
            X_val = X_centered[val_idx]
            y_val = y[val_idx]

            # Ridge solution: W = (X^T X + lam I)^-1 X^T Y
            # Use the dual form for efficiency when n_features > n_samples
            # W = X^T (X X^T + lam I)^-1 Y
            XXT = X_train @ X_train.T
            XXT[np.diag_indices_from(XXT)] += lam
            alpha = np.linalg.solve(XXT, Y_train)   # [n_train, 10]
            W = X_train.T @ alpha                   # [n_features, 10]
            # Bias: we can compute bias as the mean of (Y - X W) or simply use class priors.
            # For simplicity, set bias to zero and rely on centered features.
            logits_val = X_val @ W
            preds = np.argmax(logits_val, axis=1)
            cv_acc += np.mean(preds == y_val)
        cv_acc /= n_folds
        if cv_acc > best_cv_acc:
            best_cv_acc = cv_acc
            best_lambda = lam

    # Train final model on all data with best lambda
    lam = best_lambda
    XXT = X_centered @ X_centered.T
    XXT[np.diag_indices_from(XXT)] += lam
    alpha = np.linalg.solve(XXT, Y)
    W = X_centered.T @ alpha                      # [1568, 10]
    # Bias: compute optimal bias for the least-squares solution
    # We can solve for bias by including a column of ones, but simpler: set bias to
    # the mean of Y - X_centered @ W over training samples.
    residual = Y - X_centered @ W
    bias = residual.mean(axis=0)                  # [10]

    # Transpose to match head.weight shape [10, 1568]
    head_weight = W.T.astype(np.float32)
    head_bias = bias.astype(np.float32)

    # Update the weights dictionary
    w["head.weight"] = head_weight
    w["head.bias"] = head_bias

    # Diagnostics
    # Compute training accuracy and cross-entropy with the new head
    logits_train = X_centered @ W + bias
    # Softmax for cross-entropy
    logits_train_stable = logits_train - logits_train.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits_train_stable)
    probs_train = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    train_acc = np.mean(np.argmax(logits_train, axis=1) == y)
    eps = 1e-12
    ce = -np.mean(np.log(probs_train[np.arange(n_samples), y] + eps))

    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_cv_acc),
        "train_accuracy": float(train_acc),
        "cross_entropy": float(ce),
        "max_abs_head_weight": float(np.max(np.abs(head_weight))),
        "max_abs_head_bias": float(np.max(np.abs(head_bias))),
    }

    return {"weights": w, "diagnostics": diagnostics}
