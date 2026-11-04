import numpy as np

def edit(w, f):
    # Extract full feedback features and labels
    X = f["features"].astype(np.float64)          # (2000, 1568)
    y = f["y"].astype(np.int64)                   # (2000,)
    N, D = X.shape
    C = 10

    # Current head parameters
    W_old = w["head.weight"].astype(np.float64)   # (10, 1568)
    b_old = w["head.bias"].astype(np.float64)     # (10,)

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Augment features with bias column
    X_aug = np.concatenate([X, np.ones((N, 1), dtype=np.float64)], axis=1)  # (N, D+1)
    W0_aug = np.concatenate([W_old, b_old[:, None]], axis=1)                # (C, D+1)

    # Precompute for efficiency
    XtX = X_aug.T @ X_aug                     # (D+1, D+1)
    XtY = X_aug.T @ Y                         # (D+1, C)

    # Candidate regularization strengths (lambda)
    lambdas = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

    # 5-fold cross-validation to select lambda
    n_folds = 5
    fold_size = N // n_folds
    indices = np.arange(N)
    rng = np.random.RandomState(42)
    rng.shuffle(indices)

    best_lambda = None
    best_cv_acc = -1.0
    best_cv_ce = np.inf

    for lam in lambdas:
        cv_acc = 0.0
        cv_ce = 0.0
        for k in range(n_folds):
            val_idx = indices[k*fold_size:(k+1)*fold_size]
            train_idx = np.setdiff1d(indices, val_idx)

            X_tr = X_aug[train_idx]
            Y_tr = Y[train_idx]
            X_val = X_aug[val_idx]
            Y_val = Y[val_idx]

            # Ridge solution with prior W0_aug
            # (X^T X + lam I) W^T = X^T Y + lam W0^T
            A = X_tr.T @ X_tr + lam * np.eye(D+1)
            B = X_tr.T @ Y_tr + lam * W0_aug.T
            W_aug = np.linalg.solve(A, B).T   # (C, D+1)

            # Predict on validation
            logits_val = X_val @ W_aug.T      # (n_val, C)
            preds = np.argmax(logits_val, axis=1)
            acc = np.mean(preds == y[val_idx])
            cv_acc += acc

            # Cross-entropy
            logits_val -= np.max(logits_val, axis=1, keepdims=True)
            probs = np.exp(logits_val)
            probs /= np.sum(probs, axis=1, keepdims=True)
            ce = -np.mean(np.log(probs[np.arange(len(val_idx)), y[val_idx]] + 1e-12))
            cv_ce += ce

        cv_acc /= n_folds
        cv_ce /= n_folds

        if cv_acc > best_cv_acc or (cv_acc == best_cv_acc and cv_ce < best_cv_ce):
            best_cv_acc = cv_acc
            best_cv_ce = cv_ce
            best_lambda = lam

    # Final fit on all data using best lambda
    lam = best_lambda
    A = XtX + lam * np.eye(D+1)
    B = XtY + lam * W0_aug.T
    W_aug = np.linalg.solve(A, B).T   # (C, D+1)

    W_new = W_aug[:, :D].astype(np.float32)
    b_new = W_aug[:, D].astype(np.float32)

    # Write back
    w["head.weight"] = W_new
    w["head.bias"] = b_new

    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_cv_acc),
        "cv_cross_entropy": float(best_cv_ce),
        "delta_W_norm": float(np.linalg.norm(W_new - W_old)),
        "delta_b_norm": float(np.linalg.norm(b_new - b_old)),
        "head_weight_norm": float(np.linalg.norm(W_new)),
        "head_bias_norm": float(np.linalg.norm(b_new)),
    }
    return {"weights": w, "diagnostics": diagnostics}
