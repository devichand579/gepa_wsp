import numpy as np

def edit(w, f):
    # Extract frozen features and labels from the full feedback pool
    X = f["features"].astype(np.float64)          # (2000, 1568)
    y = f["y"].astype(np.int64)                   # (2000,)
    N, D = X.shape
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Augment features with a bias column
    X_aug = np.hstack([X, np.ones((N, 1), dtype=np.float64)])
    D_aug = D + 1

    # Candidate ridge regularization strengths
    lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    best_lambda = None
    best_cv_acc = -1.0
    cv_results = {}

    # 5-fold cross-validation
    rng = np.random.default_rng(42)
    indices = np.arange(N)
    rng.shuffle(indices)
    folds = np.array_split(indices, 5)

    for lam in lambdas:
        accs = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])

            X_train = X_aug[train_idx]
            Y_train = Y[train_idx]
            X_val = X_aug[val_idx]
            Y_val = Y[val_idx]

            # Closed-form ridge solution
            A = X_train.T @ X_train + lam * np.eye(D_aug)
            B = X_train.T @ Y_train
            W = np.linalg.solve(A, B).T          # (C, D_aug)

            # Enforce the absolute value constraint
            W = np.clip(W, -1000.0, 1000.0)

            # Validation accuracy
            logits = X_val @ W.T
            pred = np.argmax(logits, axis=1)
            true = np.argmax(Y_val, axis=1)
            accs.append(np.mean(pred == true))

        mean_acc = float(np.mean(accs))
        cv_results[lam] = mean_acc
        if mean_acc > best_cv_acc:
            best_cv_acc = mean_acc
            best_lambda = lam

    # Train on the full feedback pool using the best lambda
    lam = best_lambda
    A = X_aug.T @ X_aug + lam * np.eye(D_aug)
    B = X_aug.T @ Y
    W_full = np.linalg.solve(A, B).T             # (10, 1569)
    W_full = np.clip(W_full, -1000.0, 1000.0)

    # Update only the head
    w["head.weight"] = W_full[:, :D].astype(np.float32)
    w["head.bias"]   = W_full[:, D].astype(np.float32)

    # Diagnostics
    logits_full = X_aug @ W_full.T
    pred_full = np.argmax(logits_full, axis=1)
    train_acc = float(np.mean(pred_full == y))
    max_abs_w = float(np.max(np.abs(w["head.weight"])))
    max_abs_b = float(np.max(np.abs(w["head.bias"])))

    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_accuracy": best_cv_acc,
        "train_accuracy": train_acc,
        "max_abs_head_weight": max_abs_w,
        "max_abs_head_bias": max_abs_b,
        "cv_results": {str(k): v for k, v in cv_results.items()},
    }

    return {"weights": w, "diagnostics": diagnostics}
