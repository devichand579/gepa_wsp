import numpy as np

def edit(w, f):
    # Extract training features and labels
    X = f["features"].astype(np.float64)   # (2000, 1568)
    y = f["y"].astype(np.int64)            # (2000,)
    N, D = X.shape
    C = 10

    # One-hot encode labels
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Augment features with a bias column
    Xa = np.concatenate([X, np.ones((N, 1))], axis=1)  # (2000, 1569)

    # Ridge regression with cross-validation to choose lambda
    lambdas = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]
    best_lambda = None
    best_cv_acc = -1.0

    n_folds = 5
    indices = np.arange(N)
    rng = np.random.RandomState(0)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    for lam in lambdas:
        accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])

            X_train = Xa[train_idx]
            Y_train = Y[train_idx]
            X_val = Xa[val_idx]
            Y_val = Y[val_idx]

            # Normal equations: (X^T X + lam I) W = X^T Y
            A = X_train.T @ X_train + lam * np.eye(D + 1)
            B = X_train.T @ Y_train
            W = np.linalg.solve(A, B)

            logits = X_val @ W
            pred = np.argmax(logits, axis=1)
            true = np.argmax(Y_val, axis=1)
            accs.append(np.mean(pred == true))

        mean_acc = np.mean(accs)
        if mean_acc > best_cv_acc:
            best_cv_acc = mean_acc
            best_lambda = lam

    # Refit on full training data with the best lambda
    A = Xa.T @ Xa + best_lambda * np.eye(D + 1)
    B = Xa.T @ Y
    W_full = np.linalg.solve(A, B)  # (1569, 10)

    # Extract head weights and bias
    head_weight = W_full[:D, :].T   # (10, 1568)
    head_bias = W_full[D, :]        # (10,)

    # Update the weights dictionary
    w["head.weight"] = head_weight.astype(np.float32)
    w["head.bias"] = head_bias.astype(np.float32)

    # Diagnostics
    logits_full = Xa @ W_full
    pred_full = np.argmax(logits_full, axis=1)
    train_acc = np.mean(pred_full == y)

    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_cv_acc),
        "train_accuracy": float(train_acc),
        "head_weight_norm": float(np.linalg.norm(head_weight)),
        "head_bias_norm": float(np.linalg.norm(head_bias)),
    }

    return {"weights": w, "diagnostics": diagnostics}
