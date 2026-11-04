import numpy as np

def edit(w, f):
    X = f["features"].astype(np.float64)
    y = f["y"]
    n, d = X.shape

    # One-hot targets
    Y = np.zeros((n, 10), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    # 5-fold CV to select ridge lambda
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    n_folds = 5
    fold_sizes = [n // n_folds] * n_folds
    for i in range(n % n_folds):
        fold_sizes[i] += 1

    lambdas = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    best_lambda = 0.1
    best_acc = -1.0

    for lam in lambdas:
        accs = []
        start = 0
        for fs in fold_sizes:
            val_idx = indices[start:start+fs]
            train_idx = np.setdiff1d(indices, val_idx, assume_unique=False)
            X_train = X[train_idx]
            Y_train = Y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]

            X_mean = X_train.mean(axis=0)
            Y_mean = Y_train.mean(axis=0)
            X_c = X_train - X_mean
            Y_c = Y_train - Y_mean

            A = X_c.T @ X_c + lam * np.eye(d)
            B = X_c.T @ Y_c
            W_T = np.linalg.solve(A, B)
            W = W_T.T
            b = Y_mean - W @ X_mean

            logits = X_val @ W.T + b
            pred = np.argmax(logits, axis=1)
            acc = np.mean(pred == y_val)
            accs.append(acc)
            start += fs

        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lambda = lam

    # Refit on full data with best lambda
    X_mean = X.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    X_c = X - X_mean
    Y_c = Y - Y_mean

    A = X_c.T @ X_c + best_lambda * np.eye(d)
    B = X_c.T @ Y_c
    W_T = np.linalg.solve(A, B)
    W = W_T.T
    b = Y_mean - W @ X_mean

    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

    # Diagnostics
    logits = X @ W.T + b
    pred = np.argmax(logits, axis=1)
    train_acc = float(np.mean(pred == y))

    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_acc": float(best_acc),
        "train_acc": train_acc,
        "head_weight_std": float(np.std(W)),
        "head_bias_mean": float(np.mean(b)),
    }

    return {"weights": w, "diagnostics": diagnostics}
