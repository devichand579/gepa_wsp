import numpy as np

def edit(w, f):
    # Extract feedback data
    X = f["features"].astype(np.float64)  # (2000, 128) current hidden features
    y = f["y"].astype(np.int64)
    N = X.shape[0]
    C = 10

    # One-hot targets
    Y = np.zeros((N, C))
    Y[np.arange(N), y] = 1.0

    # Current fc2 parameters
    W_cur = w["fc2.weight"].astype(np.float64)  # (10, 128)
    b_cur = w["fc2.bias"].astype(np.float64)    # (10,)

    # Helper: evaluate accuracy of a linear classifier on features X
    def accuracy(W, b):
        logits = X @ W.T + b
        pred = np.argmax(logits, axis=1)
        return np.mean(pred == y)

    # Current accuracy on full pool
    acc_cur = accuracy(W_cur, b_cur)

    # 5-fold cross-validation indices
    rng = np.random.RandomState(42)
    idx = rng.permutation(N)
    folds = np.array_split(idx, 5)

    # Try several ridge lambdas
    best_lam = None
    best_cv_acc = -1.0
    best_W = None
    best_b = None

    for lam in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        cv_accs = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            X_train, Y_train = X[train_idx], Y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Add bias column
            X_train_aug = np.hstack([X_train, np.ones((len(X_train), 1))])
            X_val_aug = np.hstack([X_val, np.ones((len(X_val), 1))])

            # Ridge normal equations: (X^T X + lam I) W = X^T Y
            A = X_train_aug.T @ X_train_aug + lam * np.eye(X_train_aug.shape[1])
            B = X_train_aug.T @ Y_train
            W_fc2 = np.linalg.solve(A, B)  # (129, 10)

            # Evaluate
            logits = X_val_aug @ W_fc2
            pred = np.argmax(logits, axis=1)
            cv_accs.append(np.mean(pred == y_val))

        mean_cv = np.mean(cv_accs)
        if mean_cv > best_cv_acc:
            best_cv_acc = mean_cv
            best_lam = lam

    # Fit best ridge on full pool
    X_aug = np.hstack([X, np.ones((N, 1))])
    A = X_aug.T @ X_aug + best_lam * np.eye(X_aug.shape[1])
    B = X_aug.T @ Y
    W_new = np.linalg.solve(A, B)  # (129, 10)
    W_new_fc2 = W_new[:128, :].T  # (10, 128)
    b_new_fc2 = W_new[128, :]     # (10,)

    # Evaluate new classifier on full pool
    acc_new = accuracy(W_new_fc2, b_new_fc2)

    # Only replace if new is at least as accurate as current on full pool
    if acc_new >= acc_cur:
        w["fc2.weight"] = W_new_fc2.astype(np.float32)
        w["fc2.bias"] = b_new_fc2.astype(np.float32)
        used_new = True
    else:
        used_new = False

    # Clip to safe range
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)

    diagnostics = {
        "acc_cur_full": float(acc_cur),
        "acc_new_full": float(acc_new),
        "best_lambda": float(best_lam),
        "best_cv_acc": float(best_cv_acc),
        "used_new_fc2": bool(used_new),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"]))
    }
    return {"weights": w, "diagnostics": diagnostics}
