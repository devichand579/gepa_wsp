import numpy as np

def edit(w, f):
    X = f["features"].astype(np.float64)          # (2000, 128)
    y = f["y"].astype(np.int64)                   # (2000,)
    N, D = X.shape
    C = 10

    W0 = w["fc2.weight"].astype(np.float64)       # (10, 128)
    b0 = w["fc2.bias"].astype(np.float64)         # (10,)

    # Standardize features
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    Xs = (X - mean) / std

    # Current fc2 in standardized space
    W0s = W0 * std
    b0s = b0 + W0 @ mean

    # One-hot targets
    Y = np.zeros((N, C))
    Y[np.arange(N), y] = 1.0

    # Augment with intercept
    Xa = np.hstack([Xs, np.ones((N, 1))])         # (N, D+1)
    W0_aug = np.hstack([W0s, b0s.reshape(-1, 1)]) # (10, D+1)

    # 5-fold CV to select ridge lambda
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, 5)

    def cv_ridge(lam):
        accs = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            Xa_train = Xa[train_idx]
            Y_train = Y[train_idx]
            Xa_val = Xa[val_idx]
            y_val = y[val_idx]

            A = Xa_train.T @ Xa_train + lam * np.eye(D + 1)
            B = Xa_train.T @ Y_train + lam * W0_aug.T
            W_aug = np.linalg.solve(A, B).T
            logits = Xa_val @ W_aug.T
            preds = np.argmax(logits, axis=1)
            accs.append(np.mean(preds == y_val))
        return np.mean(accs)

    lambdas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    best_lam = None
    best_cv = -1.0
    for lam in lambdas:
        cv = cv_ridge(lam)
        if cv > best_cv:
            best_cv = cv
            best_lam = lam

    # Refit on full pool with best lambda
    lam = best_lam
    A = Xa.T @ Xa + lam * np.eye(D + 1)
    B = Xa.T @ Y + lam * W0_aug.T
    W_aug = np.linalg.solve(A, B).T               # (10, D+1)

    Ws = W_aug[:, :D]
    bs = W_aug[:, D]

    # Convert back to original feature space
    W_new = Ws / std
    b_new = bs - W_new @ mean

    # Compare full-pool accuracy
    logits_current = X @ W0.T + b0
    acc_current = np.mean(np.argmax(logits_current, axis=1) == y)

    logits_new = X @ W_new.T + b_new
    acc_new = np.mean(np.argmax(logits_new, axis=1) == y)

    if acc_new > acc_current:
        w["fc2.weight"] = W_new.astype(np.float32)
        w["fc2.bias"] = b_new.astype(np.float32)
        method = "ridge_prior"
    else:
        method = "keep_current"

    diag = {
        "method": method,
        "lambda": float(lam),
        "cv_accuracy": float(best_cv),
        "acc_current": float(acc_current),
        "acc_new": float(acc_new),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
    }
    return {"weights": w, "diagnostics": diag}
