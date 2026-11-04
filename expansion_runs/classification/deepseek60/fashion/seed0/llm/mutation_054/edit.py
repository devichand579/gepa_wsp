import numpy as np

def edit(w, f):
    # Extract features and labels
    X = f["features"]          # (2000, 1568)
    y = f["y"]                 # (2000,)
    N, D = X.shape
    C = 10

    # One-hot encode labels
    Y = np.zeros((N, C), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    # Center features and labels
    X_mean = X.mean(axis=0, keepdims=True)
    Y_mean = Y.mean(axis=0, keepdims=True)
    Xc = X - X_mean
    Yc = Y - Y_mean

    def ridge_fit(Xc, Yc, lam):
        n = Xc.shape[0]
        # Dual form: W = Xc^T (Xc Xc^T + lam I)^-1 Yc
        K = Xc @ Xc.T
        K_reg = K + lam * np.eye(n, dtype=np.float32)
        alpha = np.linalg.solve(K_reg, Yc)
        W = Xc.T @ alpha
        W = W.T
        b = Y_mean.flatten() - W @ X_mean.flatten()
        return W, b

    # 5-fold cross-validation to select lambda
    rng = np.random.RandomState(42)
    indices = rng.permutation(N)
    n_folds = 5
    fold_size = N // n_folds
    lambdas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    best_lam = 1000.0
    best_acc = -1.0

    for lam in lambdas:
        accs = []
        for fold in range(n_folds):
            val_idx = indices[fold*fold_size:(fold+1)*fold_size]
            train_idx = np.setdiff1d(indices, val_idx)
            X_train = X[train_idx]
            Y_train = Y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]

            X_mean_tr = X_train.mean(axis=0, keepdims=True)
            Y_mean_tr = Y_train.mean(axis=0, keepdims=True)
            Xc_tr = X_train - X_mean_tr
            Yc_tr = Y_train - Y_mean_tr

            W, b = ridge_fit(Xc_tr, Yc_tr, lam)
            logits = X_val @ W.T + b
            preds = np.argmax(logits, axis=1)
            acc = np.mean(preds == y_val)
            accs.append(acc)
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lam = lam

    # Fit on full feedback pool with best lambda
    W, b = ridge_fit(Xc, Yc, best_lam)

    # Update head weights
    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lam),
        "cv_accuracy": float(best_acc),
        "head_weight_norm": float(np.linalg.norm(w["head.weight"])),
        "head_bias_norm": float(np.linalg.norm(w["head.bias"])),
        "head_weight_max_abs": float(np.max(np.abs(w["head.weight"]))),
        "head_bias_max_abs": float(np.max(np.abs(w["head.bias"]))),
    }
    return {"weights": w, "diagnostics": diagnostics}
