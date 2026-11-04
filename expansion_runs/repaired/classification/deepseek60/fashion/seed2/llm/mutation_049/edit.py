import numpy as np

def edit(w, f):
    # Extract features, labels, and current logits
    X = f["features"].astype(np.float64)      # (2000, 1568)
    y = f["y"].astype(np.int64)               # (2000,)
    L = f["logits"].astype(np.float64)        # (2000, 10)
    n, d = X.shape
    n_classes = 10

    # Precompute full-data statistics for fast cross-validation
    XTX = X.T @ X                             # (d, d)
    XTL = X.T @ L                             # (d, 10)
    Y_onehot = np.zeros((n, n_classes))
    Y_onehot[np.arange(n), y] = 1.0
    XTY = X.T @ Y_onehot                      # (d, 10)

    # Helper: fit ridge regression to targets (L or Y) given training indices and lambda
    def fit_ridge(train_idx, lam, target):
        X_tr = X[train_idx]
        n_tr = len(train_idx)
        if target == 'L':
            T_tr = L[train_idx]
        else:
            T_tr = Y_onehot[train_idx]
        X_mean = X_tr.mean(axis=0)
        T_mean = T_tr.mean(axis=0)
        Xc = X_tr - X_mean
        Tc = T_tr - T_mean
        XTX_tr = Xc.T @ Xc
        XTT_tr = Xc.T @ Tc
        # Solve (XTX + lam I) W^T = XTT
        W = np.linalg.solve(XTX_tr + lam * np.eye(d), XTT_tr).T  # (10, d)
        b = T_mean - X_mean @ W.T
        return W, b

    # Helper: fit LDA with shrinkage given training indices and lambda
    def fit_lda(train_idx, lam):
        X_tr = X[train_idx]
        y_tr = y[train_idx]
        n_tr = len(train_idx)
        # Class means
        mu = np.zeros((n_classes, d))
        priors = np.zeros(n_classes)
        for c in range(n_classes):
            idx = np.where(y_tr == c)[0]
            n_c = len(idx)
            if n_c > 0:
                mu[c] = X_tr[idx].mean(axis=0)
                priors[c] = n_c / n_tr
            else:
                mu[c] = 0.0
                priors[c] = 0.0
        # Within-class scatter
        Sw = np.zeros((d, d))
        for c in range(n_classes):
            idx = np.where(y_tr == c)[0]
            if len(idx) > 0:
                Xc = X_tr[idx] - mu[c]
                Sw += Xc.T @ Xc
        # Shrinkage: Sigma = Sw + lam * trace(Sw)/d * I
        trace = np.trace(Sw)
        reg = lam * trace / d if trace > 0 else lam
        Sigma = Sw + reg * np.eye(d)
        # Solve Sigma * W^T = mu^T
        W = np.linalg.solve(Sigma, mu.T).T  # (10, d)
        b = -0.5 * np.sum(mu * W, axis=1) + np.log(priors + 1e-12)
        return W, b

    # Cross-validation setup
    n_folds = 5
    rng = np.random.RandomState(42)
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    # Lambda grids
    lambdas_ridge = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    lambdas_lda = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

    best_method = None
    best_lam = None
    best_cv_acc = -1.0
    cv_results = {}

    # Evaluate ridge to logits
    for lam in lambdas_ridge:
        accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            W, b = fit_ridge(train_idx, lam, 'L')
            logits_val = X[val_idx] @ W.T + b
            preds = np.argmax(logits_val, axis=1)
            acc = np.mean(preds == y[val_idx])
            accs.append(acc)
        mean_acc = np.mean(accs)
        cv_results[f'ridge_L_{lam}'] = mean_acc
        if mean_acc > best_cv_acc:
            best_cv_acc = mean_acc
            best_method = ('ridge', 'L')
            best_lam = lam

    # Evaluate ridge to one-hot
    for lam in lambdas_ridge:
        accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            W, b = fit_ridge(train_idx, lam, 'Y')
            logits_val = X[val_idx] @ W.T + b
            preds = np.argmax(logits_val, axis=1)
            acc = np.mean(preds == y[val_idx])
            accs.append(acc)
        mean_acc = np.mean(accs)
        cv_results[f'ridge_Y_{lam}'] = mean_acc
        if mean_acc > best_cv_acc:
            best_cv_acc = mean_acc
            best_method = ('ridge', 'Y')
            best_lam = lam

    # Evaluate LDA
    for lam in lambdas_lda:
        accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            W, b = fit_lda(train_idx, lam)
            logits_val = X[val_idx] @ W.T + b
            preds = np.argmax(logits_val, axis=1)
            acc = np.mean(preds == y[val_idx])
            accs.append(acc)
        mean_acc = np.mean(accs)
        cv_results[f'lda_{lam}'] = mean_acc
        if mean_acc > best_cv_acc:
            best_cv_acc = mean_acc
            best_method = ('lda', None)
            best_lam = lam

    # Fit on full data with best method and lambda
    if best_method[0] == 'ridge':
        W, b = fit_ridge(np.arange(n), best_lam, best_method[1])
    else:
        W, b = fit_lda(np.arange(n), best_lam)

    # Ensure weights are within [-1000, 1000]
    max_abs = max(np.max(np.abs(W)), np.max(np.abs(b)))
    if max_abs > 1000.0:
        scale = 1000.0 / max_abs
        W *= scale
        b *= scale
        max_abs = 1000.0

    # Update head weights and biases
    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

    # Compute training accuracy on full feedback pool
    logits_full = X @ W.T + b
    preds_full = np.argmax(logits_full, axis=1)
    train_acc = np.mean(preds_full == y)

    diagnostics = {
        "best_method": best_method[0] + ("_" + best_method[1] if best_method[1] else ""),
        "best_lambda": float(best_lam),
        "cv_accuracy": float(best_cv_acc),
        "train_accuracy": float(train_acc),
        "max_abs_head_weight": float(np.max(np.abs(W))),
        "max_abs_head_bias": float(np.max(np.abs(b))),
        "cv_results": {k: float(v) for k, v in cv_results.items()}
    }

    return {"weights": w, "diagnostics": diagnostics}
