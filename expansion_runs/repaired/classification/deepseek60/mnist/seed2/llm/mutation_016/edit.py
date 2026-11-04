def edit(w, f):
    import numpy as np

    # Extract full feedback pool
    X_img = f["x"].reshape(2000, 784).astype(np.float32)
    y = f["y"].astype(np.int64)
    W1 = w["fc1.weight"].astype(np.float32)
    b1 = w["fc1.bias"].astype(np.float32)

    # Pre-activations for all 2000 examples
    z = X_img @ W1.T + b1  # (2000, 128)

    # Closed-form linear discriminant
    def lda_fit(X, y, alpha):
        N, D = X.shape
        classes = np.unique(y)
        C = len(classes)
        mu = np.zeros((C, D), dtype=np.float64)
        for i, c in enumerate(classes):
            mu[i] = X[y == c].mean(axis=0)
        S = np.zeros((D, D), dtype=np.float64)
        for i, c in enumerate(classes):
            Xc = X[y == c]
            Xc_centered = Xc - mu[i]
            S += Xc_centered.T @ Xc_centered
        S /= (N - C)
        S_reg = S + alpha * np.trace(S) / D * np.eye(D)
        W = np.linalg.solve(S_reg, mu.T).T
        b = -0.5 * np.sum(mu * W, axis=1) + np.log(np.array([np.sum(y == c) for c in classes]) / N)
        return W, b

    # Closed-form ridge on one-hot targets
    def ridge_fit(X, y, alpha):
        N, D = X.shape
        classes = np.unique(y)
        C = len(classes)
        Y = np.zeros((N, C), dtype=np.float64)
        for i, c in enumerate(classes):
            Y[y == c, i] = 1.0
        X_aug = np.hstack([X, np.ones((N, 1))])
        A = X_aug.T @ X_aug + alpha * np.eye(D + 1)
        B = X_aug.T @ Y
        W_aug = np.linalg.solve(A, B)
        W = W_aug[:-1].T
        b = W_aug[-1]
        return W, b

    # 5-fold cross-validation on the feedback pool
    def cv_score(z, y, transform, method, alpha, n_folds=5):
        N = z.shape[0]
        indices = np.arange(N)
        rng = np.random.RandomState(42)
        rng.shuffle(indices)
        folds = np.array_split(indices, n_folds)
        accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            z_train = z[train_idx]
            z_val = z[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]

            if transform == 'none':
                X_train = np.maximum(z_train, 0)
                X_val = np.maximum(z_val, 0)
            elif transform == 'center_mean':
                mean_z = z_train.mean(axis=0)
                X_train = np.maximum(z_train - mean_z, 0)
                X_val = np.maximum(z_val - mean_z, 0)
            elif transform == 'center_scale':
                mean_z = z_train.mean(axis=0)
                std_z = z_train.std(axis=0) + 1e-8
                X_train = np.maximum((z_train - mean_z) / std_z, 0)
                X_val = np.maximum((z_val - mean_z) / std_z, 0)
            elif transform == 'center_median':
                median_z = np.median(z_train, axis=0)
                X_train = np.maximum(z_train - median_z, 0)
                X_val = np.maximum(z_val - median_z, 0)
            else:
                raise ValueError

            if method == 'lda':
                W, b = lda_fit(X_train, y_train, alpha)
            else:
                W, b = ridge_fit(X_train, y_train, alpha)

            logits = X_val @ W.T + b
            pred = np.argmax(logits, axis=1)
            accs.append(np.mean(pred == y_val))
        return np.mean(accs)

    transforms = ['none', 'center_mean', 'center_scale', 'center_median']
    methods = ['lda', 'ridge']
    alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

    best_score = -1.0
    best_params = None
    for tname in transforms:
        for method in methods:
            for alpha in alphas:
                score = cv_score(z, y, tname, method, alpha)
                if score > best_score:
                    best_score = score
                    best_params = (tname, method, alpha)

    tname, method, alpha = best_params

    # Compute global statistics for the chosen transform
    if tname == 'none':
        X_all = np.maximum(z, 0)
    elif tname == 'center_mean':
        mean_z = z.mean(axis=0)
        X_all = np.maximum(z - mean_z, 0)
    elif tname == 'center_scale':
        mean_z = z.mean(axis=0)
        std_z = z.std(axis=0) + 1e-8
        X_all = np.maximum((z - mean_z) / std_z, 0)
    elif tname == 'center_median':
        median_z = np.median(z, axis=0)
        X_all = np.maximum(z - median_z, 0)

    # Final fit on all 2000 examples
    if method == 'lda':
        W2_new, b2_new = lda_fit(X_all, y, alpha)
    else:
        W2_new, b2_new = ridge_fit(X_all, y, alpha)

    # Update fc1 weights/bias according to the chosen transform
    if tname == 'none':
        pass
    elif tname == 'center_mean':
        w["fc1.bias"] = (b1 - mean_z).astype(np.float32)
    elif tname == 'center_scale':
        w["fc1.weight"] = (W1 / std_z[:, None]).astype(np.float32)
        w["fc1.bias"] = ((b1 - mean_z) / std_z).astype(np.float32)
    elif tname == 'center_median':
        w["fc1.bias"] = (b1 - median_z).astype(np.float32)

    # Update fc2 readout
    w["fc2.weight"] = W2_new.astype(np.float32)
    w["fc2.bias"] = b2_new.astype(np.float32)

    # Clip to safe finite range
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)

    diagnostics = {
        "best_transform": tname,
        "best_method": method,
        "best_alpha": float(alpha),
        "best_cv_accuracy": float(best_score),
        "fc1_bias_mean": float(w["fc1.bias"].mean()),
        "fc1_bias_std": float(w["fc1.bias"].std()),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"]))
    }

    return {"weights": w, "diagnostics": diagnostics}
