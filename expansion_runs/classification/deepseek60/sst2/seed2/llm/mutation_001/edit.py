def edit(w, f):
    import numpy as np

    X = f["features"].astype(np.float64)
    y = f["y"].astype(np.int64)
    N, D = X.shape

    # Standardize features for stable closed-form fitting.
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-6] = 1.0
    Xs = (X - mu) / sigma

    Y = np.zeros((N, 2), dtype=np.float64)
    Y[np.arange(N), y] = 1.0
    X_aug = np.hstack([Xs, np.ones((N, 1), dtype=np.float64)])

    # 5-fold CV split.
    np.random.seed(42)
    indices = np.random.permutation(N)
    folds = np.array_split(indices, 5)

    # --- LDA CV ---
    lda_accs = []
    for i in range(5):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
        X_train = Xs[train_idx]
        y_train = y[train_idx]
        X_test = Xs[test_idx]
        y_test = y[test_idx]

        mu0 = X_train[y_train == 0].mean(axis=0)
        mu1 = X_train[y_train == 1].mean(axis=0)

        cov = np.zeros((D, D), dtype=np.float64)
        for c in [0, 1]:
            Xc = X_train[y_train == c]
            Xc_centered = Xc - Xc.mean(axis=0)
            cov += Xc_centered.T @ Xc_centered
        cov /= (len(train_idx) - 2)
        cov += np.eye(D) * 1e-6

        w_lda = np.linalg.solve(cov, mu1 - mu0)
        b_lda = -0.5 * (mu1 + mu0) @ w_lda
        logits = X_test @ w_lda + b_lda
        preds = (logits > 0).astype(np.int64)
        lda_accs.append(np.mean(preds == y_test))
    lda_mean_acc = float(np.mean(lda_accs))

    # --- Ridge CV ---
    lambdas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    best_ridge_acc = -1.0
    best_lambda = None
    for lam in lambdas:
        accs = []
        for i in range(5):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            X_train = X_aug[train_idx]
            Y_train = Y[train_idx]
            X_test = X_aug[test_idx]
            y_test = y[test_idx]

            reg = np.eye(D + 1) * lam
            reg[D, D] = 0.0
            A = X_train.T @ X_train + reg
            B = X_train.T @ Y_train
            W = np.linalg.solve(A, B)

            logits = X_test @ W
            preds = np.argmax(logits, axis=1)
            accs.append(np.mean(preds == y_test))
        mean_acc = float(np.mean(accs))
        if mean_acc > best_ridge_acc:
            best_ridge_acc = mean_acc
            best_lambda = lam

    # --- Choose best method and fit on full pool ---
    if lda_mean_acc >= best_ridge_acc:
        mu0 = Xs[y == 0].mean(axis=0)
        mu1 = Xs[y == 1].mean(axis=0)

        cov = np.zeros((D, D), dtype=np.float64)
        for c in [0, 1]:
            Xc = Xs[y == c]
            Xc_centered = Xc - Xc.mean(axis=0)
            cov += Xc_centered.T @ Xc_centered
        cov /= (N - 2)
        cov += np.eye(D) * 1e-6

        w_lda = np.linalg.solve(cov, mu1 - mu0)
        b_lda = -0.5 * (mu1 + mu0) @ w_lda

        w_orig = w_lda / sigma
        b_orig = b_lda - (mu / sigma) @ w_lda
        w_diff = w_orig
        b_diff = b_orig
        method = "LDA"
        cv_acc = lda_mean_acc
    else:
        lam = best_lambda
        reg = np.eye(D + 1) * lam
        reg[D, D] = 0.0
        A = X_aug.T @ X_aug + reg
        B = X_aug.T @ Y
        W = np.linalg.solve(A, B)

        W_orig = W[:D] / sigma[:, None]
        b_orig = W[D] - (mu / sigma) @ W[:D]
        w_diff = W_orig[:, 1] - W_orig[:, 0]
        b_diff = b_orig[1] - b_orig[0]
        method = f"Ridge lambda={best_lambda}"
        cv_acc = best_ridge_acc

    # Symmetric binary head: class 1 uses +w_diff, class 0 uses -w_diff.
    w["head.weight"][1] = w_diff.astype(np.float32)
    w["head.weight"][0] = (-w_diff).astype(np.float32)
    w["head.bias"][1] = np.float32(b_diff)
    w["head.bias"][0] = np.float32(-b_diff)

    # Safety clip.
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)

    # Diagnostics.
    logits_new = X @ w["head.weight"].T.astype(np.float64) + w["head.bias"].astype(np.float64)
    preds_new = np.argmax(logits_new, axis=1)
    train_acc = float(np.mean(preds_new == y))

    diagnostics = {
        "method": method,
        "cv_accuracy": cv_acc,
        "train_accuracy": train_acc,
        "head_weight_norm": float(np.linalg.norm(w["head.weight"])),
        "head_bias": w["head.bias"].tolist(),
    }
    return {"weights": w, "diagnostics": diagnostics}
