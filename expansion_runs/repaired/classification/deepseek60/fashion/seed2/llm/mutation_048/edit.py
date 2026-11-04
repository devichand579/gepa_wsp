import numpy as np

def edit(w, f):
    # Extract features and labels
    X = f["features"].astype(np.float64)
    y = f["y"].astype(np.int64)
    n, d = X.shape
    n_classes = 10

    # One-hot targets
    Y = np.zeros((n, n_classes))
    Y[np.arange(n), y] = 1.0

    # 5-fold CV setup
    n_folds = 5
    seed = 42
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    results = []

    # --- Ridge regression CV ---
    alphas = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]
    for alpha in alphas:
        accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            X_train = X[train_idx]
            Y_train = Y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]

            X_mean = X_train.mean(axis=0)
            Y_mean = Y_train.mean(axis=0)
            Xc = X_train - X_mean
            Yc = Y_train - Y_mean

            # SVD of centered training features
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            UT_Yc = U.T @ Yc
            D = S / (S**2 + alpha)
            W = Vt.T @ (D[:, None] * UT_Yc)
            bias = Y_mean - X_mean @ W

            scores = X_val @ W + bias
            pred = np.argmax(scores, axis=1)
            accs.append(np.mean(pred == y_val))
        results.append(("ridge", alpha, np.mean(accs)))

    # --- LDA CV ---
    lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]
    for lam in lambdas:
        accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]

            # Class means and priors
            mu = np.zeros((n_classes, d))
            priors = np.zeros(n_classes)
            for c in range(n_classes):
                idx = np.where(y_train == c)[0]
                if len(idx) > 0:
                    mu[c] = X_train[idx].mean(axis=0)
                    priors[c] = len(idx) / len(y_train)
                else:
                    mu[c] = 0.0
                    priors[c] = 1e-6

            # Within-class scatter
            global_mean = X_train.mean(axis=0)
            Xc = X_train - global_mean
            S_t = Xc.T @ Xc
            S_b = np.zeros((d, d))
            for c in range(n_classes):
                n_c = np.sum(y_train == c)
                if n_c > 0:
                    diff = mu[c] - global_mean
                    S_b += n_c * np.outer(diff, diff)
            S_w = S_t - S_b

            # Eigendecomposition of S_w
            s, V = np.linalg.eigh(S_w)

            for lam_val in [lam]:  # single lambda per iteration
                inv_s = 1.0 / (s + lam_val)
                W = V @ (inv_s[:, None] * (V.T @ mu.T))
                bias = np.zeros(n_classes)
                for c in range(n_classes):
                    bias[c] = -0.5 * mu[c] @ W[:, c] + np.log(priors[c] + 1e-12)

                scores = X_val @ W + bias
                pred = np.argmax(scores, axis=1)
                accs.append(np.mean(pred == y_val))
        results.append(("lda", lam, np.mean(accs)))

    # --- Nearest Centroid CV ---
    accs = []
    for i in range(n_folds):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]

        mu = np.zeros((n_classes, d))
        for c in range(n_classes):
            idx = np.where(y_train == c)[0]
            if len(idx) > 0:
                mu[c] = X_train[idx].mean(axis=0)

        scores = X_val @ (2 * mu.T) - np.sum(mu**2, axis=1)
        pred = np.argmax(scores, axis=1)
        accs.append(np.mean(pred == y_val))
    results.append(("centroid", None, np.mean(accs)))

    # Select best method
    best_method, best_param, best_acc = max(results, key=lambda x: x[2])

    # --- Refit on full data ---
    if best_method == "ridge":
        alpha = best_param
        X_mean = X.mean(axis=0)
        Y_mean = Y.mean(axis=0)
        Xc = X - X_mean
        Yc = Y - Y_mean
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        UT_Yc = U.T @ Yc
        D = S / (S**2 + alpha)
        W = Vt.T @ (D[:, None] * UT_Yc)
        bias = Y_mean - X_mean @ W

    elif best_method == "lda":
        lam = best_param
        mu = np.zeros((n_classes, d))
        priors = np.zeros(n_classes)
        for c in range(n_classes):
            idx = np.where(y == c)[0]
            mu[c] = X[idx].mean(axis=0)
            priors[c] = len(idx) / n

        global_mean = X.mean(axis=0)
        Xc = X - global_mean
        S_t = Xc.T @ Xc
        S_b = np.zeros((d, d))
        for c in range(n_classes):
            n_c = np.sum(y == c)
            diff = mu[c] - global_mean
            S_b += n_c * np.outer(diff, diff)
        S_w = S_t - S_b

        s, V = np.linalg.eigh(S_w)
        inv_s = 1.0 / (s + lam)
        W = V @ (inv_s[:, None] * (V.T @ mu.T))
        bias = np.zeros(n_classes)
        for c in range(n_classes):
            bias[c] = -0.5 * mu[c] @ W[:, c] + np.log(priors[c] + 1e-12)

    else:  # centroid
        mu = np.zeros((n_classes, d))
        for c in range(n_classes):
            idx = np.where(y == c)[0]
            mu[c] = X[idx].mean(axis=0)
        W = 2 * mu.T
        bias = -np.sum(mu**2, axis=1)

    # Convert to head weight shape [10, 1568]
    head_weight = W.T.astype(np.float32)
    head_bias = bias.astype(np.float32)

    # Clip to [-1000, 1000]
    head_weight = np.clip(head_weight, -1000.0, 1000.0)
    head_bias = np.clip(head_bias, -1000.0, 1000.0)

    # Update weights
    w["head.weight"] = head_weight
    w["head.bias"] = head_bias

    diagnostics = {
        "best_method": best_method,
        "best_param": best_param,
        "best_cv_acc": float(best_acc),
        "all_results": [(m, p, float(a)) for m, p, a in results],
        "head_weight_max_abs": float(np.max(np.abs(head_weight))),
        "head_bias_max_abs": float(np.max(np.abs(head_bias))),
    }

    return {"weights": w, "diagnostics": diagnostics}
