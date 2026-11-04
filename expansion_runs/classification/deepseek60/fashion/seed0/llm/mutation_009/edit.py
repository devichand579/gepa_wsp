import numpy as np

def edit(w, f):
    X = f["features"].astype(np.float64)          # (2000, 1568)
    y = f["y"].astype(np.int64)                   # (2000,)
    n, d = X.shape
    classes = np.unique(y)
    K = len(classes)

    # Class means and counts on full data
    mu = np.zeros((K, d))
    n_k = np.zeros(K)
    for k in classes:
        idx = (y == k)
        n_k[k] = idx.sum()
        mu[k] = X[idx].mean(axis=0)

    # Pooled within-class covariance: Sw = X^T X - sum n_k mu_k mu_k^T
    XtX = X.T @ X
    M = mu.T @ (mu * n_k[:, None])
    Sw = XtX - M

    # 5-fold cross-validation to choose LDA ridge lambda
    n_folds = 5
    indices = np.arange(n)
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    best_lambda = None
    best_cv_acc = -1.0

    for lam in lambdas:
        accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])

            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]

            mu_train = np.zeros((K, d))
            n_k_train = np.zeros(K)
            for k in classes:
                idx = (y_train == k)
                n_k_train[k] = idx.sum()
                if n_k_train[k] > 0:
                    mu_train[k] = X_train[idx].mean(axis=0)

            XtX_train = X_train.T @ X_train
            M_train = mu_train.T @ (mu_train * n_k_train[:, None])
            Sw_train = XtX_train - M_train

            A = Sw_train + lam * np.eye(d)
            W = np.linalg.solve(A, mu_train.T)          # (d, K)
            pi = n_k_train / n_k_train.sum()
            b = -0.5 * np.sum(mu_train * W.T, axis=1) + np.log(pi + 1e-12)

            scores = X_val @ W + b
            pred = np.argmax(scores, axis=1)
            accs.append((pred == y_val).mean())

        mean_acc = float(np.mean(accs))
        if mean_acc > best_cv_acc:
            best_cv_acc = mean_acc
            best_lambda = lam

    # Fit final LDA on all data
    A = Sw + best_lambda * np.eye(d)
    W = np.linalg.solve(A, mu.T)                    # (d, K)
    pi = n_k / n_k.sum()
    b = -0.5 * np.sum(mu * W.T, axis=1) + np.log(pi + 1e-12)

    # Optional logit scaling to reduce CE without changing argmax
    logits = X @ W + b
    def ce_for_scale(s):
        z = logits * s
        z = z - z.max(axis=1, keepdims=True)
        exp_z = np.exp(z)
        probs = exp_z / exp_z.sum(axis=1, keepdims=True)
        return -np.mean(np.log(probs[np.arange(n), y] + 1e-12))

    best_s = 1.0
    best_ce = ce_for_scale(1.0)
    for s in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
        c = ce_for_scale(s)
        if c < best_ce:
            best_ce = c
            best_s = s

    W = W * best_s
    b = b * best_s

    w["head.weight"] = W.T.astype(np.float32)       # (10, 1568)
    w["head.bias"] = b.astype(np.float32)

    # Keep conv1, conv2 unchanged
    # Return diagnostics
    diagnostics = {
        "best_lambda": float(best_lambda),
        "best_cv_accuracy": float(best_cv_acc),
        "best_scale": float(best_s),
        "best_ce": float(best_ce),
        "head_weight_norm": float(np.linalg.norm(w["head.weight"])),
        "head_bias_norm": float(np.linalg.norm(w["head.bias"])),
    }
    return {"weights": w, "diagnostics": diagnostics}
