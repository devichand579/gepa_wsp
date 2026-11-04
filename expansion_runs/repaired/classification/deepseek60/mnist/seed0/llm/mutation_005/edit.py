import numpy as np

def edit(w, f):
    # Read feedback arrays
    X = f["features"].astype(np.float64)          # (2000, 128)
    y = f["y"].astype(np.int64)                   # (2000,)
    N, d = X.shape

    # Current fc2 parameters
    W_cur = w["fc2.weight"].astype(np.float64)    # (10, 128)
    b_cur = w["fc2.bias"].astype(np.float64)      # (10,)

    # Helper: accuracy
    def acc(X_, y_, W_, b_):
        logits = X_ @ W_.T + b_
        pred = np.argmax(logits, axis=1)
        return float(np.mean(pred == y_))

    # Helper: ridge fit on augmented features
    def fit_ridge(X_tr, y_tr, alpha):
        n = X_tr.shape[0]
        Y = np.zeros((n, 10))
        Y[np.arange(n), y_tr] = 1.0
        Xa = np.hstack([X_tr, np.ones((n, 1))])
        A = Xa.T @ Xa + alpha * np.eye(d + 1)
        B = Xa.T @ Y
        Wa = np.linalg.solve(A, B)
        W = Wa[:d, :].T
        b = Wa[d, :]
        return W, b

    # Helper: LDA with shrinkage
    def fit_lda(X_tr, y_tr, gamma):
        mu = np.zeros((10, d))
        priors = np.zeros(10)
        for c in range(10):
            idx = y_tr == c
            if np.any(idx):
                mu[c] = X_tr[idx].mean(axis=0)
                priors[c] = idx.mean()
            else:
                priors[c] = 1e-12
        S = np.zeros((d, d))
        for c in range(10):
            idx = y_tr == c
            if np.any(idx):
                diff = X_tr[idx] - mu[c]
                S += diff.T @ diff
        S /= max(1, len(y_tr) - 10)
        S_shrunk = (1.0 - gamma) * S + gamma * (np.trace(S) / d) * np.eye(d)
        W = np.linalg.solve(S_shrunk, mu.T).T
        b = -0.5 * np.sum(mu * W, axis=1) + np.log(priors + 1e-12)
        return W, b

    # 5-fold CV splits
    rng = np.random.RandomState(0)
    indices = rng.permutation(N)
    folds = np.array_split(indices, 5)

    # Evaluate current fc2 on validation folds
    acc_cur = []
    for val_idx in folds:
        acc_cur.append(acc(X[val_idx], y[val_idx], W_cur, b_cur))
    best_name = "current"
    best_acc = float(np.mean(acc_cur))
    best_W = W_cur.copy()
    best_b = b_cur.copy()

    # Ridge candidates
    for alpha in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]:
        accs = []
        for val_idx in folds:
            train_idx = np.setdiff1d(np.arange(N), val_idx)
            W, b = fit_ridge(X[train_idx], y[train_idx], alpha)
            accs.append(acc(X[val_idx], y[val_idx], W, b))
        mean_acc = float(np.mean(accs))
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_name = f"ridge_alpha_{alpha}"
            best_W, best_b = fit_ridge(X, y, alpha)

    # LDA candidates
    for gamma in [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.9]:
        accs = []
        for val_idx in folds:
            train_idx = np.setdiff1d(np.arange(N), val_idx)
            W, b = fit_lda(X[train_idx], y[train_idx], gamma)
            accs.append(acc(X[val_idx], y[val_idx], W, b))
        mean_acc = float(np.mean(accs))
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_name = f"lda_gamma_{gamma}"
            best_W, best_b = fit_lda(X, y, gamma)

    # Bayesian prior update: shrink ridge solution toward current fc2
    for lam in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]:
        accs = []
        for val_idx in folds:
            train_idx = np.setdiff1d(np.arange(N), val_idx)
            X_tr, y_tr = X[train_idx], y[train_idx]
            n = X_tr.shape[0]
            Y = np.zeros((n, 10))
            Y[np.arange(n), y_tr] = 1.0
            Xa = np.hstack([X_tr, np.ones((n, 1))])
            # Prior augmented parameters
            W_prior_aug = np.vstack([W_cur.T, b_cur])  # (d+1, 10)
            A = Xa.T @ Xa + lam * np.eye(d + 1)
            B = Xa.T @ Y + lam * W_prior_aug
            Wa = np.linalg.solve(A, B)
            W = Wa[:d, :].T
            b = Wa[d, :]
            accs.append(acc(X[val_idx], y[val_idx], W, b))
        mean_acc = float(np.mean(accs))
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_name = f"prior_lam_{lam}"
            # Refit on full pool
            n = N
            Y = np.zeros((n, 10))
            Y[np.arange(n), y] = 1.0
            Xa = np.hstack([X, np.ones((n, 1))])
            W_prior_aug = np.vstack([W_cur.T, b_cur])
            A = Xa.T @ Xa + lam * np.eye(d + 1)
            B = Xa.T @ Y + lam * W_prior_aug
            Wa = np.linalg.solve(A, B)
            best_W = Wa[:d, :].T
            best_b = Wa[d, :]

    # Simple blend between current and ridge solution
    # First get a reasonable ridge solution (alpha=1.0) on full pool
    W_ridge, b_ridge = fit_ridge(X, y, 1.0)
    for alpha_blend in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        W_blend = (1.0 - alpha_blend) * W_cur + alpha_blend * W_ridge
        b_blend = (1.0 - alpha_blend) * b_cur + alpha_blend * b_ridge
        accs = []
        for val_idx in folds:
            # For blend, we need to fit ridge on training fold only to avoid leakage
            train_idx = np.setdiff1d(np.arange(N), val_idx)
            W_r, b_r = fit_ridge(X[train_idx], y[train_idx], 1.0)
            W_b = (1.0 - alpha_blend) * W_cur + alpha_blend * W_r
            b_b = (1.0 - alpha_blend) * b_cur + alpha_blend * b_r
            accs.append(acc(X[val_idx], y[val_idx], W_b, b_b))
        mean_acc = float(np.mean(accs))
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_name = f"blend_alpha_{alpha_blend}"
            best_W = (1.0 - alpha_blend) * W_cur + alpha_blend * W_ridge
            best_b = (1.0 - alpha_blend) * b_cur + alpha_blend * b_ridge

    # Write chosen fc2 weights
    w["fc2.weight"] = best_W.astype(np.float32)
    w["fc2.bias"] = best_b.astype(np.float32)

    # Diagnostics
    diag = {
        "chosen": best_name,
        "cv_accuracy": best_acc,
        "current_cv_accuracy": float(np.mean(acc_cur)),
        "fc2_weight_norm": float(np.linalg.norm(best_W)),
        "fc2_bias_norm": float(np.linalg.norm(best_b)),
        "full_train_accuracy": acc(X, y, best_W, best_b),
    }
    return {"weights": w, "diagnostics": diag}
