import numpy as np

def edit(w, f):
    # Load full feedback pool
    F = f["features"].astype(np.float64)          # (2000, 128) relu(fc1(x))
    y = f["y"].astype(np.int64)                   # (2000,)
    N, D = F.shape
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Standardization statistics (computed on full pool)
    mu = F.mean(axis=0)
    sigma = F.std(axis=0)
    sigma[sigma < 1e-6] = 1.0
    F_std = (F - mu) / sigma

    # ---------- closed-form fitters ----------
    def fit_ridge(X, Y, alpha):
        # X: (N, D+1) with bias column already appended
        A = np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ Y)
        W = A[:D].T          # (C, D)
        b = A[D]             # (C,)
        return W, b

    def fit_lda(X, y, reg):
        N, D = X.shape
        C = 10
        mu_c = np.zeros((C, D))
        counts = np.zeros(C)
        for c in range(C):
            idx = (y == c)
            counts[c] = idx.sum()
            if counts[c] > 0:
                mu_c[c] = X[idx].mean(axis=0)

        # Pooled within-class covariance
        S = np.zeros((D, D))
        for c in range(C):
            idx = (y == c)
            if counts[c] > 1:
                Xc = X[idx] - mu_c[c]
                S += Xc.T @ Xc
        S /= max(N - C, 1)
        S += reg * np.eye(D)

        invS = np.linalg.inv(S)
        W = mu_c @ invS
        b = -0.5 * np.sum((mu_c @ invS) * mu_c, axis=1) + np.log(counts / N + 1e-12)
        return W, b

    # ---------- cross-validation to choose method & hyperparameter ----------
    def cv_accuracy(method, X, y, params, folds=5):
        N = X.shape[0]
        indices = np.arange(N)
        rng = np.random.RandomState(0)
        rng.shuffle(indices)
        fold_size = N // folds
        best_acc = -1.0
        best_param = None

        for param in params:
            accs = []
            for k in range(folds):
                val_idx = indices[k*fold_size:(k+1)*fold_size]
                train_idx = np.concatenate([indices[:k*fold_size], indices[(k+1)*fold_size:]])
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_val = X[val_idx]
                y_val = y[val_idx]

                if method == 'ridge':
                    X_train_b = np.hstack([X_train, np.ones((len(X_train), 1))])
                    X_val_b = np.hstack([X_val, np.ones((len(X_val), 1))])
                    W, b = fit_ridge(X_train_b, Y[train_idx], param)
                    logits = X_val_b @ np.vstack([W.T, b.reshape(1, -1)])
                else:  # lda
                    W, b = fit_lda(X_train, y_train, param)
                    logits = X_val @ W.T + b

                pred = logits.argmax(axis=1)
                accs.append((pred == y_val).mean())

            mean_acc = float(np.mean(accs))
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_param = param

        return best_param, best_acc

    # Hyperparameter grids
    alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]
    regs   = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]

    best_overall = None   # (cv_acc, method, use_std, param)

    for use_std in [False, True]:
        X = F_std if use_std else F

        # Ridge
        best_alpha, best_acc = cv_accuracy('ridge', X, y, alphas)
        if best_overall is None or best_acc > best_overall[0]:
            best_overall = (best_acc, 'ridge', use_std, best_alpha)

        # LDA
        best_reg, best_acc = cv_accuracy('lda', X, y, regs)
        if best_acc > best_overall[0]:
            best_overall = (best_acc, 'lda', use_std, best_reg)

    # ---------- final fit on full pool ----------
    cv_acc, method, use_std, param = best_overall
    X = F_std if use_std else F

    if method == 'ridge':
        X_b = np.hstack([X, np.ones((N, 1))])
        W, b = fit_ridge(X_b, Y, param)
    else:
        W, b = fit_lda(X, y, param)

    # De-standardize if needed
    if use_std:
        W = W / sigma
        b = b - W @ mu

    # Safety: finite and bounded
    W = np.clip(W, -1000.0, 1000.0).astype(np.float32)
    b = np.clip(b, -1000.0, 1000.0).astype(np.float32)

    # Write back to fc2 only
    w["fc2.weight"] = W
    w["fc2.bias"] = b

    # ---------- diagnostics ----------
    logits = F @ W.T + b
    pred = logits.argmax(axis=1)
    full_acc = float((pred == y).mean())

    # Stable cross-entropy
    logits_stable = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits_stable)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    ce = float(-np.mean(np.log(probs[np.arange(N), y] + 1e-12)))

    diagnostics = {
        "method": method,
        "use_std": bool(use_std),
        "param": float(param),
        "cv_accuracy": float(cv_acc),
        "full_accuracy": full_acc,
        "full_cross_entropy": ce,
        "fc2_weight_mean": float(W.mean()),
        "fc2_weight_std": float(W.std()),
        "fc2_bias_mean": float(b.mean()),
        "fc2_bias_std": float(b.std()),
    }

    return {"weights": w, "diagnostics": diagnostics}
