import numpy as np

def edit(w, f):
    # Extract full feedback pool
    X = f["features"].astype(np.float64)  # [2000, 128]
    y = f["y"].astype(np.int64)           # [2000]
    N, D = X.shape
    C = 10

    # Current fc2 parameters
    W_cur = w["fc2.weight"].astype(np.float64)  # [10, 128]
    b_cur = w["fc2.bias"].astype(np.float64)    # [10]

    # Current full-pool accuracy
    logits_cur = X @ W_cur.T + b_cur
    acc_cur = float((np.argmax(logits_cur, axis=1) == y).mean())

    # ---------- closed-form fitters ----------
    def fit_ridge(X_train, y_train, lam):
        Y = np.zeros((len(y_train), C))
        Y[np.arange(len(y_train)), y_train] = 1.0
        X_mean = X_train.mean(axis=0)
        Y_mean = Y.mean(axis=0)
        Xc = X_train - X_mean
        Yc = Y - Y_mean
        A = Xc.T @ Xc + lam * np.eye(D)
        B = Xc.T @ Yc
        W = np.linalg.solve(A, B)          # D x C
        b = Y_mean - X_mean @ W
        return W.T, b                      # C x D, C

    def fit_lda(X_train, y_train, lam):
        mu = np.zeros((C, D))
        for c in range(C):
            mu[c] = X_train[y_train == c].mean(axis=0)
        S = np.zeros((D, D))
        for c in range(C):
            Xc = X_train[y_train == c] - mu[c]
            S += Xc.T @ Xc
        Sigma = S / (len(y_train) - C)
        Sigma_reg = Sigma + lam * np.eye(D)
        Sigma_inv = np.linalg.inv(Sigma_reg)
        W = Sigma_inv @ mu.T               # D x C
        b = np.zeros(C)
        for c in range(C):
            b[c] = -0.5 * mu[c] @ Sigma_inv @ mu[c] + np.log(np.mean(y_train == c))
        return W.T, b                      # C x D, C

    # ---------- cross-validation ----------
    def cv_score(fit_func, X_data, y_data, lambdas, n_folds=5):
        N = len(y_data)
        indices = np.arange(N)
        np.random.seed(42)
        np.random.shuffle(indices)
        folds = np.array_split(indices, n_folds)
        best_lam = None
        best_acc = -1.0
        for lam in lambdas:
            accs = []
            for i in range(n_folds):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
                X_train, y_train = X_data[train_idx], y_data[train_idx]
                X_val, y_val = X_data[val_idx], y_data[val_idx]
                W, b = fit_func(X_train, y_train, lam)
                logits = X_val @ W.T + b
                pred = np.argmax(logits, axis=1)
                accs.append((pred == y_val).mean())
            mean_acc = float(np.mean(accs))
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_lam = lam
        return best_lam, best_acc

    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]

    # Raw features
    best_lam_ridge, best_acc_ridge = cv_score(fit_ridge, X, y, lambdas)
    best_lam_lda, best_acc_lda = cv_score(fit_lda, X, y, lambdas)

    # Standardized features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-6] = 1.0
    X_std_data = (X - X_mean) / X_std

    best_lam_ridge_std, best_acc_ridge_std = cv_score(fit_ridge, X_std_data, y, lambdas)
    best_lam_lda_std, best_acc_lda_std = cv_score(fit_lda, X_std_data, y, lambdas)

    # Choose best method
    candidates = [
        ("ridge_raw", best_acc_ridge, best_lam_ridge, fit_ridge, X, False),
        ("lda_raw", best_acc_lda, best_lam_lda, fit_lda, X, False),
        ("ridge_std", best_acc_ridge_std, best_lam_ridge_std, fit_ridge, X_std_data, True),
        ("lda_std", best_acc_lda_std, best_lam_lda_std, fit_lda, X_std_data, True),
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)
    method_name, best_cv_acc, best_lam, fit_func, X_used, is_std = candidates[0]

    # If the best CV accuracy is not clearly better than the current in-sample accuracy,
    # keep the current fc2 to avoid degrading a well-trained classifier.
    if best_cv_acc < acc_cur - 0.002:
        # Keep current weights
        W_new = W_cur
        b_new = b_cur
        used_method = "keep_current"
        used_lam = None
    else:
        # Fit on all data
        W_new, b_new = fit_func(X_used, y, best_lam)
        if is_std:
            # Convert from standardized space back to original feature space
            # Z = (X - mean) / std  =>  X = Z * std + mean
            # logits = Z @ W_Z.T + b_Z = ((X - mean)/std) @ W_Z.T + b_Z
            #        = X @ (W_Z / std).T + (b_Z - mean @ (W_Z / std).T)
            W_new = W_new / X_std  # broadcasting: [C, D] / [D] -> [C, D]
            b_new = b_new - X_mean @ W_new.T
        used_method = method_name
        used_lam = best_lam

    # Ensure float32 and finite, clip to safe range
    W_new = np.clip(W_new, -1000.0, 1000.0).astype(np.float32)
    b_new = np.clip(b_new, -1000.0, 1000.0).astype(np.float32)

    # Write back
    w["fc2.weight"] = W_new
    w["fc2.bias"] = b_new

    # Diagnostics
    logits_new = X @ W_new.T + b_new
    acc_new = float((np.argmax(logits_new, axis=1) == y).mean())
    diagnostics = {
        "method": used_method,
        "lambda": used_lam,
        "cv_accuracy": best_cv_acc,
        "current_full_accuracy": acc_cur,
        "new_full_accuracy": acc_new,
        "fc2_weight_norm": float(np.linalg.norm(W_new)),
        "fc2_bias_norm": float(np.linalg.norm(b_new)),
    }
    return {"weights": w, "diagnostics": diagnostics}
