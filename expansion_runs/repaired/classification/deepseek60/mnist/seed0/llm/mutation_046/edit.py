import numpy as np

def edit(w, f):
    # Extract full feedback pool
    X = f["features"].astype(np.float64)          # (2000, 128)
    y = f["y"].astype(np.int64)                   # (2000,)
    N, D = X.shape
    num_classes = 10
    Y = np.zeros((N, num_classes), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Current fc2 parameters
    W_cur = w["fc2.weight"].astype(np.float64)    # (10, 128)
    b_cur = w["fc2.bias"].astype(np.float64)      # (10,)
    W_cur_orig = W_cur.T                          # (128, 10)
    b_cur_orig = b_cur.reshape(1, 10)             # (1, 10)

    # Current full-pool accuracy
    logits_cur = f["logits"]
    preds_cur = np.argmax(logits_cur, axis=1)
    acc_cur = float(np.mean(preds_cur == y))

    # ---------- Ridge regression (standardized features) ----------
    def fit_ridge(X_train, Y_train, lambda_, use_prior=False):
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        std[std < 1e-6] = 1.0
        X_train_std = (X_train - mean) / std
        N_train = X_train_std.shape[0]
        Xb = np.concatenate([X_train_std, np.ones((N_train, 1))], axis=1)
        D_plus = Xb.shape[1]
        A = Xb.T @ Xb
        reg = lambda_ * np.eye(D_plus)
        reg[-1, -1] = 0.0                         # do not penalize bias
        rhs = Xb.T @ Y_train
        if use_prior:
            W_prior_std = W_cur_orig * std.reshape(-1, 1)
            b_prior_std = mean @ W_cur_orig + b_cur_orig
            W0 = np.vstack([W_prior_std, b_prior_std])
            rhs += reg @ W0
        W = np.linalg.solve(A + reg, rhs)
        return W, mean, std

    # ---------- Regularized LDA ----------
    def fit_lda(X_train, y_train, lambda_):
        mu_c = np.zeros((num_classes, D))
        priors = np.zeros(num_classes)
        for c in range(num_classes):
            idx = (y_train == c)
            if np.sum(idx) > 0:
                mu_c[c] = np.mean(X_train[idx], axis=0)
                priors[c] = np.sum(idx) / len(y_train)
            else:
                mu_c[c] = 0.0
                priors[c] = 1e-6
        S_w = np.zeros((D, D))
        for c in range(num_classes):
            idx = (y_train == c)
            if np.sum(idx) > 0:
                diff = X_train[idx] - mu_c[c]
                S_w += diff.T @ diff
        S_w_reg = S_w + lambda_ * np.eye(D)
        inv_S_w = np.linalg.inv(S_w_reg)
        W = inv_S_w @ mu_c.T                      # (D, num_classes)
        b = np.zeros(num_classes)
        for c in range(num_classes):
            b[c] = -0.5 * mu_c[c] @ inv_S_w @ mu_c[c] + np.log(priors[c] + 1e-12)
        return W, b

    # ---------- Cross-validation ----------
    lambdas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    n_folds = 5
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    best_cv_acc = -1.0
    best_method = None
    best_params = None

    # Evaluate ridge variants
    for use_prior in [False, True]:
        for lam in lambdas:
            accs = []
            for i in range(n_folds):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
                X_train, Y_train = X[train_idx], Y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]
                W_std, mean, std = fit_ridge(X_train, Y_train, lam, use_prior)
                X_val_std = (X_val - mean) / std
                Xb_val = np.concatenate([X_val_std, np.ones((X_val.shape[0], 1))], axis=1)
                scores = Xb_val @ W_std
                preds = np.argmax(scores, axis=1)
                accs.append(np.mean(preds == y_val))
            mean_acc = float(np.mean(accs))
            if mean_acc > best_cv_acc:
                best_cv_acc = mean_acc
                best_method = "ridge"
                best_params = (lam, use_prior)

    # Evaluate LDA variants
    for lam in lambdas:
        accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            W_lda, b_lda = fit_lda(X_train, y_train, lam)
            scores = X_val @ W_lda + b_lda
            preds = np.argmax(scores, axis=1)
            accs.append(np.mean(preds == y_val))
        mean_acc = float(np.mean(accs))
        if mean_acc > best_cv_acc:
            best_cv_acc = mean_acc
            best_method = "lda"
            best_params = (lam,)

    # ---------- Fit best method on full pool ----------
    if best_method == "ridge":
        lam, use_prior = best_params
        W_std_full, mean_full, std_full = fit_ridge(X, Y, lam, use_prior)
        W_orig = W_std_full[:D, :] / std_full.reshape(-1, 1)
        b_orig = W_std_full[D, :] - mean_full @ W_orig
        new_fc2_weight = W_orig.T                  # (10, 128)
        new_fc2_bias = b_orig.reshape(10)
    else:  # lda
        lam = best_params[0]
        W_lda, b_lda = fit_lda(X, y, lam)
        new_fc2_weight = W_lda.T                   # (10, 128)
        new_fc2_bias = b_lda.reshape(10)

    # Clip and cast
    new_fc2_weight = np.clip(new_fc2_weight, -1000, 1000).astype(np.float32)
    new_fc2_bias = np.clip(new_fc2_bias, -1000, 1000).astype(np.float32)

    # Evaluate full-pool accuracy of the candidate
    logits_new = X @ new_fc2_weight.T.astype(np.float64) + new_fc2_bias.astype(np.float64)
    preds_new = np.argmax(logits_new, axis=1)
    acc_new = float(np.mean(preds_new == y))

    # Only apply if it improves full-pool accuracy
    if acc_new > acc_cur:
        w["fc2.weight"] = new_fc2_weight
        w["fc2.bias"] = new_fc2_bias
        applied = True
    else:
        applied = False

    diagnostics = {
        "method": best_method,
        "best_params": list(best_params) if best_params is not None else None,
        "cv_accuracy": best_cv_acc,
        "full_acc_current": acc_cur,
        "full_acc_candidate": acc_new,
        "applied": applied,
        "fc2_weight_norm": float(np.linalg.norm(new_fc2_weight)),
        "fc2_bias_norm": float(np.linalg.norm(new_fc2_bias)),
    }
    return {"weights": w, "diagnostics": diagnostics}
