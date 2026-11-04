import numpy as np

def edit(w, f):
    # Extract feedback arrays
    H = f["features"].astype(np.float64)          # (2000, 128)
    y = f["y"].astype(np.int64)                   # (2000,)
    N, D = H.shape
    C = 10

    # Current fc2 parameters
    W_cur = w["fc2.weight"].astype(np.float64)    # (10, 128)
    b_cur = w["fc2.bias"].astype(np.float64)      # (10,)

    # Current logits and accuracy on the full feedback pool
    logits_cur = H @ W_cur.T + b_cur
    acc_cur = np.mean(np.argmax(logits_cur, axis=1) == y)

    # One-hot targets
    Y = np.zeros((N, C))
    Y[np.arange(N), y] = 1.0

    # Standardize hidden features
    mean = H.mean(axis=0)
    std = H.std(axis=0)
    std[std < 1e-6] = 1.0
    H_std = (H - mean) / std

    # Augment with bias column
    X = np.hstack([H_std, np.ones((N, 1))])       # (N, D+1)

    # Cross-validation setup
    np.random.seed(0)
    indices = np.random.permutation(N)
    n_folds = 5
    fold_size = N // n_folds

    lambdas = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

    # Baseline CV accuracy for current model
    cv_acc_cur = []
    for fold in range(n_folds):
        val_idx = indices[fold*fold_size:(fold+1)*fold_size]
        val_y = y[val_idx]
        val_logits = logits_cur[val_idx]
        acc = np.mean(np.argmax(val_logits, axis=1) == val_y)
        cv_acc_cur.append(acc)
    cv_acc_cur_mean = np.mean(cv_acc_cur)

    best_cv_acc = -1.0
    best_method = None
    best_lam = None

    # --- Method 1: Ridge on one-hot targets ---
    for lam in lambdas:
        cv_accs = []
        for fold in range(n_folds):
            val_idx = indices[fold*fold_size:(fold+1)*fold_size]
            train_idx = np.setdiff1d(indices, val_idx)
            X_train = X[train_idx]
            Y_train = Y[train_idx]
            X_val = X[val_idx]
            reg = np.eye(D+1) * lam
            reg[-1, -1] = 0.0
            A = X_train.T @ X_train + reg
            B = X_train.T @ Y_train
            P = np.linalg.solve(A, B)
            logits_val = X_val @ P
            acc = np.mean(np.argmax(logits_val, axis=1) == y[val_idx])
            cv_accs.append(acc)
        cv_acc = np.mean(cv_accs)
        if cv_acc > best_cv_acc:
            best_cv_acc = cv_acc
            best_method = "ridge"
            best_lam = lam

    # --- Method 2: Ridge with shrinkage toward current fc2 ---
    # Convert current parameters to standardized space
    W_cur_std = W_cur * std[None, :]
    b_cur_std = b_cur + W_cur @ mean
    P_cur = np.vstack([W_cur_std.T, b_cur_std[None, :]])  # (D+1, C)

    for lam in lambdas:
        cv_accs = []
        for fold in range(n_folds):
            val_idx = indices[fold*fold_size:(fold+1)*fold_size]
            train_idx = np.setdiff1d(indices, val_idx)
            X_train = X[train_idx]
            Y_train = Y[train_idx]
            X_val = X[val_idx]
            R_train = Y_train - X_train @ P_cur
            reg = np.eye(D+1) * lam
            reg[-1, -1] = 0.0
            A = X_train.T @ X_train + reg
            B = X_train.T @ R_train
            dP = np.linalg.solve(A, B)
            P = P_cur + dP
            logits_val = X_val @ P
            acc = np.mean(np.argmax(logits_val, axis=1) == y[val_idx])
            cv_accs.append(acc)
        cv_acc = np.mean(cv_accs)
        if cv_acc > best_cv_acc:
            best_cv_acc = cv_acc
            best_method = "shrink"
            best_lam = lam

    # --- Method 3: Regularized LDA ---
    for lam in lambdas:
        cv_accs = []
        for fold in range(n_folds):
            val_idx = indices[fold*fold_size:(fold+1)*fold_size]
            train_idx = np.setdiff1d(indices, val_idx)
            H_train = H_std[train_idx]
            y_train = y[train_idx]
            H_val = H_std[val_idx]
            means = np.zeros((C, D))
            priors = np.zeros(C)
            for c in range(C):
                idx_c = np.where(y_train == c)[0]
                if len(idx_c) > 0:
                    means[c] = H_train[idx_c].mean(axis=0)
                    priors[c] = len(idx_c) / len(train_idx)
                else:
                    priors[c] = 1e-6
            S = np.zeros((D, D))
            for c in range(C):
                idx_c = np.where(y_train == c)[0]
                if len(idx_c) > 1:
                    diff = H_train[idx_c] - means[c]
                    S += diff.T @ diff
            S /= (len(train_idx) - C)
            S_reg = S + lam * np.eye(D)
            W_lda = np.linalg.solve(S_reg, means.T)  # (D, C)
            b_lda = -0.5 * np.sum(means * W_lda.T, axis=1) + np.log(priors)
            logits_val = H_val @ W_lda + b_lda
            acc = np.mean(np.argmax(logits_val, axis=1) == y[val_idx])
            cv_accs.append(acc)
        cv_acc = np.mean(cv_accs)
        if cv_acc > best_cv_acc:
            best_cv_acc = cv_acc
            best_method = "lda"
            best_lam = lam

    # --- Fit on full pool with best method and lambda ---
    if best_method == "ridge":
        reg = np.eye(D+1) * best_lam
        reg[-1, -1] = 0.0
        A = X.T @ X + reg
        B = X.T @ Y
        P = np.linalg.solve(A, B)
        W_new_std = P[:D, :].T
        b_new_std = P[D, :]
        W_new = W_new_std / std[None, :]
        b_new = b_new_std - W_new @ mean
    elif best_method == "shrink":
        R = Y - X @ P_cur
        reg = np.eye(D+1) * best_lam
        reg[-1, -1] = 0.0
        A = X.T @ X + reg
        B = X.T @ R
        dP = np.linalg.solve(A, B)
        P = P_cur + dP
        W_new_std = P[:D, :].T
        b_new_std = P[D, :]
        W_new = W_new_std / std[None, :]
        b_new = b_new_std - W_new @ mean
    elif best_method == "lda":
        means = np.zeros((C, D))
        priors = np.zeros(C)
        for c in range(C):
            idx_c = np.where(y == c)[0]
            means[c] = H_std[idx_c].mean(axis=0)
            priors[c] = len(idx_c) / N
        S = np.zeros((D, D))
        for c in range(C):
            idx_c = np.where(y == c)[0]
            diff = H_std[idx_c] - means[c]
            S += diff.T @ diff
        S /= (N - C)
        S_reg = S + best_lam * np.eye(D)
        W_lda = np.linalg.solve(S_reg, means.T)
        b_lda = -0.5 * np.sum(means * W_lda.T, axis=1) + np.log(priors)
        W_new_std = W_lda.T
        b_new_std = b_lda
        W_new = W_new_std / std[None, :]
        b_new = b_new_std - W_new @ mean
    else:
        W_new = W_cur
        b_new = b_cur

    # Evaluate new model on full pool
    logits_new = H @ W_new.T + b_new
    acc_new = np.mean(np.argmax(logits_new, axis=1) == y)

    # Keep current if new is worse
    if acc_new < acc_cur:
        W_new = W_cur
        b_new = b_cur
        acc_new = acc_cur

    # Update weights, clip to [-1000, 1000], cast to float32
    w["fc2.weight"] = np.clip(W_new, -1000, 1000).astype(np.float32)
    w["fc2.bias"] = np.clip(b_new, -1000, 1000).astype(np.float32)

    diagnostics = {
        "method": best_method,
        "lambda": best_lam,
        "cv_acc": float(best_cv_acc),
        "cv_acc_current": float(cv_acc_cur_mean),
        "full_acc_current": float(acc_cur),
        "full_acc_new": float(acc_new),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
    }
    return {"weights": w, "diagnostics": diagnostics}
