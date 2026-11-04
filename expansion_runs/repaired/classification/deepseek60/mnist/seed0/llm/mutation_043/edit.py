import numpy as np

def edit(w, f):
    # Extract data
    X = f["features"].astype(np.float64)          # [2000, 128]
    y = f["y"].astype(np.int64)                   # [2000]
    N, D = X.shape
    C = 10

    # Current fc2 parameters
    W0 = w["fc2.weight"].astype(np.float64)       # [10, 128]
    b0 = w["fc2.bias"].astype(np.float64)         # [10]

    # Current logits and predictions
    Z0 = X @ W0.T + b0                            # [2000, 10]
    pred0 = np.argmax(Z0, axis=1)
    acc0 = np.mean(pred0 == y)

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Residuals from current logits to one-hot targets
    R = Y - Z0                                    # [2000, 10]

    # Augment features with a bias column
    X_aug = np.hstack([X, np.ones((N, 1))])       # [2000, 129]

    # Precompute some matrices for CV
    # We'll use 5-fold CV
    n_folds = 5
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    # Hyperparameter grids
    lambdas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    lambda_bs = [0.01, 0.1, 1.0, 10.0, 100.0]
    alphas = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

    best_acc = acc0
    best_params = None
    best_delta = None

    # Helper to solve weighted ridge for a given alpha, lambda, lambda_b on a subset
    def solve_correction(idx_train, alpha, lam, lam_b):
        X_tr = X_aug[idx_train]
        R_tr = R[idx_train]
        pred_tr = pred0[idx_train]
        y_tr = y[idx_train]
        # Weights: 1 + alpha * (misclassified)
        weights = 1.0 + alpha * (pred_tr != y_tr).astype(np.float64)
        # Weighted normal equations
        # X_aug^T D X_aug + Lambda
        # D = diag(weights)
        Xw = X_tr * weights[:, None]
        XtDX = X_tr.T @ Xw                         # [129, 129]
        XtDR = X_tr.T @ (R_tr * weights[:, None])  # [129, 10]
        # Regularization matrix: lambda for first 128, lambda_b for bias
        Lambda = np.eye(D + 1) * lam
        Lambda[D, D] = lam_b
        # Solve for delta (129, 10)
        delta = np.linalg.solve(XtDX + Lambda, XtDR)
        return delta

    # Evaluate a candidate on validation fold
    def eval_correction(idx_train, idx_val, alpha, lam, lam_b):
        delta = solve_correction(idx_train, alpha, lam, lam_b)
        dW = delta[:D, :].T          # [10, 128]
        db = delta[D, :]             # [10]
        Z_val = X_aug[idx_val] @ delta  # [n_val, 10]
        pred_val = np.argmax(Z_val, axis=1)
        return np.mean(pred_val == y[idx_val])

    # Also try direct ridge (no prior, i.e., W0=0, b0=0)
    def solve_direct(idx_train, lam, lam_b):
        X_tr = X_aug[idx_train]
        Y_tr = Y[idx_train]
        XtX = X_tr.T @ X_tr
        XtY = X_tr.T @ Y_tr
        Lambda = np.eye(D + 1) * lam
        Lambda[D, D] = lam_b
        W_aug = np.linalg.solve(XtX + Lambda, XtY)
        return W_aug

    def eval_direct(idx_train, idx_val, lam, lam_b):
        W_aug = solve_direct(idx_train, lam, lam_b)
        Z_val = X_aug[idx_val] @ W_aug
        pred_val = np.argmax(Z_val, axis=1)
        return np.mean(pred_val == y[idx_val])

    # LDA with shrinkage
    def solve_lda(idx_train, shrinkage):
        X_tr = X[idx_train]
        y_tr = y[idx_train]
        means = np.zeros((C, D))
        for c in range(C):
            means[c] = X_tr[y_tr == c].mean(axis=0)
        # Pooled covariance
        S = np.zeros((D, D))
        for c in range(C):
            Xc = X_tr[y_tr == c]
            if len(Xc) > 1:
                Xc_centered = Xc - means[c]
                S += Xc_centered.T @ Xc_centered
        n_total = len(idx_train)
        cov = S / (n_total - C)
        # Shrinkage
        trace = np.trace(cov)
        cov_shrunk = (1 - shrinkage) * cov + shrinkage * (trace / D) * np.eye(D)
        # Solve for W
        # W_c = cov_shrunk^-1 @ means[c]
        # b_c = -0.5 * means[c] @ W_c + log(prior_c)
        # priors are equal
        W_lda = np.linalg.solve(cov_shrunk, means.T).T  # [C, D]
        b_lda = -0.5 * np.sum(means * W_lda, axis=1)    # [C]
        return W_lda, b_lda

    def eval_lda(idx_train, idx_val, shrinkage):
        W_lda, b_lda = solve_lda(idx_train, shrinkage)
        Z_val = X[idx_val] @ W_lda.T + b_lda
        pred_val = np.argmax(Z_val, axis=1)
        return np.mean(pred_val == y[idx_val])

    # Cross-validation to select best method
    best_cv_acc = acc0
    best_method = "current"
    best_hyper = None

    # Evaluate current on CV (just accuracy on all, but we can compute per fold)
    # Actually current is fixed, so CV accuracy is just acc0. But we can still compare.

    # Try correction ridge
    for alpha in alphas:
        for lam in lambdas:
            for lam_b in lambda_bs:
                accs = []
                for i in range(n_folds):
                    val_idx = folds[i]
                    train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
                    acc = eval_correction(train_idx, val_idx, alpha, lam, lam_b)
                    accs.append(acc)
                cv_acc = np.mean(accs)
                if cv_acc > best_cv_acc:
                    best_cv_acc = cv_acc
                    best_method = "correction"
                    best_hyper = (alpha, lam, lam_b)

    # Try direct ridge
    for lam in lambdas:
        for lam_b in lambda_bs:
            accs = []
            for i in range(n_folds):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
                acc = eval_direct(train_idx, val_idx, lam, lam_b)
                accs.append(acc)
            cv_acc = np.mean(accs)
            if cv_acc > best_cv_acc:
                best_cv_acc = cv_acc
                best_method = "direct"
                best_hyper = (lam, lam_b)

    # Try LDA
    shrinkages = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99]
    for shrink in shrinkages:
        accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            acc = eval_lda(train_idx, val_idx, shrink)
            accs.append(acc)
        cv_acc = np.mean(accs)
        if cv_acc > best_cv_acc:
            best_cv_acc = cv_acc
            best_method = "lda"
            best_hyper = shrink

    # Now apply the best method on the full data
    if best_method == "current":
        # Keep current weights
        pass
    elif best_method == "correction":
        alpha, lam, lam_b = best_hyper
        delta = solve_correction(np.arange(N), alpha, lam, lam_b)
        dW = delta[:D, :].T
        db = delta[D, :]
        w["fc2.weight"] = (W0 + dW).astype(np.float32)
        w["fc2.bias"] = (b0 + db).astype(np.float32)
    elif best_method == "direct":
        lam, lam_b = best_hyper
        W_aug = solve_direct(np.arange(N), lam, lam_b)
        w["fc2.weight"] = W_aug[:D, :].T.astype(np.float32)
        w["fc2.bias"] = W_aug[D, :].astype(np.float32)
    elif best_method == "lda":
        shrink = best_hyper
        W_lda, b_lda = solve_lda(np.arange(N), shrink)
        w["fc2.weight"] = W_lda.astype(np.float32)
        w["fc2.bias"] = b_lda.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "method": best_method,
        "best_hyper": best_hyper if best_hyper is not None else None,
        "cv_accuracy": float(best_cv_acc),
        "current_accuracy": float(acc0),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
    }

    return {"weights": w, "diagnostics": diagnostics}
