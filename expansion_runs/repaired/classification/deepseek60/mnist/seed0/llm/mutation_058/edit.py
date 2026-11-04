import numpy as np

def edit(w, f):
    # Extract feedback arrays
    H = f["features"].astype(np.float64)  # (2000, 128)
    y = f["y"].astype(np.int64)           # (2000,)
    N, D = H.shape
    C = 10
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Current fc2 parameters
    W0 = w["fc2.weight"].astype(np.float64)  # (10, 128)
    b0 = w["fc2.bias"].astype(np.float64)    # (10,)

    # Current logits and accuracy on full pool
    L0 = H @ W0.T + b0
    pred0 = np.argmax(L0, axis=1)
    acc0 = np.mean(pred0 == y)

    # Standardize features
    H_mean = H.mean(axis=0)
    H_std = H.std(axis=0)
    H_std[H_std < 1e-6] = 1.0
    H_norm = (H - H_mean) / H_std

    # Add bias column
    H_aug = np.hstack([H_norm, np.ones((N, 1))])  # (2000, 129)

    # 5-fold cross-validation indices
    np.random.seed(0)
    idx = np.random.permutation(N)
    folds = np.array_split(idx, 5)

    # Helper: fit ridge on given train indices and lambda, return W_orig, b_orig
    def fit_ridge(train_idx, lam):
        H_tr = H_aug[train_idx]
        Y_tr = Y[train_idx]
        # Ridge normal equations: (H^T H + lam I) W = H^T Y
        A = H_tr.T @ H_tr + lam * np.eye(H_aug.shape[1])
        B = H_tr.T @ Y_tr
        W_aug = np.linalg.solve(A, B).T  # (10, 129)
        W_std = W_aug[:, :D]
        b_std = W_aug[:, D]
        # Convert back to original feature space
        W_orig = W_std / H_std
        b_orig = b_std - W_std @ (H_mean / H_std)
        return W_orig, b_orig

    # Helper: evaluate a model on validation indices
    def eval_model(W, b, val_idx):
        L = H[val_idx] @ W.T + b
        pred = np.argmax(L, axis=1)
        return np.mean(pred == y[val_idx])

    # Evaluate current model on the same folds
    cv_acc0 = []
    for val_idx in folds:
        cv_acc0.append(eval_model(W0, b0, val_idx))
    cv_acc0 = np.mean(cv_acc0)

    # Grid of lambda values for ridge
    lambdas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    best_lam = None
    best_cv_acc = -1.0
    best_W = None
    best_b = None

    for lam in lambdas:
        fold_accs = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            W, b = fit_ridge(train_idx, lam)
            fold_accs.append(eval_model(W, b, val_idx))
        mean_acc = np.mean(fold_accs)
        if mean_acc > best_cv_acc:
            best_cv_acc = mean_acc
            best_lam = lam
            # Fit on all data with this lambda
            best_W, best_b = fit_ridge(np.arange(N), lam)

    # Also try LDA with shrinkage
    # Compute class means and pooled covariance
    class_means = np.zeros((C, D))
    for c in range(C):
        class_means[c] = H_norm[y == c].mean(axis=0)
    overall_mean = H_norm.mean(axis=0)
    Sw = np.zeros((D, D))
    for c in range(C):
        Hc = H_norm[y == c] - class_means[c]
        Sw += Hc.T @ Hc
    Sw /= (N - C)

    lda_lambdas = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_lda_cv = -1.0
    best_lda_W = None
    best_lda_b = None
    best_lda_lam = None

    for lam in lda_lambdas:
        # Regularized LDA: W = (Sw + lam I)^-1 (mu_c - mu)
        Sw_reg = Sw + lam * np.eye(D)
        # Solve for each class direction
        W_lda = np.zeros((C, D))
        for c in range(C):
            diff = class_means[c] - overall_mean
            W_lda[c] = np.linalg.solve(Sw_reg, diff)
        # Bias: we can set bias to make class means equal? For simplicity, use zero bias.
        # Better: compute bias by centering logits on training data.
        # We'll compute bias on the fly during CV.
        fold_accs = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            # Recompute W_lda on train? Actually LDA uses all data for covariance. For CV, we should fit on train only.
            # To be proper, we should recompute Sw and means on train. But for speed, we can use the full data for covariance? That's leakage.
            # Let's do it properly but it's a bit more code. We'll just use the full data for LDA since it's a closed-form fit and we're evaluating on the same data? No, that's leakage.
            # For simplicity, we'll skip LDA and just use ridge, which we already CV'd properly.
            pass
        # Since LDA proper CV is more involved, we'll rely on ridge.

    # Decide whether to replace fc2
    # Use a small margin to avoid replacing if not clearly better
    if best_cv_acc > cv_acc0 + 0.001:
        W_new = best_W
        b_new = best_b
        # Clip to safe range
        W_new = np.clip(W_new, -1000.0, 1000.0)
        b_new = np.clip(b_new, -1000.0, 1000.0)
        w["fc2.weight"] = W_new.astype(np.float32)
        w["fc2.bias"] = b_new.astype(np.float32)
        method = "ridge_replaced"
    else:
        method = "keep_original"

    # Diagnostics
    diagnostics = {
        "method": method,
        "current_cv_acc": float(cv_acc0),
        "best_ridge_cv_acc": float(best_cv_acc),
        "best_lambda": float(best_lam) if best_lam is not None else None,
        "current_full_acc": float(acc0),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
    }

    return {"weights": w, "diagnostics": diagnostics}
