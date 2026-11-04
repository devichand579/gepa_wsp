import numpy as np

def edit(w, f):
    # Extract arrays
    X = f["features"].astype(np.float64)          # [2000, 128]
    y = f["y"].astype(np.int64)                   # [2000]
    N, D = X.shape
    C = 10

    # Current fc2 parameters
    W0 = w["fc2.weight"].astype(np.float64)       # [10, 128]
    b0 = w["fc2.bias"].astype(np.float64)         # [10]

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Standardize features
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    Xs = (X - mu) / sigma

    # Convert prior to standardized space
    # logits = X W^T + b = Xs (W * sigma)^T + (mu W^T + b)
    W0s = W0 * sigma[None, :]                     # [10, 128]
    b0s = b0 + mu @ W0.T                          # [10]

    # Augment with bias column
    X_aug = np.hstack([Xs, np.ones((N, 1))])      # [2000, 129]
    W0_aug = np.hstack([W0s, b0s[:, None]])       # [10, 129]

    # Helper: ridge solution with prior
    def ridge_solve(Xtr, Ytr, W0_aug, lam):
        XtX = Xtr.T @ Xtr
        XtY = Xtr.T @ Ytr
        # W_aug = (XtX + lam I)^{-1} (XtY + lam W0_aug^T)
        # XtY is [129, 10], W0_aug^T is [129, 10]
        A = XtX + lam * np.eye(XtX.shape[0])
        B = XtY + lam * W0_aug.T
        W_aug = np.linalg.solve(A, B).T           # [10, 129]
        return W_aug

    # Cross-validation to choose lambda
    lambdas = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e4]
    n_folds = 5
    fold_size = N // n_folds
    best_lam = None
    best_acc = -1.0

    # Baseline: current model accuracy on full pool
    logits_cur = X @ W0.T + b0
    pred_cur = np.argmax(logits_cur, axis=1)
    acc_cur = np.mean(pred_cur == y)

    for lam in lambdas:
        accs = []
        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else N
            val_idx = np.arange(val_start, val_end)
            tr_idx = np.setdiff1d(np.arange(N), val_idx)

            Xtr = X_aug[tr_idx]
            Ytr = Y[tr_idx]
            Xval = X_aug[val_idx]
            Yval = Y[val_idx]

            W_aug = ridge_solve(Xtr, Ytr, W0_aug, lam)
            logits_val = Xval @ W_aug.T
            pred_val = np.argmax(logits_val, axis=1)
            acc = np.mean(pred_val == np.argmax(Yval, axis=1))
            accs.append(acc)
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lam = lam

    # If best CV accuracy is not better than current model, keep current
    # We compare against current model's CV accuracy? We don't have it directly.
    # But we can compute current model's accuracy on the full pool as a proxy.
    # Actually, we can compute current model's CV accuracy by using lam = infinity.
    # Let's compute current model's CV accuracy:
    accs_cur = []
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else N
        val_idx = np.arange(val_start, val_end)
        Xval = X_aug[val_idx]
        logits_val = Xval @ W0_aug.T
        pred_val = np.argmax(logits_val, axis=1)
        acc = np.mean(pred_val == np.argmax(Y[val_idx], axis=1))
        accs_cur.append(acc)
    acc_cur_cv = np.mean(accs_cur)

    if best_acc <= acc_cur_cv + 1e-6:
        # No improvement, keep current
        w["fc2.weight"] = W0.astype(np.float32)
        w["fc2.bias"] = b0.astype(np.float32)
        diagnostics = {
            "method": "keep_current",
            "best_lam": best_lam,
            "best_cv_acc": float(best_acc),
            "current_cv_acc": float(acc_cur_cv),
            "current_full_acc": float(acc_cur)
        }
        return {"weights": w, "diagnostics": diagnostics}

    # Fit final model on full pool with best lambda
    W_aug_final = ridge_solve(X_aug, Y, W0_aug, best_lam)
    Ws_final = W_aug_final[:, :D]
    bs_final = W_aug_final[:, D]

    # Convert back to original feature space
    # Ws_final = W * sigma  =>  W = Ws_final / sigma
    W_final = Ws_final / sigma[None, :]
    b_final = bs_final - mu @ W_final.T

    # Clip to safe range
    W_final = np.clip(W_final, -1000.0, 1000.0)
    b_final = np.clip(b_final, -1000.0, 1000.0)

    w["fc2.weight"] = W_final.astype(np.float32)
    w["fc2.bias"] = b_final.astype(np.float32)

    diagnostics = {
        "method": "ridge_prior_standardized",
        "best_lam": float(best_lam),
        "best_cv_acc": float(best_acc),
        "current_cv_acc": float(acc_cur_cv),
        "current_full_acc": float(acc_cur),
        "new_full_acc": float(np.mean(np.argmax(X @ W_final.T + b_final, axis=1) == y)),
        "fc2_weight_norm": float(np.linalg.norm(W_final)),
        "fc2_bias_norm": float(np.linalg.norm(b_final))
    }
    return {"weights": w, "diagnostics": diagnostics}
