import numpy as np

def edit(w, f):
    # Extract data
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)  # (2000, 784)
    y = f["y"].astype(np.int64)                                 # (2000,)
    N = X.shape[0]
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Helper: ridge regression with CV to select alpha
    def fit_ridge_cv(F, Y, alphas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)):
        N, D = F.shape
        F_aug = np.hstack([F, np.ones((N, 1))])  # add bias
        # 5-fold CV
        n_folds = 5
        fold_size = N // n_folds
        best_alpha = None
        best_acc = -1.0
        for alpha in alphas:
            accs = []
            for k in range(n_folds):
                val_start = k * fold_size
                val_end = (k + 1) * fold_size if k < n_folds - 1 else N
                val_idx = np.arange(val_start, val_end)
                train_idx = np.setdiff1d(np.arange(N), val_idx)
                F_tr = F_aug[train_idx]
                Y_tr = Y[train_idx]
                F_va = F_aug[val_idx]
                Y_va = Y[val_idx]
                # Ridge solution: (F^T F + alpha I)^-1 F^T Y
                A = F_tr.T @ F_tr + alpha * np.eye(D + 1)
                b = F_tr.T @ Y_tr
                W = np.linalg.solve(A, b)
                pred = F_va @ W
                acc = np.mean(np.argmax(pred, axis=1) == np.argmax(Y_va, axis=1))
                accs.append(acc)
            mean_acc = np.mean(accs)
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_alpha = alpha
        # Fit on full data with best alpha
        A = F_aug.T @ F_aug + best_alpha * np.eye(D + 1)
        b = F_aug.T @ Y
        W_full = np.linalg.solve(A, b)
        return W_full, best_alpha, best_acc

    # Candidate 1: current features
    F_cur = f["features"].astype(np.float64)
    W_cur, alpha_cur, acc_cur = fit_ridge_cv(F_cur, Y)

    # Candidate 2: PCA features from raw pixels
    X_mean = X.mean(axis=0)
    X_centered = X - X_mean
    # SVD for PCA
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # Take first 128 components
    components = Vt[:128, :]  # (128, 784)
    # New fc1: weight = components, bias = -components @ mean
    fc1_w_pca = components.astype(np.float32)
    fc1_b_pca = (-components @ X_mean).astype(np.float32)
    # Compute new hidden features with ReLU
    H_pca = np.maximum(0, X_centered @ components.T)  # (2000, 128)
    W_pca, alpha_pca, acc_pca = fit_ridge_cv(H_pca, Y)

    # Choose better candidate
    if acc_pca > acc_cur:
        # Use PCA fc1
        w["fc1.weight"] = fc1_w_pca
        w["fc1.bias"] = fc1_b_pca
        W_best = W_pca
        chosen = "pca"
        best_cv_acc = acc_pca
        best_alpha = alpha_pca
    else:
        # Keep current fc1
        W_best = W_cur
        chosen = "current"
        best_cv_acc = acc_cur
        best_alpha = alpha_cur

    # W_best is (D+1, 10) where last row is bias
    fc2_w = W_best[:128, :].T.astype(np.float32)  # (10, 128)
    fc2_b = W_best[128, :].astype(np.float32)     # (10,)

    w["fc2.weight"] = fc2_w
    w["fc2.bias"] = fc2_b

    # Diagnostics
    diagnostics = {
        "chosen": chosen,
        "cv_accuracy_current": float(acc_cur),
        "cv_accuracy_pca": float(acc_pca),
        "best_cv_accuracy": float(best_cv_acc),
        "best_alpha": float(best_alpha),
        "fc2_weight_norm": float(np.linalg.norm(fc2_w)),
        "fc2_bias_norm": float(np.linalg.norm(fc2_b)),
    }
    return {"weights": w, "diagnostics": diagnostics}
