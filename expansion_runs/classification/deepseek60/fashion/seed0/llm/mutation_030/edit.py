import numpy as np
import math

def edit(w, f):
    # Extract features and labels
    X = f["features"].astype(np.float64)  # (2000, 1568)
    y = f["y"].astype(np.int64)           # (2000,)
    n, d = X.shape
    n_classes = 10

    # Standardize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-8] = 1.0
    Xs = (X - X_mean) / X_std

    # One-hot labels
    Y = np.zeros((n, n_classes), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    # Center X and Y
    Xc = Xs - Xs.mean(axis=0)
    Yc = Y - Y.mean(axis=0)

    # SVD of Xc for efficient ridge solutions
    # Xc = U @ diag(S) @ Vt
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # U: (n, k), S: (k,), Vt: (k, d) where k = min(n, d)
    k = len(S)
    # Precompute U^T Yc
    UTY = U.T @ Yc  # (k, n_classes)

    # Candidate lambdas
    lambdas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    # 5-fold cross-validation
    n_folds = 5
    fold_size = n // n_folds
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)

    best_lambda = None
    best_acc = -1.0

    for lam in lambdas:
        accs = []
        for fold in range(n_folds):
            val_idx = indices[fold*fold_size:(fold+1)*fold_size]
            train_idx = np.setdiff1d(indices, val_idx)
            X_tr = Xs[train_idx]
            Y_tr = Y[train_idx]
            X_val = Xs[val_idx]
            Y_val = Y[val_idx]

            # Center training data
            X_tr_mean = X_tr.mean(axis=0)
            Y_tr_mean = Y_tr.mean(axis=0)
            X_tr_c = X_tr - X_tr_mean
            Y_tr_c = Y_tr - Y_tr_mean

            # Solve ridge: W = (X^T X + lam I)^-1 X^T Y
            # Use SVD for stability
            U_tr, S_tr, Vt_tr = np.linalg.svd(X_tr_c, full_matrices=False)
            S_tr_inv = S_tr / (S_tr**2 + lam)
            W = Vt_tr.T @ (S_tr_inv[:, None] * (U_tr.T @ Y_tr_c))
            b = Y_tr_mean - X_tr_mean @ W

            # Predict on validation
            logits = X_val @ W + b
            pred = np.argmax(logits, axis=1)
            acc = np.mean(pred == Y_val.argmax(axis=1))
            accs.append(acc)
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lambda = lam

    # Train on full data with best lambda
    lam = best_lambda if best_lambda is not None else 1.0
    S_inv = S / (S**2 + lam)
    W_full = Vt.T @ (S_inv[:, None] * UTY)
    b_full = Y.mean(axis=0) - Xs.mean(axis=0) @ W_full

    # Set head weights and bias
    w["head.weight"] = W_full.astype(np.float32)
    w["head.bias"] = b_full.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "best_lambda": float(lam),
        "cv_accuracy": float(best_acc),
        "head_weight_std": float(np.std(w["head.weight"])),
        "head_weight_max_abs": float(np.max(np.abs(w["head.weight"]))),
        "head_bias_mean": float(np.mean(w["head.bias"])),
        "head_bias_std": float(np.std(w["head.bias"])),
    }

    return {"weights": w, "diagnostics": diagnostics}
