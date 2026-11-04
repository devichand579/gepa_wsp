import numpy as np

def edit(w, f):
    # Extract features and labels from the full feedback pool
    X = f["features"].astype(np.float64)  # (2000, 1568)
    y = f["y"].astype(np.int64)           # (2000,)
    n_samples, n_features = X.shape
    n_classes = 10

    # Precompute per-class sums and scatter matrices for the full dataset
    class_indices = [np.where(y == c)[0] for c in range(n_classes)]
    n_c = np.array([len(idx) for idx in class_indices], dtype=np.float64)
    sum_c = np.zeros((n_classes, n_features), dtype=np.float64)
    sum2_c = np.zeros((n_classes, n_features, n_features), dtype=np.float64)
    for c in range(n_classes):
        idx = class_indices[c]
        Xc = X[idx]
        sum_c[c] = Xc.sum(axis=0)
        sum2_c[c] = Xc.T @ Xc

    # Full within-class scatter matrix
    Sw_full = np.zeros((n_features, n_features), dtype=np.float64)
    for c in range(n_classes):
        mu_c = sum_c[c] / n_c[c]
        Sw_full += sum2_c[c] - n_c[c] * np.outer(mu_c, mu_c)

    # 5-fold cross-validation to select shrinkage lambda
    np.random.seed(42)
    indices = np.random.permutation(n_samples)
    folds = np.array_split(indices, 5)
    lambdas = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]
    best_lambda = None
    best_cv_acc = -1.0

    for lam in lambdas:
        cv_acc = 0.0
        for val_idx in folds:
            # Compute validation set per-class sums
            val_mask = np.zeros(n_samples, dtype=bool)
            val_mask[val_idx] = True
            n_c_val = np.zeros(n_classes, dtype=np.float64)
            sum_c_val = np.zeros((n_classes, n_features), dtype=np.float64)
            sum2_c_val = np.zeros((n_classes, n_features, n_features), dtype=np.float64)
            for c in range(n_classes):
                idx_c = class_indices[c]
                idx_val = idx_c[val_mask[idx_c]]
                if len(idx_val) > 0:
                    Xc_val = X[idx_val]
                    n_c_val[c] = len(idx_val)
                    sum_c_val[c] = Xc_val.sum(axis=0)
                    sum2_c_val[c] = Xc_val.T @ Xc_val

            # Training set sums
            n_c_train = n_c - n_c_val
            sum_c_train = sum_c - sum_c_val
            sum2_c_train = sum2_c - sum2_c_val

            # Compute training means and within-class scatter
            mu_c_train = np.zeros((n_classes, n_features), dtype=np.float64)
            Sw_train = np.zeros((n_features, n_features), dtype=np.float64)
            for c in range(n_classes):
                if n_c_train[c] > 0:
                    mu_c_train[c] = sum_c_train[c] / n_c_train[c]
                    Sw_train += sum2_c_train[c] - n_c_train[c] * np.outer(mu_c_train[c], mu_c_train[c])

            # Regularize
            Sw_reg = Sw_train + lam * np.eye(n_features, dtype=np.float64)

            # Solve for LDA weights: Sw_reg @ W = mu_c_train.T
            try:
                W = np.linalg.solve(Sw_reg, mu_c_train.T)  # (1568, 10)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if singular
                W = np.linalg.pinv(Sw_reg) @ mu_c_train.T

            # Bias: -0.5 * mu_c^T Sw^{-1} mu_c
            bias = -0.5 * np.sum(mu_c_train * W.T, axis=1)  # (10,)

            # Evaluate on validation fold
            X_val = X[val_idx]
            y_val = y[val_idx]
            logits = X_val @ W + bias
            preds = np.argmax(logits, axis=1)
            acc = np.mean(preds == y_val)
            cv_acc += acc

        cv_acc /= len(folds)
        if cv_acc > best_cv_acc:
            best_cv_acc = cv_acc
            best_lambda = lam

    # If no lambda improved over default, use a small lambda
    if best_lambda is None:
        best_lambda = 1e-3

    # Compute final LDA on full data with best_lambda
    Sw_reg = Sw_full + best_lambda * np.eye(n_features, dtype=np.float64)
    mu_full = sum_c / n_c[:, None]  # (10, 1568)
    try:
        W_final = np.linalg.solve(Sw_reg, mu_full.T)  # (1568, 10)
    except np.linalg.LinAlgError:
        W_final = np.linalg.pinv(Sw_reg) @ mu_full.T
    bias_final = -0.5 * np.sum(mu_full * W_final.T, axis=1)  # (10,)

    # Update head weights and bias
    w["head.weight"] = W_final.T.astype(np.float32)  # (10, 1568)
    w["head.bias"] = bias_final.astype(np.float32)   # (10,)

    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_cv_acc),
        "head_weight_max_abs": float(np.max(np.abs(w["head.weight"]))),
        "head_bias_max_abs": float(np.max(np.abs(w["head.bias"]))),
    }
    return {"weights": w, "diagnostics": diagnostics}
