import numpy as np

def edit(w, f):
    # Extract feedback arrays
    H = f["features"].astype(np.float64)          # (2000, 128)
    Z = f["logits"].astype(np.float64)            # (2000, 10)
    y = f["y"].astype(np.int64)                   # (2000,)
    N = H.shape[0]
    H1 = np.concatenate([H, np.ones((N, 1))], axis=1)  # (2000, 129)

    # Precompute true class, max-other class, and margin for each example
    true = y
    mask = np.ones((N, 10), dtype=bool)
    mask[np.arange(N), true] = False
    Z_masked = np.where(mask, Z, -np.inf)
    other = np.argmax(Z_masked, axis=1)
    margin = Z[np.arange(N), true] - Z[np.arange(N), other]

    def compute_targets(margin_target, weight_error):
        delta = np.maximum(0.0, margin_target - margin) / 2.0
        T = np.zeros((N, 10))
        T[np.arange(N), true] = delta
        T[np.arange(N), other] = -delta
        W = np.ones(N)
        W[margin < 0] = weight_error
        return T, W

    def fit_theta(H1_sub, T_sub, W_sub, lam):
        H1W = H1_sub * W_sub[:, None]
        A = H1_sub.T @ H1W + lam * np.eye(H1_sub.shape[1])
        B = H1_sub.T @ (T_sub * W_sub[:, None])
        return np.linalg.solve(A, B)

    # Cross-validation grid search
    K = 5
    indices = np.arange(N)
    np.random.seed(0)
    np.random.shuffle(indices)
    folds = np.array_split(indices, K)

    best_acc = -1.0
    best_params = None

    lam_grid = [0.1, 1.0, 10.0]
    margin_grid = [0.0, 0.5, 1.0, 2.0]
    weight_grid = [1.0, 3.0]
    alpha_grid = [0.5, 1.0, 1.5]

    for lam in lam_grid:
        for mt in margin_grid:
            T, W = compute_targets(mt, 1.0)  # weight_error will be overridden per fold? Actually we need to recompute W per weight_error.
            # We'll loop weight_error inside
            for we in weight_grid:
                T, W = compute_targets(mt, we)
                for alpha in alpha_grid:
                    cv_acc = []
                    for i in range(K):
                        val_idx = folds[i]
                        train_idx = np.concatenate([folds[j] for j in range(K) if j != i])
                        H1_tr = H1[train_idx]
                        T_tr = T[train_idx]
                        W_tr = W[train_idx]
                        Theta = fit_theta(H1_tr, T_tr, W_tr, lam)
                        Z_val = Z[val_idx]
                        H1_val = H1[val_idx]
                        Z_new = Z_val + alpha * (H1_val @ Theta)
                        pred = np.argmax(Z_new, axis=1)
                        acc = np.mean(pred == y[val_idx])
                        cv_acc.append(acc)
                    mean_acc = np.mean(cv_acc)
                    if mean_acc > best_acc:
                        best_acc = mean_acc
                        best_params = (lam, mt, we, alpha)

    # Fallback if something went wrong
    if best_params is None:
        best_params = (1.0, 1.0, 1.0, 1.0)
        best_acc = 0.0

    lam, mt, we, alpha = best_params

    # Compute final correction on all data
    T, W = compute_targets(mt, we)
    Theta = fit_theta(H1, T, W, lam)

    # Theta shape (129, 10). First 128 rows are weight correction, last row is bias correction.
    dW = Theta[:128].T   # (10, 128)
    db = Theta[128]      # (10,)

    # Apply correction
    w["fc2.weight"] = (w["fc2.weight"].astype(np.float64) + alpha * dW).astype(np.float32)
    w["fc2.bias"]   = (w["fc2.bias"].astype(np.float64)   + alpha * db).astype(np.float32)

    # Clip to safe range
    w["fc2.weight"] = np.clip(w["fc2.weight"], -1000.0, 1000.0)
    w["fc2.bias"]   = np.clip(w["fc2.bias"], -1000.0, 1000.0)

    diagnostics = {
        "best_params": {
            "lambda": float(lam),
            "margin_target": float(mt),
            "weight_error": float(we),
            "alpha": float(alpha)
        },
        "cv_accuracy": float(best_acc),
        "correction_norm": float(np.linalg.norm(alpha * dW)),
        "bias_correction_norm": float(np.linalg.norm(alpha * db))
    }

    return {"weights": w, "diagnostics": diagnostics}
