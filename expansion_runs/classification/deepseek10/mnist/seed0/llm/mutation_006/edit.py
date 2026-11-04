def edit(w, f):
    import numpy as np

    # Extract current hidden features and labels from the full feedback pool
    H = f["features"].astype(np.float64)   # (2000, 128)
    y = f["y"].astype(np.int64)            # (2000,)
    N, D = H.shape
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Cross-validation to select ridge alpha
    n_folds = 5
    indices = np.arange(N)
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    alphas = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
    best_alpha = None
    best_acc = -1.0

    for alpha in alphas:
        accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            H_tr = H[train_idx]
            Y_tr = Y[train_idx]
            H_va = H[val_idx]
            y_va = y[val_idx]

            # Augment with bias
            H_tr_aug = np.hstack([H_tr, np.ones((H_tr.shape[0], 1))])
            H_va_aug = np.hstack([H_va, np.ones((H_va.shape[0], 1))])

            # Ridge closed-form: (H^T H + alpha I) W = H^T Y
            A = H_tr_aug.T @ H_tr_aug + alpha * np.eye(H_tr_aug.shape[1])
            B = H_tr_aug.T @ Y_tr
            W = np.linalg.solve(A, B)

            logits = H_va_aug @ W
            pred = np.argmax(logits, axis=1)
            accs.append(np.mean(pred == y_va))
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_alpha = alpha

    # Refit on the full pool with the best alpha
    H_aug = np.hstack([H, np.ones((N, 1))])
    A = H_aug.T @ H_aug + best_alpha * np.eye(H_aug.shape[1])
    B = H_aug.T @ Y
    W_full = np.linalg.solve(A, B)   # (129, 10)

    # Update fc2 weights and bias
    w["fc2.weight"] = W_full[:D, :].T.astype(np.float32)   # (10, 128)
    w["fc2.bias"]   = W_full[D, :].astype(np.float32)      # (10,)

    # Diagnostics
    diagnostics = {
        "method": "ridge_fc2",
        "best_alpha": best_alpha,
        "cv_accuracy": float(best_acc),
        "fc2_weight_mean": float(w["fc2.weight"].mean()),
        "fc2_weight_std": float(w["fc2.weight"].std()),
        "fc2_bias_mean": float(w["fc2.bias"].mean()),
        "fc2_bias_std": float(w["fc2.bias"].std()),
    }

    return {"weights": w, "diagnostics": diagnostics}
