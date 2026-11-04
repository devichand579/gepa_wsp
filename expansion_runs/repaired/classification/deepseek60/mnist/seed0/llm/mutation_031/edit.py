import numpy as np

def edit(w, f):
    # Extract full feedback pool
    H = f["features"].astype(np.float64)          # (2000, 128)
    y = f["y"].astype(np.int64)                   # (2000,)
    N, D = H.shape
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Current fc2 weights and bias
    W0 = w["fc2.weight"].astype(np.float64)       # (10, 128)
    b0 = w["fc2.bias"].astype(np.float64)         # (10,)

    # Augment features with a bias column
    H_aug = np.hstack([H, np.ones((N, 1), dtype=np.float64)])  # (N, 129)
    W0_aug = np.hstack([W0, b0[:, None]])                       # (10, 129)
    W0_aug_T = W0_aug.T                                         # (129, 10)

    # Precompute normal-equation pieces
    HtH = H_aug.T @ H_aug          # (129, 129)
    HtY = H_aug.T @ Y              # (129, 10)

    # Lambda grid for ridge regularization
    lambdas = np.logspace(-6, 3, 19)   # 1e-6 ... 1e3

    # 5-fold cross-validation to choose lambda
    n_folds = 5
    fold_size = N // n_folds
    indices = np.arange(N)
    rng = np.random.RandomState(42)
    rng.shuffle(indices)

    best_lambda = None
    best_cv_acc = -1.0

    for lam in lambdas:
        cv_accs = []
        for fold in range(n_folds):
            val_idx = indices[fold * fold_size:(fold + 1) * fold_size]
            train_idx = np.setdiff1d(indices, val_idx)

            H_train = H_aug[train_idx]
            Y_train = Y[train_idx]
            H_val = H_aug[val_idx]
            y_val = y[val_idx]

            HtH_train = H_train.T @ H_train
            HtY_train = H_train.T @ Y_train

            A = HtH_train + lam * np.eye(H_aug.shape[1])
            B = HtY_train + lam * W0_aug_T
            W_aug_T = np.linalg.solve(A, B)          # (129, 10)

            logits_val = H_val @ W_aug_T
            preds = np.argmax(logits_val, axis=1)
            acc = np.mean(preds == y_val)
            cv_accs.append(acc)

        mean_acc = np.mean(cv_accs)
        if mean_acc > best_cv_acc:
            best_cv_acc = mean_acc
            best_lambda = lam

    # Final closed-form ridge solution on the full pool
    lam = best_lambda
    A = HtH + lam * np.eye(H_aug.shape[1])
    B = HtY + lam * W0_aug_T
    W_aug_T = np.linalg.solve(A, B)                  # (129, 10)
    W_aug = W_aug_T.T                                # (10, 129)

    # Update fc2 weights and bias
    w["fc2.weight"] = W_aug[:, :D].astype(np.float32)
    w["fc2.bias"] = W_aug[:, D].astype(np.float32)

    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lambda),
        "best_cv_acc": float(best_cv_acc),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
        "fc2_weight_mean": float(np.mean(w["fc2.weight"])),
        "fc2_weight_std": float(np.std(w["fc2.weight"])),
        "fc2_bias_mean": float(np.mean(w["fc2.bias"])),
        "fc2_bias_std": float(np.std(w["fc2.bias"])),
    }

    return {"weights": w, "diagnostics": diagnostics}
