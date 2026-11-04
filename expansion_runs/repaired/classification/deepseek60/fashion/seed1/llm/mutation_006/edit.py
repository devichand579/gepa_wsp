import numpy as np

def edit(w, f):
    X = f["features"].astype(np.float64)  # (2000, 1568)
    y = f["y"].astype(np.int64)
    N, D = X.shape

    # Standardize features to improve conditioning
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1e-6
    X_std = (X - mean) / std

    # Augment with bias column
    X_aug = np.concatenate([X_std, np.ones((N, 1))], axis=1)  # (N, D+1)

    # One-hot targets
    Y = np.zeros((N, 10))
    Y[np.arange(N), y] = 1.0

    # Convert current head to standardized space (as prior)
    W_prior_orig = w["head.weight"].astype(np.float64)  # (10, D)
    b_prior_orig = w["head.bias"].astype(np.float64)    # (10,)
    W_prior_std = W_prior_orig * std                    # (10, D)
    b_prior_std = mean @ W_prior_orig.T + b_prior_orig  # (10,)
    W_prior_aug = np.concatenate([W_prior_std, b_prior_std[:, None]], axis=1)  # (10, D+1)
    W_prior_aug_T = W_prior_aug.T                       # (D+1, 10)

    # Precompute full Gram matrix and X^T Y
    G_full = X_aug.T @ X_aug      # (D+1, D+1)
    XTY_full = X_aug.T @ Y        # (D+1, 10)

    # 5-fold cross-validation to choose lambda and whether to use prior
    n_folds = 5
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    lambdas = np.logspace(-2, 5, 15)  # 0.01 to 100000

    best_acc = -1.0
    best_lam = None
    best_use_prior = False

    for use_prior in [False, True]:
        for lam in lambdas:
            accs = []
            for i in range(n_folds):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
                X_val = X_aug[val_idx]
                Y_val = Y[val_idx]
                y_val = y[val_idx]

                G_val = X_val.T @ X_val
                XTY_val = X_val.T @ Y_val

                G_train = G_full - G_val
                XTY_train = XTY_full - XTY_val

                if use_prior:
                    XTY_train = XTY_train + lam * W_prior_aug_T

                G_reg = G_train + lam * np.eye(D + 1)
                try:
                    W_aug = np.linalg.solve(G_reg, XTY_train)  # (D+1, 10)
                except np.linalg.LinAlgError:
                    G_reg += 1e-6 * np.eye(D + 1)
                    W_aug = np.linalg.solve(G_reg, XTY_train)

                logits_val = X_val @ W_aug
                preds = np.argmax(logits_val, axis=1)
                accs.append(np.mean(preds == y_val))
            mean_acc = np.mean(accs)
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_lam = lam
                best_use_prior = use_prior

    # Fit on all data with the best hyperparameters
    G_reg = G_full + best_lam * np.eye(D + 1)
    XTY = XTY_full.copy()
    if best_use_prior:
        XTY = XTY + best_lam * W_prior_aug_T
    W_aug = np.linalg.solve(G_reg, XTY)  # (D+1, 10)

    # Reshape to (10, D+1)
    W_aug = W_aug.T
    W = W_aug[:, :D]   # (10, D)
    b = W_aug[:, D]    # (10,)

    # Convert back to original feature space
    W_orig = W / std
    b_orig = b - W @ (mean / std)

    # Safety clip
    W_orig = np.clip(W_orig, -1000, 1000)
    b_orig = np.clip(b_orig, -1000, 1000)

    # Update weights
    w["head.weight"] = W_orig.astype(np.float32)
    w["head.bias"] = b_orig.astype(np.float32)

    # Diagnostics
    diag = {
        "best_lambda": float(best_lam),
        "best_use_prior": bool(best_use_prior),
        "cv_acc": float(best_acc),
        "W_mean": float(np.mean(W_orig)),
        "W_std": float(np.std(W_orig)),
        "b_mean": float(np.mean(b_orig)),
        "b_std": float(np.std(b_orig)),
    }
    return {"weights": w, "diagnostics": diag}
