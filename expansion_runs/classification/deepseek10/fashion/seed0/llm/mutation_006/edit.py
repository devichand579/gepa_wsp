import numpy as np

def edit(w, f):
    # Extract features and labels
    X = f["features"].astype(np.float64)          # [2000, 1568]
    y = f["y"].astype(np.int64)                   # [2000]
    N, D = X.shape
    C = 10

    # One-hot labels
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Augment with bias column
    X_aug = np.concatenate([X, np.ones((N, 1), dtype=np.float64)], axis=1)  # [2000, 1569]
    D_aug = D + 1

    # Full normal-equation pieces
    A = X_aug.T @ X_aug          # [1569, 1569]
    B = X_aug.T @ Y              # [1569, 10]

    # Regularization matrix: do not regularize bias
    reg_base = np.eye(D_aug, dtype=np.float64)
    reg_base[-1, -1] = 0.0

    # 5-fold cross-validation to choose lambda
    n_folds = 5
    indices = np.arange(N)
    rng = np.random.RandomState(0)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    lambdas = np.logspace(-4, 4, 17)
    best_lambda = 1.0
    best_acc = -1.0

    for lam in lambdas:
        accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])

            X_val = X_aug[val_idx]
            Y_val = Y[val_idx]
            X_train = X_aug[train_idx]
            Y_train = Y[train_idx]

            A_val = X_val.T @ X_val
            B_val = X_val.T @ Y_val
            A_train = A - A_val
            B_train = B - B_val

            W = np.linalg.solve(A_train + lam * reg_base, B_train)
            logits_val = X_val @ W
            pred = np.argmax(logits_val, axis=1)
            acc = np.mean(pred == y[val_idx])
            accs.append(acc)

        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lambda = lam

    # Refit on all data with best lambda
    W_full = np.linalg.solve(A + best_lambda * reg_base, B)

    # Extract head weights and bias
    head_weight = W_full[:-1, :].T.astype(np.float32)   # [10, 1568]
    head_bias = W_full[-1, :].astype(np.float32)        # [10]

    # Clip to allowed range
    head_weight = np.clip(head_weight, -1000.0, 1000.0)
    head_bias = np.clip(head_bias, -1000.0, 1000.0)

    # Update weights
    w["head.weight"] = head_weight
    w["head.bias"] = head_bias

    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_acc),
        "head_weight_max_abs": float(np.max(np.abs(head_weight))),
        "head_bias_max_abs": float(np.max(np.abs(head_bias))),
        "head_weight_mean_abs": float(np.mean(np.abs(head_weight))),
    }

    return {"weights": w, "diagnostics": diagnostics}
