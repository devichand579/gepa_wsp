import numpy as np

def edit(w, f):
    # Extract features and labels from the full feedback pool
    X = f["features"].astype(np.float64)  # (2000, 128)
    y = f["y"].astype(int)                # (2000,)
    N = X.shape[0]
    num_classes = 10

    # One-hot targets
    Y = np.zeros((N, num_classes), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Current fc2 parameters (prior)
    W0 = w["fc2.weight"].astype(np.float64)  # (10, 128)
    b0 = w["fc2.bias"].astype(np.float64)    # (10,)

    # Augment features with a bias column
    X_aug = np.hstack([X, np.ones((N, 1))])  # (2000, 129)
    W0_aug = np.hstack([W0, b0[:, None]])    # (10, 129)

    # Candidate ridge strengths
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    best_lambda = None
    best_acc = -1.0

    # 5-fold cross-validation to pick lambda
    rng = np.random.default_rng(42)
    indices = rng.permutation(N)
    folds = np.array_split(indices, 5)

    for lam in lambdas:
        accs = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])

            X_train = X_aug[train_idx]
            Y_train = Y[train_idx]
            X_val = X_aug[val_idx]
            y_val = y[val_idx]

            A = X_train.T @ X_train + lam * np.eye(X_train.shape[1])
            B = X_train.T @ Y_train + lam * W0_aug.T
            W_aug = np.linalg.solve(A, B).T  # (10, 129)

            logits_val = X_val @ W_aug.T
            pred = np.argmax(logits_val, axis=1)
            acc = np.mean(pred == y_val)
            accs.append(acc)

        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lambda = lam

    # Fit on all data with the best lambda
    lam = best_lambda
    A = X_aug.T @ X_aug + lam * np.eye(X_aug.shape[1])
    B = X_aug.T @ Y + lam * W0_aug.T
    W_aug = np.linalg.solve(A, B).T  # (10, 129)

    W_new = W_aug[:, :128]
    b_new = W_aug[:, 128]

    # Scale logits to match the current logit standard deviation
    logits_new = X_aug @ W_aug.T
    std_new = np.std(logits_new)
    std_cur = np.std(f["logits"])
    if std_new > 1e-8:
        scale = std_cur / std_new
        W_new *= scale
        b_new *= scale

    # Write back into the weights dictionary
    w["fc2.weight"] = W_new.astype(np.float32)
    w["fc2.bias"] = b_new.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_acc),
        "scale": float(scale) if std_new > 1e-8 else 1.0,
        "fc2_weight_norm_before": float(np.linalg.norm(W0)),
        "fc2_weight_norm_after": float(np.linalg.norm(W_new)),
        "fc2_bias_norm_after": float(np.linalg.norm(b_new)),
    }

    return {"weights": w, "diagnostics": diagnostics}
