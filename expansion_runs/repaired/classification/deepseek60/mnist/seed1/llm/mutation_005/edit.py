import numpy as np

def edit(w, f):
    # Extract current hidden features and labels from the entire feedback pool
    H = f["features"].astype(np.float64)   # [2000, 128]
    y = f["y"].astype(np.int64)            # [2000]
    N, D = H.shape
    C = 10

    # One-hot encode labels
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Augment features with a bias column
    H_aug = np.concatenate([H, np.ones((N, 1))], axis=1)  # [N, D+1]

    # Candidate ridge regularization strengths
    lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]

    # 5-fold cross-validation to select lambda
    n_folds = 5
    indices = np.arange(N)
    np.random.seed(0)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    best_lambda = None
    best_cv_acc = -1.0

    for lam in lambdas:
        accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])

            H_train = H_aug[train_idx]
            Y_train = Y[train_idx]
            H_val = H_aug[val_idx]
            y_val = y[val_idx]

            # Closed-form ridge regression: W = (H^T H + lam I)^-1 H^T Y
            A = H_train.T @ H_train + lam * np.eye(D + 1)
            b = H_train.T @ Y_train
            W = np.linalg.solve(A, b)  # [D+1, C]

            logits_val = H_val @ W
            preds = np.argmax(logits_val, axis=1)
            accs.append(np.mean(preds == y_val))

        mean_acc = np.mean(accs)
        if mean_acc > best_cv_acc:
            best_cv_acc = mean_acc
            best_lambda = lam

    # Refit on all feedback data with the best lambda
    A = H_aug.T @ H_aug + best_lambda * np.eye(D + 1)
    b = H_aug.T @ Y
    W_full = np.linalg.solve(A, b)  # [D+1, C]

    # Extract fc2 weights and bias
    fc2_weight = W_full[:D, :].T  # [C, D]
    fc2_bias = W_full[D, :]       # [C]

    # Update the weights dictionary
    w["fc2.weight"] = fc2_weight.astype(np.float32)
    w["fc2.bias"] = fc2_bias.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_cv_acc),
        "fc2_weight_mean": float(np.mean(fc2_weight)),
        "fc2_weight_std": float(np.std(fc2_weight)),
        "fc2_bias_mean": float(np.mean(fc2_bias)),
        "fc2_bias_std": float(np.std(fc2_bias)),
    }

    return {"weights": w, "diagnostics": diagnostics}
