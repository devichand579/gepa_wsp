import numpy as np

def edit(w, f):
    # Extract data
    X = f["x"].reshape(2000, 784).astype(np.float32)
    y = f["y"].astype(np.int64)
    Y = np.eye(10, dtype=np.float32)[y]  # (2000, 10)

    W1 = w["fc1.weight"].astype(np.float32)  # (128, 784)
    b1 = w["fc1.bias"].astype(np.float32)    # (128,)

    # Pre-activations for the full pool
    Pre = X @ W1.T + b1  # (2000, 128)

    # Candidate shifts for ReLU threshold
    shifts = [-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0]
    # Candidate ridge lambdas (regularization strength)
    lambdas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    # 5-fold cross-validation indices
    n = 2000
    indices = np.arange(n)
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    folds = np.array_split(indices, 5)

    best_cv_acc = -1.0
    best_shift = 0.0
    best_lambda = 1.0

    for shift in shifts:
        H = np.maximum(0.0, Pre + shift)  # (2000, 128)
        H_aug = np.concatenate([H, np.ones((n, 1), dtype=np.float32)], axis=1)  # (2000, 129)

        for lam in lambdas:
            # Regularization matrix: do not penalize bias (last element)
            reg = np.eye(129, dtype=np.float32) * lam
            reg[-1, -1] = 0.0

            cv_acc = 0.0
            for val_idx in folds:
                train_idx = np.setdiff1d(indices, val_idx)
                H_train = H_aug[train_idx]
                Y_train = Y[train_idx]
                H_val = H_aug[val_idx]
                y_val = y[val_idx]

                # Solve ridge: W = (H^T H + reg)^-1 H^T Y
                A = H_train.T @ H_train + reg
                B = H_train.T @ Y_train
                W_aug = np.linalg.solve(A, B)  # (129, 10)

                # Predict
                logits = H_val @ W_aug  # (n_val, 10)
                preds = np.argmax(logits, axis=1)
                cv_acc += np.mean(preds == y_val)

            cv_acc /= 5.0
            if cv_acc > best_cv_acc:
                best_cv_acc = cv_acc
                best_shift = shift
                best_lambda = lam

    # Refit on full pool with best shift and lambda
    H_best = np.maximum(0.0, Pre + best_shift)
    H_aug_best = np.concatenate([H_best, np.ones((n, 1), dtype=np.float32)], axis=1)
    reg_best = np.eye(129, dtype=np.float32) * best_lambda
    reg_best[-1, -1] = 0.0
    A = H_aug_best.T @ H_aug_best + reg_best
    B = H_aug_best.T @ Y
    W_aug_best = np.linalg.solve(A, B)  # (129, 10)

    # Update weights
    w["fc1.bias"] = (b1 + best_shift).astype(np.float32)
    w["fc2.weight"] = W_aug_best[:128, :].T.astype(np.float32)  # (10, 128)
    w["fc2.bias"] = W_aug_best[128, :].astype(np.float32)       # (10,)

    # Diagnostics
    train_logits = H_aug_best @ W_aug_best
    train_preds = np.argmax(train_logits, axis=1)
    train_acc = np.mean(train_preds == y)

    diagnostics = {
        "best_shift": float(best_shift),
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_cv_acc),
        "train_accuracy": float(train_acc),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
    }

    return {"weights": w, "diagnostics": diagnostics}
