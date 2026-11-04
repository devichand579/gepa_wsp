import numpy as np

def edit(w, f):
    # Extract full feedback pool
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float32)  # (2000, 784)
    y = f["y"].astype(np.int64)                                 # (2000,)
    N = X.shape[0]

    # Current first layer parameters
    W1 = w["fc1.weight"].astype(np.float32)  # (128, 784)
    b1 = w["fc1.bias"].astype(np.float32)    # (128,)

    # Compute pre-activations for the whole pool
    Z = X @ W1.T + b1  # (2000, 128)

    # Re-center each unit's pre-activation to its median (robust centering)
    median_Z = np.median(Z, axis=0)  # (128,)
    b1_new = -median_Z.astype(np.float32)

    # New hidden features
    H = np.maximum(0, X @ W1.T + b1_new)  # (2000, 128)

    # One-hot targets
    Y = np.zeros((N, 10), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    # Augment hidden features with a bias column
    H_aug = np.concatenate([H, np.ones((N, 1), dtype=np.float32)], axis=1)  # (2000, 129)

    # --- Closed-form ridge regression with 5-fold CV to choose alpha ---
    alphas = [1e-4, 1e-3, 1e-2, 0.1, 1.0]
    best_alpha = 1e-3
    best_acc = 0.0

    n_folds = 5
    indices = np.arange(N)
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    fold_size = N // n_folds

    for alpha in alphas:
        accs = []
        for fold in range(n_folds):
            test_idx = indices[fold * fold_size:(fold + 1) * fold_size]
            train_idx = np.setdiff1d(indices, test_idx)

            H_train = H_aug[train_idx]
            Y_train = Y[train_idx]
            H_test = H_aug[test_idx]
            y_test = y[test_idx]

            # Ridge normal equations: (H^T H + alpha I) W = H^T Y
            A = H_train.T @ H_train + alpha * np.eye(H_train.shape[1], dtype=np.float32)
            B = H_train.T @ Y_train
            W = np.linalg.solve(A, B)  # (129, 10)

            logits = H_test @ W
            preds = np.argmax(logits, axis=1)
            acc = np.mean(preds == y_test)
            accs.append(acc)
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_alpha = alpha

    # Final fit on the entire pool with the best alpha
    A = H_aug.T @ H_aug + best_alpha * np.eye(H_aug.shape[1], dtype=np.float32)
    B = H_aug.T @ Y
    W_full = np.linalg.solve(A, B)  # (129, 10)

    fc2_weight = W_full[:128, :].T.astype(np.float32)  # (10, 128)
    fc2_bias = W_full[128, :].astype(np.float32)       # (10,)

    # Update the network parameters
    w["fc1.bias"] = b1_new
    w["fc2.weight"] = fc2_weight
    w["fc2.bias"] = fc2_bias

    # Diagnostics
    diagnostics = {
        "best_alpha": float(best_alpha),
        "cv_accuracy": float(best_acc),
        "b1_new_mean": float(b1_new.mean()),
        "b1_new_std": float(b1_new.std()),
        "fc2_weight_norm": float(np.linalg.norm(fc2_weight)),
        "fc2_bias_norm": float(np.linalg.norm(fc2_bias)),
    }

    return {"weights": w, "diagnostics": diagnostics}
