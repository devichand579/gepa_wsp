import numpy as np

def edit(w, f):
    # Extract full feedback pool
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)
    y = f["y"].astype(np.int64)
    N = X.shape[0]
    Y = np.eye(10)[y]  # one-hot targets

    W1_orig = w["fc1.weight"].astype(np.float64)
    b1_orig = w["fc1.bias"].astype(np.float64)

    # Hyperparameter grid
    scales = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
    bias_modes = ['orig', 'zero', 'add2', 'add5', 'add10']
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    # 5-fold cross-validation splits
    n_folds = 5
    indices = np.arange(N)
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    folds = []
    fold_size = N // n_folds
    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else N
        val_idx = indices[start:end]
        train_idx = np.setdiff1d(indices, val_idx)
        folds.append((train_idx, val_idx))

    best_acc = -1.0
    best_params = None

    for s in scales:
        W1 = s * W1_orig
        for mode in bias_modes:
            if mode == 'orig':
                b1 = b1_orig
            elif mode == 'zero':
                b1 = np.zeros_like(b1_orig)
            elif mode == 'add2':
                b1 = b1_orig + 2.0
            elif mode == 'add5':
                b1 = b1_orig + 5.0
            elif mode == 'add10':
                b1 = b1_orig + 10.0
            else:
                continue

            # Compute hidden features for this fc1 configuration
            H = np.maximum(0.0, X @ W1.T + b1)  # (N, 128)
            # Standardize features (closed-form, no leakage across folds in final fit)
            H_mean = H.mean(axis=0)
            H_std = H.std(axis=0) + 1e-6
            H_norm = (H - H_mean) / H_std
            H_aug = np.concatenate([H_norm, np.ones((N, 1))], axis=1)  # (N, 129)

            for alpha in alphas:
                accs = []
                for train_idx, val_idx in folds:
                    H_train = H_aug[train_idx]
                    Y_train = Y[train_idx]
                    H_val = H_aug[val_idx]

                    A = H_train.T @ H_train
                    reg = np.eye(A.shape[0]) * alpha
                    reg[-1, -1] = 0.0  # do not regularize bias
                    W = np.linalg.solve(A + reg, H_train.T @ Y_train)

                    logits = H_val @ W
                    preds = np.argmax(logits, axis=1)
                    acc = np.mean(preds == y[val_idx])
                    accs.append(acc)
                mean_acc = float(np.mean(accs))
                if mean_acc > best_acc:
                    best_acc = mean_acc
                    best_params = (s, mode, alpha)

    # Refit on all 2000 examples using the best hyperparameters
    s, mode, alpha = best_params
    W1 = s * W1_orig
    if mode == 'orig':
        b1 = b1_orig
    elif mode == 'zero':
        b1 = np.zeros_like(b1_orig)
    elif mode == 'add2':
        b1 = b1_orig + 2.0
    elif mode == 'add5':
        b1 = b1_orig + 5.0
    elif mode == 'add10':
        b1 = b1_orig + 10.0

    H = np.maximum(0.0, X @ W1.T + b1)
    H_mean = H.mean(axis=0)
    H_std = H.std(axis=0) + 1e-6
    H_norm = (H - H_mean) / H_std
    H_aug = np.concatenate([H_norm, np.ones((N, 1))], axis=1)

    A = H_aug.T @ H_aug
    reg = np.eye(A.shape[0]) * alpha
    reg[-1, -1] = 0.0
    W_norm = np.linalg.solve(A + reg, H_aug.T @ Y)

    # Convert standardized-feature weights back to raw-feature weights
    W_raw = W_norm[:128] / H_std[:, None]          # (128, 10)
    b_raw = W_norm[128] - H_mean @ W_raw           # (10,)

    # Write back to the parameter dictionary
    w["fc1.weight"] = W1.astype(np.float32)
    w["fc1.bias"] = b1.astype(np.float32)
    w["fc2.weight"] = W_raw.T.astype(np.float32)   # (10, 128)
    w["fc2.bias"] = b_raw.astype(np.float32)       # (10,)

    # Safety clip to keep values finite and within bounds
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)

    diagnostics = {
        "best_scale": float(s),
        "best_bias_mode": mode,
        "best_alpha": float(alpha),
        "cv_accuracy": float(best_acc),
        "fc1_weight_norm": float(np.linalg.norm(w["fc1.weight"])),
        "fc1_bias_norm": float(np.linalg.norm(w["fc1.bias"])),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
    }

    return {"weights": w, "diagnostics": diagnostics}
