import numpy as np

def edit(w, f):
    # Extract full feedback features and labels
    H = f["features"].astype(np.float64)  # (2000, 128)
    y = f["y"].astype(np.int64)           # (2000,)
    N, D = H.shape
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Standardize features (zero mean, unit variance)
    mean = np.mean(H, axis=0)
    std = np.std(H, axis=0) + 1e-6
    H_std = (H - mean) / std

    # Add bias column
    H_aug = np.hstack([H_std, np.ones((N, 1))])
    D_aug = D + 1

    # Ridge regularization matrix (do not penalize bias)
    I_reg = np.eye(D_aug)
    I_reg[-1, -1] = 0.0

    # 5-fold cross-validation to pick alpha
    alphas = np.logspace(-4, 3, 20)
    best_alpha = None
    best_acc = -1.0

    n_folds = 5
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    for alpha in alphas:
        accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])

            H_train = H_aug[train_idx]
            Y_train = Y[train_idx]
            H_val = H_aug[val_idx]

            A = H_train.T @ H_train + alpha * I_reg
            B = H_train.T @ Y_train
            W = np.linalg.solve(A, B)

            logits_val = H_val @ W
            preds = np.argmax(logits_val, axis=1)
            acc = np.mean(preds == y[val_idx])
            accs.append(acc)
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_alpha = alpha

    # Final fit on all data with best alpha
    A = H_aug.T @ H_aug + best_alpha * I_reg
    B = H_aug.T @ Y
    W_full = np.linalg.solve(A, B)  # (D_aug, C)

    W_std = W_full[:-1, :]  # (D, C)
    b_std = W_full[-1, :]   # (C,)

    # Convert back to original feature space
    # logits = H_std @ W_std + b_std = ((H - mean)/std) @ W_std + b_std
    #        = H @ (W_std / std) + (b_std - mean/std @ W_std)
    fc2_weight = (W_std / std[:, None]).T  # (C, D)
    fc2_bias = b_std - (mean / std) @ W_std  # (C,)

    # Update weights
    w["fc2.weight"] = fc2_weight.astype(np.float32)
    w["fc2.bias"] = fc2_bias.astype(np.float32)

    # Diagnostics
    # Training accuracy on full pool
    logits_full = H @ fc2_weight.T + fc2_bias
    preds_full = np.argmax(logits_full, axis=1)
    train_acc = np.mean(preds_full == y)

    # Paired batch accuracy using cached fc1 output
    H_paired = f["layer.fc1.output"].astype(np.float64)  # (128, 128)
    y_paired = f["batch.y"].astype(np.int64)
    logits_paired = H_paired @ fc2_weight.T + fc2_bias
    preds_paired = np.argmax(logits_paired, axis=1)
    paired_acc = np.mean(preds_paired == y_paired)

    diagnostics = {
        "best_alpha": float(best_alpha),
        "cv_accuracy": float(best_acc),
        "train_accuracy": float(train_acc),
        "paired_batch_accuracy": float(paired_acc),
        "fc2_weight_norm": float(np.linalg.norm(fc2_weight)),
        "fc2_bias_norm": float(np.linalg.norm(fc2_bias)),
        "fc2_weight_min": float(np.min(fc2_weight)),
        "fc2_weight_max": float(np.max(fc2_weight)),
        "fc2_bias_min": float(np.min(fc2_bias)),
        "fc2_bias_max": float(np.max(fc2_bias)),
    }

    return {"weights": w, "diagnostics": diagnostics}
