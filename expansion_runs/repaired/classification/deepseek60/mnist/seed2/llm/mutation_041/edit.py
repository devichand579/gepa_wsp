import numpy as np

def edit(w, f):
    # Extract features and labels
    H = f["features"].astype(np.float64)          # (2000, 128)
    y = f["y"].astype(np.int64)                   # (2000,)
    N, D = H.shape
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Augment features with bias
    H_aug = np.concatenate([H, np.ones((N, 1), dtype=np.float64)], axis=1)  # (2000, 129)

    # Current fc2 parameters as prior
    W0 = w["fc2.weight"].astype(np.float64)       # (10, 128)
    b0 = w["fc2.bias"].astype(np.float64)         # (10,)
    W0_aug = np.concatenate([W0, b0[:, None]], axis=1)  # (10, 129)

    # 5-fold cross-validation to choose lambda
    n_folds = 5
    indices = np.arange(N)
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    lambdas = np.logspace(-4, 4, 25)
    best_lambda = None
    best_acc = -1.0

    for lam in lambdas:
        accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])

            H_tr = H_aug[train_idx]
            Y_tr = Y[train_idx]
            H_va = H_aug[val_idx]
            y_va = y[val_idx]

            # Ridge with prior: W^T = (H^T H + lam I)^-1 (H^T Y + lam W0_aug^T)
            A = H_tr.T @ H_tr + lam * np.eye(H_tr.shape[1])
            B = H_tr.T @ Y_tr + lam * W0_aug.T
            W_new = np.linalg.solve(A, B)  # (129, 10)

            logits_va = H_va @ W_new
            preds = np.argmax(logits_va, axis=1)
            accs.append(np.mean(preds == y_va))
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lambda = lam

    # Fit on all data with best lambda
    lam = best_lambda
    A = H_aug.T @ H_aug + lam * np.eye(H_aug.shape[1])
    B = H_aug.T @ Y + lam * W0_aug.T
    W_new = np.linalg.solve(A, B)  # (129, 10)

    # Extract new fc2 weights and bias
    fc2_weight = W_new[:D, :].T   # (10, 128)
    fc2_bias = W_new[D, :]        # (10,)

    # Scale logits to match current logit standard deviation
    current_logits = f["logits"].astype(np.float64)
    current_std = np.std(current_logits)
    new_logits = H_aug @ W_new
    new_std = np.std(new_logits)
    scale = 1.0
    if new_std > 1e-8:
        scale = current_std / new_std
        fc2_weight *= scale
        fc2_bias *= scale

    # Clip to safe bounds and cast to float32
    fc2_weight = np.clip(fc2_weight, -1000.0, 1000.0).astype(np.float32)
    fc2_bias = np.clip(fc2_bias, -1000.0, 1000.0).astype(np.float32)

    # Update weights
    w["fc2.weight"] = fc2_weight
    w["fc2.bias"] = fc2_bias

    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_acc),
        "scale": float(scale),
        "fc2_weight_norm": float(np.linalg.norm(fc2_weight)),
        "fc2_bias_norm": float(np.linalg.norm(fc2_bias)),
    }

    return {"weights": w, "diagnostics": diagnostics}
