import numpy as np

def edit(w, f):
    # Extract full feedback pool features and labels
    X = f["features"].astype(np.float64)  # (2000, 128)
    y = f["y"].astype(np.int64)           # (2000,)
    N, D = X.shape
    num_classes = 10

    # Standardize features (clip std to avoid division by zero)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.maximum(std, 1e-3)
    X_std = (X - mean) / std

    # Augment with bias column
    X_aug = np.hstack([X_std, np.ones((N, 1))])  # (N, D+1)

    # One-hot targets
    Y = np.eye(num_classes)[y]  # (N, 10)

    # Cross-validation to select ridge lambda
    lambdas = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_lambda = None
    best_acc = -1.0

    n_folds = 5
    indices = np.arange(N)
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    for lam in lambdas:
        accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])

            X_train = X_aug[train_idx]
            Y_train = Y[train_idx]
            X_val = X_aug[val_idx]
            y_val = y[val_idx]

            XtX = X_train.T @ X_train
            XtY = X_train.T @ Y_train
            reg = lam * np.eye(XtX.shape[0])
            W = np.linalg.solve(XtX + reg, XtY)

            logits = X_val @ W
            preds = np.argmax(logits, axis=1)
            accs.append(np.mean(preds == y_val))

        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lambda = lam

    # Fit on full data with best lambda
    XtX = X_aug.T @ X_aug
    XtY = X_aug.T @ Y
    reg = best_lambda * np.eye(XtX.shape[0])
    W_full = np.linalg.solve(XtX + reg, XtY)  # (D+1, 10)

    # Extract weights and bias for standardized features
    W_std = W_full[:D, :].T  # (10, D)
    b_std = W_full[D, :]     # (10,)

    # Convert back to original feature space:
    # logits = W_std @ ((X - mean)/std).T + b_std
    #        = (W_std / std) @ X.T + (b_std - W_std @ (mean/std))
    fc2_weight = W_std / std[None, :]
    fc2_bias = b_std - W_std @ (mean / std)

    # Rescale logits to match current logit std (preserves accuracy, improves CE)
    current_logits = f["logits"].astype(np.float64)
    current_std = current_logits.std()
    new_logits = X @ fc2_weight.T + fc2_bias
    new_std = new_logits.std()
    if new_std > 1e-6:
        scale = current_std / new_std
        fc2_weight *= scale
        fc2_bias *= scale
    else:
        scale = 1.0

    # Write back to weights
    w["fc2.weight"] = fc2_weight.astype(np.float32)
    w["fc2.bias"] = fc2_bias.astype(np.float32)

    diagnostics = {
        "best_lambda": best_lambda,
        "cv_accuracy": float(best_acc),
        "current_logit_std": float(current_std),
        "new_logit_std_before_scale": float(new_std),
        "scale": float(scale),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
        "fc2_weight_min": float(w["fc2.weight"].min()),
        "fc2_weight_max": float(w["fc2.weight"].max()),
        "fc2_bias_min": float(w["fc2.bias"].min()),
        "fc2_bias_max": float(w["fc2.bias"].max()),
    }

    return {"weights": w, "diagnostics": diagnostics}
