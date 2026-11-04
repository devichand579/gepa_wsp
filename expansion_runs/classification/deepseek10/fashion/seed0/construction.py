import numpy as np

def edit(w, f):
    # Extract full training features and labels
    X = f["features"].astype(np.float64)          # (2000, 1568)
    y = f["y"].astype(np.int64)                   # (2000,)
    N, D = X.shape
    num_classes = 10

    # One-hot encode labels
    Y = np.zeros((N, num_classes), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Identify features with non-zero variance
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    keep = std > 1e-8
    D_keep = int(keep.sum())

    X_keep = X[:, keep]
    mean_keep = mean[keep]
    std_keep = std[keep]

    # Standardize kept features
    X_std = (X_keep - mean_keep) / std_keep

    # Augment with bias column
    Xa = np.concatenate([X_std, np.ones((N, 1))], axis=1)  # (N, D_keep+1)

    # Ridge regression (closed-form)
    lambda_reg = 1.0
    reg = np.eye(D_keep + 1) * lambda_reg
    reg[D_keep, D_keep] = 0.0  # do not regularize bias

    XtX = Xa.T @ Xa
    XtY = Xa.T @ Y

    W_aug = np.linalg.solve(XtX + reg, XtY)  # (D_keep+1, num_classes)

    W_std = W_aug[:D_keep, :].T   # (num_classes, D_keep)
    b_std = W_aug[D_keep, :]      # (num_classes,)

    # Convert back to original feature space
    W_orig_keep = W_std / std_keep                     # (num_classes, D_keep)
    b_orig = b_std - mean_keep @ W_orig_keep.T         # (num_classes,)

    # Build full weight matrix (zeros for dropped features)
    W_full = np.zeros((num_classes, D), dtype=np.float64)
    W_full[:, keep] = W_orig_keep

    # Clip to safe range
    W_full = np.clip(W_full, -1000.0, 1000.0)
    b_orig = np.clip(b_orig, -1000.0, 1000.0)

    # Update only the head
    w["head.weight"] = W_full.astype(np.float32)
    w["head.bias"] = b_orig.astype(np.float32)

    # Diagnostics: training accuracy
    logits = X @ W_full.T + b_orig
    preds = np.argmax(logits, axis=1)
    acc = float(np.mean(preds == y))

    diagnostics = {
        "train_accuracy": acc,
        "num_features_kept": D_keep,
        "lambda_reg": lambda_reg,
    }

    return {"weights": w, "diagnostics": diagnostics}
