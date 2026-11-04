import numpy as np

def edit(w, f):
    X = f["features"]  # (2000, 1568)
    y = f["y"]         # (2000,)
    N, D = X.shape
    num_classes = 10
    Y = np.zeros((N, num_classes), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    # Use the provided paired reflection batch as a hold-out validation set
    val_idx = f["batch.ids"]  # (128,)
    train_idx = np.setdiff1d(np.arange(N), val_idx)

    X_train = X[train_idx]
    Y_train = Y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    lambdas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    best_acc = -1.0
    best_lambda = None
    best_standardize = False

    for standardize in [False, True]:
        if standardize:
            mu = X_train.mean(axis=0)
            sigma = X_train.std(axis=0)
            sigma[sigma < 1e-6] = 1.0
            X_train_std = (X_train - mu) / sigma
            X_val_std = (X_val - mu) / sigma
        else:
            X_train_std = X_train
            X_val_std = X_val

        X_mean = X_train_std.mean(axis=0)
        Y_mean = Y_train.mean(axis=0)
        X_centered = X_train_std - X_mean
        Y_centered = Y_train - Y_mean

        XTX = X_centered.T @ X_centered
        XTY = X_centered.T @ Y_centered

        for lam in lambdas:
            A = XTX + lam * np.eye(D)
            W = np.linalg.solve(A, XTY)
            b = Y_mean - X_mean @ W
            logits = X_val_std @ W + b
            preds = np.argmax(logits, axis=1)
            acc = np.mean(preds == y_val)
            if acc > best_acc:
                best_acc = acc
                best_lambda = lam
                best_standardize = standardize

    # Refit on all 2000 examples with the best lambda and standardization choice
    if best_standardize:
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma[sigma < 1e-6] = 1.0
        X_std = (X - mu) / sigma
    else:
        X_std = X

    X_mean = X_std.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    X_centered = X_std - X_mean
    Y_centered = Y - Y_mean

    XTX = X_centered.T @ X_centered
    XTY = X_centered.T @ Y_centered
    A = XTX + best_lambda * np.eye(D)
    W = np.linalg.solve(A, XTY)
    b = Y_mean - X_mean @ W

    if best_standardize:
        W_orig = W / sigma[:, None]
        b_orig = b - (mu / sigma) @ W
    else:
        W_orig = W
        b_orig = b

    # Clip to satisfy the absolute value <= 1000 constraint
    W_orig = np.clip(W_orig, -1000.0, 1000.0)
    b_orig = np.clip(b_orig, -1000.0, 1000.0)

    w["head.weight"] = W_orig.T.astype(np.float32)
    w["head.bias"] = b_orig.astype(np.float32)

    # Diagnostics
    train_logits = X @ w["head.weight"].T + w["head.bias"]
    train_preds = np.argmax(train_logits, axis=1)
    train_acc = float(np.mean(train_preds == y))

    diagnostics = {
        "best_lambda": best_lambda,
        "best_standardize": best_standardize,
        "val_accuracy": float(best_acc),
        "train_accuracy": train_acc,
        "head_weight_norm": float(np.linalg.norm(w["head.weight"])),
        "head_bias_norm": float(np.linalg.norm(w["head.bias"])),
    }
    return {"weights": w, "diagnostics": diagnostics}
