import numpy as np

def edit(w, f):
    X = f["features"].astype(np.float64)
    y = f["y"].astype(np.int64)
    N, D = X.shape
    C = 10
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Split into train/val for lambda selection
    np.random.seed(42)
    indices = np.random.permutation(N)
    train_idx = indices[:1600]
    val_idx = indices[1600:]

    X_train = X[train_idx]
    Y_train = Y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    # Center using train means
    X_train_mean = X_train.mean(axis=0)
    Y_train_mean = Y_train.mean(axis=0)
    X_train_c = X_train - X_train_mean
    Y_train_c = Y_train - Y_train_mean

    # SVD of centered training features
    U, S, Vt = np.linalg.svd(X_train_c, full_matrices=False)
    UTY = U.T @ Y_train_c

    lambdas = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    best_lambda = None
    best_val_acc = -1.0

    for lam in lambdas:
        d = S / (S**2 + lam)
        W = Vt.T @ (d[:, None] * UTY)
        bias = Y_train_mean - X_train_mean @ W
        logits = X_val @ W + bias
        pred = np.argmax(logits, axis=1)
        acc = np.mean(pred == y_val)
        if acc > best_val_acc:
            best_val_acc = acc
            best_lambda = lam

    # Retrain on full pool with best lambda
    X_mean = X.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    Xc = X - X_mean
    Yc = Y - Y_mean
    XtX = Xc.T @ Xc
    XtY = Xc.T @ Yc
    W = np.linalg.solve(XtX + best_lambda * np.eye(D), XtY)
    bias = Y_mean - X_mean @ W

    # Update only the head
    w["head.weight"] = W.T.astype(np.float32)
    w["head.bias"] = bias.astype(np.float32)

    # Diagnostics
    train_logits = X @ W + bias
    train_pred = np.argmax(train_logits, axis=1)
    train_acc = np.mean(train_pred == y)

    batch_ids = f["batch.ids"]
    X_batch = X[batch_ids]
    y_batch = y[batch_ids]
    batch_logits = X_batch @ W + bias
    batch_pred = np.argmax(batch_logits, axis=1)
    batch_acc = np.mean(batch_pred == y_batch)

    diagnostics = {
        "best_lambda": float(best_lambda),
        "val_acc": float(best_val_acc),
        "train_acc": float(train_acc),
        "batch_acc": float(batch_acc),
        "head_weight_std": float(w["head.weight"].std()),
        "head_bias_mean": float(w["head.bias"].mean())
    }

    return {"weights": w, "diagnostics": diagnostics}
