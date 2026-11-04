import numpy as np

def edit(w, f):
    # Frozen final features and labels from the full feedback pool
    X = f["features"].astype(np.float64)          # (2000, 1568)
    y = f["y"].astype(np.int64)                   # (2000,)
    N, D = X.shape
    Y = np.zeros((N, 10), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Split for ridge-lambda selection (closed-form, no gradient steps)
    rng = np.random.RandomState(0)
    perm = rng.permutation(N)
    n_val = 400
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    Xb_train = np.concatenate([X_train, np.ones((len(train_idx), 1))], axis=1)
    Xb_val   = np.concatenate([X_val,   np.ones((len(val_idx),   1))], axis=1)

    best_lam, best_acc = 1.0, -1.0
    for lam in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
        A = Xb_train.T @ Xb_train + lam * np.eye(D + 1)
        B = Xb_train.T @ Y_train
        Wb = np.linalg.solve(A, B)
        preds = np.argmax(Xb_val @ Wb, axis=1)
        acc = np.mean(preds == y_val)
        if acc > best_acc:
            best_acc, best_lam = acc, lam

    # Final closed-form ridge fit on all 2000 samples
    Xb = np.concatenate([X, np.ones((N, 1))], axis=1)
    A = Xb.T @ Xb + best_lam * np.eye(D + 1)
    B = Xb.T @ Y
    Wb = np.linalg.solve(A, B)

    W = Wb[:D, :].T          # (10, 1568)
    b = Wb[D, :]             # (10,)

    w["head.weight"] = W.astype(np.float32)
    w["head.bias"]   = b.astype(np.float32)

    # Diagnostics
    logits_all = Xb @ Wb
    acc_all = float(np.mean(np.argmax(logits_all, axis=1) == y))
    diag = {
        "best_lambda": float(best_lam),
        "val_acc": float(best_acc),
        "train_acc_all": acc_all,
        "head_weight_std": float(np.std(w["head.weight"])),
        "head_bias_mean": float(np.mean(w["head.bias"])),
    }
    return {"weights": w, "diagnostics": diag}
