import numpy as np

def edit(w, f):
    # Extract current weights
    W1 = w["fc1.weight"].copy()   # [128, 784]
    b1 = w["fc1.bias"].copy()     # [128]
    W2 = w["fc2.weight"].copy()   # [10, 128]
    b2 = w["fc2.bias"].copy()     # [10]

    # Full feedback pool
    X = f["x"]                    # [2000, 1, 28, 28]
    y = f["y"]                    # [2000]
    H = f["features"]             # [2000, 128]  (current fc1 outputs after ReLU)

    N, D = H.shape
    C = 10

    # One-hot labels
    Y = np.zeros((N, C), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    # Standardise features (zero mean, unit variance per dimension)
    mu = H.mean(axis=0, keepdims=True)
    sigma = H.std(axis=0, keepdims=True)
    sigma[sigma < 1e-6] = 1.0
    Hs = (H - mu) / sigma

    # Add bias column
    Hs_b = np.concatenate([Hs, np.ones((N, 1), dtype=np.float32)], axis=1)  # [N, 129]

    # Ridge regression: solve for W2_new (10 x 129) = Y^T * Hs_b * (Hs_b^T * Hs_b + lambda I)^-1
    # We'll use cross-validation to pick lambda.
    def ridge_fit(Hs_b, Y, lam):
        A = Hs_b.T @ Hs_b + lam * np.eye(Hs_b.shape[1], dtype=np.float32)
        B = Hs_b.T @ Y
        W = np.linalg.solve(A, B)   # [129, 10]
        return W.T                  # [10, 129]

    # 5-fold cross-validation
    n_folds = 5
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    best_lam = None
    best_acc = -1.0

    for lam in lambdas:
        acc_sum = 0.0
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            Hs_b_train = Hs_b[train_idx]
            Y_train = Y[train_idx]
            Hs_b_val = Hs_b[val_idx]
            Y_val = Y[val_idx]
            W_fold = ridge_fit(Hs_b_train, Y_train, lam)
            logits_val = Hs_b_val @ W_fold.T
            preds = np.argmax(logits_val, axis=1)
            acc_sum += np.mean(preds == y[val_idx])
        acc = acc_sum / n_folds
        if acc > best_acc:
            best_acc = acc
            best_lam = lam

    # Fit on full data with best lambda
    W2_new = ridge_fit(Hs_b, Y, best_lam)   # [10, 129]
    W2_weight = W2_new[:, :128]             # [10, 128]
    W2_bias = W2_new[:, 128]                # [10]

    # Update weights
    w["fc2.weight"] = W2_weight.astype(np.float32)
    w["fc2.bias"] = W2_bias.astype(np.float32)

    # Diagnostics
    logits_full = Hs_b @ W2_new.T
    preds_full = np.argmax(logits_full, axis=1)
    train_acc = np.mean(preds_full == y)
    ce = -np.mean(np.log(np.exp(logits_full[np.arange(N), y]) / np.sum(np.exp(logits_full), axis=1) + 1e-12))

    diagnostics = {
        "best_lambda": float(best_lam),
        "cv_accuracy": float(best_acc),
        "train_accuracy": float(train_acc),
        "train_cross_entropy": float(ce),
        "fc2_weight_mean": float(W2_weight.mean()),
        "fc2_weight_std": float(W2_weight.std()),
        "fc2_bias_mean": float(W2_bias.mean()),
        "fc2_bias_std": float(W2_bias.std()),
    }

    return {"weights": w, "diagnostics": diagnostics}
