import numpy as np

def edit(w, f):
    # Extract data
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float32)  # (2000, 784)
    y = f["y"].astype(np.int64)                                 # (2000,)
    H_cur = f["features"].astype(np.float32)                    # (2000, 128)
    logits_cur = f["logits"].astype(np.float32)                 # (2000, 10)

    # Current training accuracy
    pred_cur = np.argmax(logits_cur, axis=1)
    acc_cur = np.mean(pred_cur == y)

    # One-hot targets
    Y = np.zeros((y.size, 10), dtype=np.float32)
    Y[np.arange(y.size), y] = 1.0

    # Validation split (fixed seed for reproducibility)
    rng = np.random.RandomState(42)
    idx = rng.permutation(y.size)
    n_train = int(0.8 * y.size)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    # Helper: ridge regression for fc2 given features H
    def fit_fc2(H, Y, train_idx, val_idx, lambdas):
        best_lambda = None
        best_val_acc = -1
        best_W = None
        best_b = None
        H_train = H[train_idx]
        Y_train = Y[train_idx]
        H_val = H[val_idx]
        Y_val = Y[val_idx]
        # Augment with bias
        H_train_aug = np.hstack([H_train, np.ones((H_train.shape[0], 1), dtype=np.float32)])
        H_val_aug = np.hstack([H_val, np.ones((H_val.shape[0], 1), dtype=np.float32)])
        for lam in lambdas:
            # Solve (H^T H + lam I) W = H^T Y
            A = H_train_aug.T @ H_train_aug + lam * np.eye(H_train_aug.shape[1], dtype=np.float32)
            B = H_train_aug.T @ Y_train
            W_aug = np.linalg.solve(A, B)  # (129, 10)
            # Evaluate on validation
            logits_val = H_val_aug @ W_aug
            pred_val = np.argmax(logits_val, axis=1)
            acc_val = np.mean(pred_val == np.argmax(Y_val, axis=1))
            if acc_val > best_val_acc:
                best_val_acc = acc_val
                best_lambda = lam
                best_W = W_aug[:-1, :].T  # (10, 128)
                best_b = W_aug[-1, :]     # (10,)
        return best_W, best_b, best_lambda, best_val_acc

    # Candidate 1: use current features
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    W2_cur, b2_cur, lam_cur, val_acc_cur = fit_fc2(H_cur, Y, train_idx, val_idx, lambdas)

    # Candidate 2: PCA-based first layer
    # Center data
    mu = X.mean(axis=0)
    Xc = X - mu
    # SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # Top 128 components
    W1_pca = Vt[:128].astype(np.float32)  # (128, 784)
    b1_pca = - (mu @ W1_pca.T).astype(np.float32)  # (128,)
    # Compute new features
    H_pca = np.maximum(0, X @ W1_pca.T + b1_pca)  # (2000, 128)
    W2_pca, b2_pca, lam_pca, val_acc_pca = fit_fc2(H_pca, Y, train_idx, val_idx, lambdas)

    # Choose better candidate based on validation accuracy
    if val_acc_pca > val_acc_cur:
        # Use PCA model
        w["fc1.weight"] = W1_pca
        w["fc1.bias"] = b1_pca
        w["fc2.weight"] = W2_pca
        w["fc2.bias"] = b2_pca
        chosen = "pca"
        val_acc = val_acc_pca
        lam = lam_pca
    else:
        # Use current features, only update fc2
        w["fc2.weight"] = W2_cur
        w["fc2.bias"] = b2_cur
        chosen = "current"
        val_acc = val_acc_cur
        lam = lam_cur

    # Diagnostics
    diagnostics = {
        "acc_cur": float(acc_cur),
        "val_acc_cur": float(val_acc_cur),
        "val_acc_pca": float(val_acc_pca),
        "chosen": chosen,
        "lambda": float(lam),
        "val_acc": float(val_acc),
        "fc1_weight_mean": float(w["fc1.weight"].mean()),
        "fc1_weight_std": float(w["fc1.weight"].std()),
        "fc2_weight_mean": float(w["fc2.weight"].mean()),
        "fc2_weight_std": float(w["fc2.weight"].std()),
    }

    return {"weights": w, "diagnostics": diagnostics}
