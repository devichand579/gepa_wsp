import numpy as np

def edit(w, f):
    # Extract feedback data
    x = f["x"]                     # [2000,1,28,28]
    y = f["y"].astype(np.int64)    # [2000]
    features = f["features"]       # [2000,128]
    X = x.reshape(len(x), -1).astype(np.float32)  # [2000,784]

    # Current fc1 parameters
    W1 = w["fc1.weight"].copy()
    b1 = w["fc1.bias"].copy()

    def get_features(X_in, W1_in, b1_in):
        return np.maximum(0.0, X_in @ W1_in.T + b1_in)

    def augment_data(X_in, y_in, shifts=(-2, -1, 1, 2)):
        N = X_in.shape[0]
        X_img = X_in.reshape(N, 28, 28)
        pad = 4
        X_pad = np.zeros((N, 28 + 2 * pad, 28 + 2 * pad), dtype=X_in.dtype)
        X_pad[:, pad:pad + 28, pad:pad + 28] = X_img
        X_list = [X_in]
        y_list = [y_in]
        for dx in shifts:
            for dy in shifts:
                X_shift = X_pad[:, pad + dx:pad + dx + 28, pad + dy:pad + dy + 28]
                X_list.append(X_shift.reshape(N, -1))
                y_list.append(y_in)
        return np.vstack(X_list), np.concatenate(y_list)

    def fit_ridge(H, y_in, lam):
        N, D = H.shape
        H_aug = np.hstack([H, np.ones((N, 1), dtype=H.dtype)])
        Y = np.zeros((N, 10), dtype=H.dtype)
        Y[np.arange(N), y_in] = 1.0
        A = H_aug.T @ H_aug + lam * np.eye(D + 1, dtype=H.dtype)
        B = H_aug.T @ Y
        W = np.linalg.solve(A, B)
        return W[:D].T.astype(np.float32), W[D].astype(np.float32)

    def fit_lda(H, y_in, shrink):
        N, D = H.shape
        classes = np.unique(y_in)
        K = len(classes)
        mu = np.zeros((K, D), dtype=H.dtype)
        pi = np.zeros(K, dtype=H.dtype)
        for i, c in enumerate(classes):
            idx = (y_in == c)
            mu[i] = H[idx].mean(axis=0)
            pi[i] = idx.sum() / N
        Sigma = np.zeros((D, D), dtype=H.dtype)
        for i, c in enumerate(classes):
            idx = (y_in == c)
            Hc = H[idx] - mu[i]
            Sigma += Hc.T @ Hc
        Sigma /= N
        Sigma_reg = Sigma + shrink * np.eye(D, dtype=H.dtype)
        Sigma_inv = np.linalg.inv(Sigma_reg)
        W = Sigma_inv @ mu.T          # [D, K]
        b = -0.5 * np.sum(mu * (Sigma_inv @ mu.T).T, axis=1) + np.log(pi + 1e-12)
        return W.T.astype(np.float32), b.astype(np.float32)

    def cv_score(X_in, y_in, W1_in, b1_in, augment, method, param, n_folds=5):
        N = len(y_in)
        rng = np.random.RandomState(42)
        idx = np.arange(N)
        rng.shuffle(idx)
        folds = np.array_split(idx, n_folds)
        accs = []
        for i in range(n_folds):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            X_tr, y_tr = X_in[train_idx], y_in[train_idx]
            X_te, y_te = X_in[test_idx], y_in[test_idx]
            if augment:
                X_tr, y_tr = augment_data(X_tr, y_tr)
            H_tr = get_features(X_tr, W1_in, b1_in)
            H_te = get_features(X_te, W1_in, b1_in)
            if method == "ridge":
                W2, b2 = fit_ridge(H_tr, y_tr, param)
            else:
                W2, b2 = fit_lda(H_tr, y_tr, param)
            logits = H_te @ W2.T + b2
            pred = np.argmax(logits, axis=1)
            accs.append(np.mean(pred == y_te))
        return float(np.mean(accs))

    # Evaluate a few sensible closed-form options on the feedback pool.
    # Keep fc1 fixed; only refit fc2.
    candidates = []
    for lam in (0.01, 0.1, 1.0, 10.0, 100.0):
        candidates.append(("ridge", lam, False))
    for shrink in (0.01, 0.1, 1.0, 10.0):
        candidates.append(("lda", shrink, False))
    for lam in (0.01, 0.1, 1.0, 10.0):
        candidates.append(("ridge", lam, True))
    for shrink in (0.01, 0.1, 1.0):
        candidates.append(("lda", shrink, True))

    best = None
    best_score = -1.0
    for method, param, augment in candidates:
        score = cv_score(X, y, W1, b1, augment, method, param)
        if score > best_score:
            best_score = score
            best = (method, param, augment)

    method, param, augment = best

    # Refit fc2 on the full feedback pool (with augmentation if selected).
    if augment:
        X_full, y_full = augment_data(X, y)
    else:
        X_full, y_full = X, y
    H_full = get_features(X_full, W1, b1)
    if method == "ridge":
        W2, b2 = fit_ridge(H_full, y_full, param)
    else:
        W2, b2 = fit_lda(H_full, y_full, param)

    # Write back only fc2 parameters; fc1 is left unchanged.
    w["fc2.weight"] = W2.astype(np.float32)
    w["fc2.bias"] = b2.astype(np.float32)

    # Small diagnostics for later reflection.
    train_logits = H_full @ W2.T + b2
    train_pred = np.argmax(train_logits, axis=1)
    train_acc = float(np.mean(train_pred == y_full))
    diag = {
        "chosen_method": method,
        "chosen_param": float(param),
        "augment": bool(augment),
        "cv_score": float(best_score),
        "train_accuracy": train_acc,
        "fc2_weight_mean": float(W2.mean()),
        "fc2_weight_std": float(W2.std()),
        "fc2_bias_mean": float(b2.mean()),
        "fc2_bias_std": float(b2.std()),
    }
    return {"weights": w, "diagnostics": diag}
