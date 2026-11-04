import numpy as np

def edit(w, f):
    # Load feedback data
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float32)
    y = f["y"].astype(np.int64)
    H = f["features"].astype(np.float32)
    N, D = X.shape
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    # ---------- closed-form linear model helpers ----------
    def ridge_fit(X_train, Y_train, lam):
        X_mean = X_train.mean(axis=0)
        Y_mean = Y_train.mean(axis=0)
        Xc = X_train - X_mean
        Yc = Y_train - Y_mean
        d = X_train.shape[1]
        A = Xc.T @ Xc + lam * np.eye(d, dtype=np.float32)
        B = Xc.T @ Yc
        W = np.linalg.solve(A, B)
        b = Y_mean - X_mean @ W
        return W.T.astype(np.float32), b.astype(np.float32)

    def lda_fit(X_train, Y_train, alpha):
        N_train = X_train.shape[0]
        d = X_train.shape[1]
        mus = np.zeros((C, d), dtype=np.float32)
        for c in range(C):
            idx = Y_train[:, c] > 0.5
            if idx.sum() > 0:
                mus[c] = X_train[idx].mean(axis=0)
        S = np.zeros((d, d), dtype=np.float32)
        for c in range(C):
            idx = Y_train[:, c] > 0.5
            if idx.sum() > 0:
                Xc = X_train[idx] - mus[c]
                S += Xc.T @ Xc
        S /= max(N_train - C, 1)
        S_shrunk = (1 - alpha) * S + alpha * (np.trace(S) / d) * np.eye(d, dtype=np.float32)
        S_inv = np.linalg.inv(S_shrunk)
        W = S_inv @ mus.T  # d x C
        b = -0.5 * np.sum(mus * (S_inv @ mus.T).T, axis=1)
        priors = np.array([(Y_train[:, c] > 0.5).sum() / N_train for c in range(C)], dtype=np.float32)
        b += np.log(priors + 1e-8)
        return W.T.astype(np.float32), b.astype(np.float32)

    def predict(X_test, W, b):
        return np.argmax(X_test @ W.T + b, axis=1)

    def cv_accuracy(X, y, method, param, folds=5):
        N = X.shape[0]
        indices = np.arange(N)
        rng = np.random.RandomState(42)
        rng.shuffle(indices)
        fold_size = N // folds
        accs = []
        for i in range(folds):
            start = i * fold_size
            end = N if i == folds - 1 else (i + 1) * fold_size
            test_idx = indices[start:end]
            train_idx = np.setdiff1d(indices, test_idx)
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            Y_train = np.zeros((len(y_train), C), dtype=np.float32)
            Y_train[np.arange(len(y_train)), y_train] = 1.0
            if method == 'ridge':
                W, b = ridge_fit(X_train, Y_train, param)
            else:
                W, b = lda_fit(X_train, Y_train, param)
            pred = predict(X_test, W, b)
            accs.append((pred == y_test).mean())
        return float(np.mean(accs))

    # ---------- 1. best linear classifier on raw pixels ----------
    best_lin_acc = 0.0
    best_lin_method = 'ridge'
    best_lin_param = 1.0
    for method in ['ridge', 'lda']:
        params = [0.1, 1.0, 10.0, 100.0] if method == 'ridge' else [0.01, 0.1, 0.5]
        for param in params:
            acc = cv_accuracy(X, y, method, param)
            if acc > best_lin_acc:
                best_lin_acc = acc
                best_lin_method = method
                best_lin_param = param

    if best_lin_method == 'ridge':
        W_lin, b_lin = ridge_fit(X, Y, best_lin_param)
    else:
        W_lin, b_lin = lda_fit(X, Y, best_lin_param)

    # ---------- 2. feature importance via F-statistic ----------
    H_mean = H.mean(axis=0)
    H_class_means = np.zeros((C, H.shape[1]), dtype=np.float32)
    for c in range(C):
        idx = y == c
        if idx.sum() > 0:
            H_class_means[c] = H[idx].mean(axis=0)
    between = np.zeros(H.shape[1], dtype=np.float32)
    for c in range(C):
        n_c = (y == c).sum()
        between += n_c * (H_class_means[c] - H_mean) ** 2
    within = np.zeros(H.shape[1], dtype=np.float32)
    for c in range(C):
        idx = y == c
        if idx.sum() > 0:
            within += ((H[idx] - H_class_means[c]) ** 2).sum(axis=0)
    F = between / (within + 1e-8)
    importance_order = np.argsort(F)[::-1]  # most important first

    # ---------- 3. new linear features ----------
    scores = X @ W_lin.T + b_lin  # N x 10
    H_lin = np.zeros((N, 20), dtype=np.float32)
    for c in range(10):
        H_lin[:, 2 * c] = np.maximum(scores[:, c], 0)
        H_lin[:, 2 * c + 1] = np.maximum(-scores[:, c], 0)

    # ---------- 4. search over number of replaced units and fc2 model ----------
    best_combined_acc = 0.0
    best_n_replace = 0
    best_fc2_method = 'ridge'
    best_fc2_param = 1.0
    best_keep = None
    best_H_new = None

    for n_replace in [0, 5, 10, 15, 20]:
        n_keep = 128 - n_replace
        keep = importance_order[:n_keep]
        H_keep = H[:, keep]
        H_new = np.concatenate([H_keep, H_lin[:, :n_replace]], axis=1)
        for method in ['ridge', 'lda']:
            params = [0.001, 0.01, 0.1, 1.0, 10.0] if method == 'ridge' else [0.01, 0.1, 0.5]
            for param in params:
                acc = cv_accuracy(H_new, y, method, param)
                if acc > best_combined_acc:
                    best_combined_acc = acc
                    best_n_replace = n_replace
                    best_fc2_method = method
                    best_fc2_param = param
                    best_keep = keep
                    best_H_new = H_new

    # ---------- 5. final fc2 fit ----------
    if best_fc2_method == 'ridge':
        W_fc2, b_fc2 = ridge_fit(best_H_new, Y, best_fc2_param)
    else:
        W_fc2, b_fc2 = lda_fit(best_H_new, Y, best_fc2_param)

    # ---------- 6. rebuild fc1 ----------
    new_fc1_weight = np.zeros((128, 784), dtype=np.float32)
    new_fc1_bias = np.zeros(128, dtype=np.float32)
    orig_fc1_weight = w["fc1.weight"].astype(np.float32)
    orig_fc1_bias = w["fc1.bias"].astype(np.float32)

    for i, idx in enumerate(best_keep):
        new_fc1_weight[i] = orig_fc1_weight[idx]
        new_fc1_bias[i] = orig_fc1_bias[idx]

    for j in range(best_n_replace):
        c = j // 2
        if j % 2 == 0:
            new_fc1_weight[len(best_keep) + j] = W_lin[c]
            new_fc1_bias[len(best_keep) + j] = b_lin[c]
        else:
            new_fc1_weight[len(best_keep) + j] = -W_lin[c]
            new_fc1_bias[len(best_keep) + j] = -b_lin[c]

    # clamp to safe range
    new_fc1_weight = np.clip(new_fc1_weight, -1000.0, 1000.0)
    new_fc1_bias = np.clip(new_fc1_bias, -1000.0, 1000.0)
    W_fc2 = np.clip(W_fc2, -1000.0, 1000.0)
    b_fc2 = np.clip(b_fc2, -1000.0, 1000.0)

    w["fc1.weight"] = new_fc1_weight.astype(np.float32)
    w["fc1.bias"] = new_fc1_bias.astype(np.float32)
    w["fc2.weight"] = W_fc2.astype(np.float32)
    w["fc2.bias"] = b_fc2.astype(np.float32)

    diagnostics = {
        "best_lin_acc": float(best_lin_acc),
        "best_lin_method": best_lin_method,
        "best_lin_param": float(best_lin_param),
        "best_combined_acc": float(best_combined_acc),
        "best_n_replace": int(best_n_replace),
        "best_fc2_method": best_fc2_method,
        "best_fc2_param": float(best_fc2_param),
        "fc1_weight_mean": float(w["fc1.weight"].mean()),
        "fc1_weight_std": float(w["fc1.weight"].std()),
        "fc2_weight_mean": float(w["fc2.weight"].mean()),
        "fc2_weight_std": float(w["fc2.weight"].std()),
        "fc2_bias_mean": float(w["fc2.bias"].mean()),
        "fc2_bias_std": float(w["fc2.bias"].std()),
    }
    return {"weights": w, "diagnostics": diagnostics}
