import numpy as np

def edit(w, f):
    # Extract full feedback pool
    X = f["features"].astype(np.float64)  # (2000, 128)
    y = f["y"].astype(np.int64)           # (2000,)
    N, D = X.shape
    C = 10

    # One-hot targets
    Y = np.zeros((N, C))
    Y[np.arange(N), y] = 1.0

    # Standardize features (avoid division by zero)
    mean_X = X.mean(axis=0)
    std_X = X.std(axis=0)
    std_X[std_X < 1e-6] = 1.0
    Xs = (X - mean_X) / std_X

    # Helper: evaluate a linear classifier on a subset
    def eval_ridge(alpha, train_idx, test_idx):
        X_tr = Xs[train_idx]
        Y_tr = Y[train_idx]
        X_te = Xs[test_idx]
        y_te = y[test_idx]
        X_tr_aug = np.hstack([X_tr, np.ones((len(train_idx), 1))])
        X_te_aug = np.hstack([X_te, np.ones((len(test_idx), 1))])
        A = X_tr_aug.T @ X_tr_aug + alpha * np.eye(D + 1)
        B = X_tr_aug.T @ Y_tr
        W = np.linalg.solve(A, B)
        pred = X_te_aug @ W
        return (pred.argmax(axis=1) == y_te).mean()

    # 5-fold CV for ridge
    alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    best_alpha = None
    best_cv_acc = -1.0
    np.random.seed(0)
    indices = np.random.permutation(N)
    folds = np.array_split(indices, 5)
    for alpha in alphas:
        accs = []
        for i in range(5):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            accs.append(eval_ridge(alpha, train_idx, test_idx))
        cv_acc = np.mean(accs)
        if cv_acc > best_cv_acc:
            best_cv_acc = cv_acc
            best_alpha = alpha

    # Full-data ridge with best alpha
    X_aug = np.hstack([Xs, np.ones((N, 1))])
    A = X_aug.T @ X_aug + best_alpha * np.eye(D + 1)
    B = X_aug.T @ Y
    W_ridge = np.linalg.solve(A, B)  # (D+1, C)

    # Transform back to original feature space
    W_orig = W_ridge[:D] / std_X[:, None]          # (D, C)
    b_orig = W_ridge[D] - mean_X @ W_orig          # (C,)

    # LDA with shrinkage
    mu = np.zeros((C, D))
    for c in range(C):
        mu[c] = X[y == c].mean(axis=0)
    S = np.zeros((D, D))
    for c in range(C):
        Xc = X[y == c] - mu[c]
        S += Xc.T @ Xc
    S /= (N - C)

    lambdas = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    best_lambda = None
    best_lda_cv = -1.0
    best_W_lda = None
    best_b_lda = None
    for lam in lambdas:
        S_reg = S + lam * np.trace(S) / D * np.eye(D)
        W_lda = np.linalg.solve(S_reg, mu.T)       # (D, C)
        b_lda = -0.5 * np.sum(mu.T * W_lda, axis=0) # (C,)
        accs = []
        for i in range(5):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            mu_tr = np.zeros((C, D))
            for c in range(C):
                mu_tr[c] = X[train_idx][y[train_idx] == c].mean(axis=0)
            S_tr = np.zeros((D, D))
            for c in range(C):
                Xc = X[train_idx][y[train_idx] == c] - mu_tr[c]
                S_tr += Xc.T @ Xc
            S_tr /= (len(train_idx) - C)
            S_tr_reg = S_tr + lam * np.trace(S_tr) / D * np.eye(D)
            W_tr = np.linalg.solve(S_tr_reg, mu_tr.T)
            b_tr = -0.5 * np.sum(mu_tr.T * W_tr, axis=0)
            pred = X[test_idx] @ W_tr + b_tr
            accs.append((pred.argmax(axis=1) == y[test_idx]).mean())
        cv_acc = np.mean(accs)
        if cv_acc > best_lda_cv:
            best_lda_cv = cv_acc
            best_lambda = lam
            best_W_lda = W_lda
            best_b_lda = b_lda

    # Full-pool accuracy for candidates
    def full_acc(W, b):
        pred = X @ W + b
        return (pred.argmax(axis=1) == y).mean()

    W_cur = w["fc2.weight"].astype(np.float64).T  # (128, 10)
    b_cur = w["fc2.bias"].astype(np.float64)
    acc_cur = full_acc(W_cur, b_cur)
    acc_ridge = full_acc(W_orig, b_orig)
    acc_lda = full_acc(best_W_lda, best_b_lda)

    candidates = [
        ("current", W_cur, b_cur, acc_cur),
        ("ridge", W_orig, b_orig, acc_ridge),
        ("lda", best_W_lda, best_b_lda, acc_lda)
    ]
    candidates.sort(key=lambda x: x[3], reverse=True)
    chosen_name, W_best, b_best, acc_best = candidates[0]

    if chosen_name != "current":
        # Clip to safe range and cast to float32
        W_best = np.clip(W_best, -1000.0, 1000.0)
        b_best = np.clip(b_best, -1000.0, 1000.0)
        w["fc2.weight"] = W_best.T.astype(np.float32)
        w["fc2.bias"] = b_best.astype(np.float32)

    diag = {
        "chosen": chosen_name,
        "acc_current": float(acc_cur),
        "acc_ridge": float(acc_ridge),
        "acc_lda": float(acc_lda),
        "best_alpha": float(best_alpha),
        "best_lambda": float(best_lambda),
        "cv_ridge": float(best_cv_acc),
        "cv_lda": float(best_lda_cv),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"]))
    }
    return {"weights": w, "diagnostics": diag}
