import numpy as np

def edit(w, f):
    # Extract feedback data
    X_feat = f["features"].astype(np.float64)  # (2000, 128)
    y = f["y"].astype(np.int64)                # (2000,)
    N, D = X_feat.shape
    C = 10

    # Helper: LDA classifier with shrinkage alpha
    def fit_lda(X, y, alpha):
        mu = np.zeros((C, D))
        for c in range(C):
            mu[c] = X[y == c].mean(axis=0)
        S = np.zeros((D, D))
        for c in range(C):
            Xc = X[y == c] - mu[c]
            S += Xc.T @ Xc
        S /= (N - C)
        S_reg = S + alpha * np.eye(D)
        # Solve S_reg @ W.T = mu.T  => W = (S_reg^{-1} mu.T).T
        W = np.linalg.solve(S_reg, mu.T).T  # (C, D)
        # Bias: -0.5 * sum(mu * W, axis=1) + log(prior)
        prior = np.bincount(y, minlength=C) / N
        b = -0.5 * np.sum(mu * W, axis=1) + np.log(prior + 1e-12)
        return W, b

    # Helper: evaluate accuracy on given data
    def accuracy(W, b, X, y):
        logits = X @ W.T + b
        pred = np.argmax(logits, axis=1)
        return np.mean(pred == y)

    # Cross-validation to select alpha for LDA
    alphas = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
    best_alpha = None
    best_cv_acc = -1.0
    # 5-fold CV
    n_folds = 5
    indices = np.arange(N)
    np.random.seed(42)  # for reproducibility
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    for alpha in alphas:
        cv_accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            X_train, y_train = X_feat[train_idx], y[train_idx]
            X_val, y_val = X_feat[val_idx], y[val_idx]
            W, b = fit_lda(X_train, y_train, alpha)
            acc = accuracy(W, b, X_val, y_val)
            cv_accs.append(acc)
        mean_acc = np.mean(cv_accs)
        if mean_acc > best_cv_acc:
            best_cv_acc = mean_acc
            best_alpha = alpha

    # Final fit on all data with best alpha
    W_new, b_new = fit_lda(X_feat, y, best_alpha)

    # Optionally blend with original fc2 to preserve some prior? 
    # But pure LDA with best alpha already gave good results in prior runs.
    # We'll use pure LDA.
    w["fc2.weight"] = W_new.astype(np.float32)
    w["fc2.bias"] = b_new.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "best_alpha": float(best_alpha),
        "cv_accuracy": float(best_cv_acc),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
        "fc2_weight_mean": float(np.mean(w["fc2.weight"])),
        "fc2_weight_std": float(np.std(w["fc2.weight"])),
    }
    return {"weights": w, "diagnostics": diagnostics}
