import numpy as np

def edit(w, f):
    X = f["features"].astype(np.float64)   # (2000, 128)
    y = f["y"].astype(np.int64)            # (2000,)
    N, D = X.shape
    C = 10

    # Class means and global mean
    means = np.zeros((C, D))
    for c in range(C):
        means[c] = X[y == c].mean(axis=0)
    mu = X.mean(axis=0)

    # Within-class scatter
    S_w = np.zeros((D, D))
    for c in range(C):
        Xc = X[y == c] - means[c]
        S_w += Xc.T @ Xc
    cov = S_w / (N - C)

    # 5-fold cross-validation to choose shrinkage alpha
    indices = np.arange(N)
    rng = np.random.RandomState(0)
    rng.shuffle(indices)
    folds = np.array_split(indices, 5)

    alphas = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
    best_alpha = 0.1
    best_acc = 0.0

    for alpha in alphas:
        accs = []
        for i in range(5):
            val_idx = folds[i]
            tr_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # LDA on training fold
            means_tr = np.zeros((C, D))
            for c in range(C):
                means_tr[c] = X_tr[y_tr == c].mean(axis=0)
            mu_tr = X_tr.mean(axis=0)

            S_w_tr = np.zeros((D, D))
            for c in range(C):
                Xc = X_tr[y_tr == c] - means_tr[c]
                S_w_tr += Xc.T @ Xc
            cov_tr = S_w_tr / (len(y_tr) - C)

            # Shrinkage
            cov_reg = (1 - alpha) * cov_tr + alpha * (np.trace(cov_tr) / D) * np.eye(D)
            W = np.linalg.solve(cov_reg, (means_tr - mu_tr).T)  # (D, C)
            b = -W.T @ mu_tr

            # Predict on validation
            logits = X_val @ W + b
            preds = np.argmax(logits, axis=1)
            accs.append((preds == y_val).mean())
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_alpha = alpha

    # Final LDA on all data with best alpha
    cov_reg = (1 - best_alpha) * cov + best_alpha * (np.trace(cov) / D) * np.eye(D)
    W = np.linalg.solve(cov_reg, (means - mu).T)  # (D, C)
    b = -W.T @ mu

    # Update fc2
    w["fc2.weight"] = W.T.astype(np.float32)
    w["fc2.bias"] = b.astype(np.float32)

    diagnostics = {
        "best_alpha": float(best_alpha),
        "cv_accuracy": float(best_acc),
        "fc2_weight_mean": float(w["fc2.weight"].mean()),
        "fc2_weight_std": float(w["fc2.weight"].std()),
        "fc2_bias_mean": float(w["fc2.bias"].mean()),
        "fc2_bias_std": float(w["fc2.bias"].std()),
    }
    return {"weights": w, "diagnostics": diagnostics}
