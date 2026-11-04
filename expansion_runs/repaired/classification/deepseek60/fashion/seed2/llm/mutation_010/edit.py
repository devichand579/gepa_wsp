import numpy as np

def edit(w, f):
    # Load full feedback features and labels
    X = f["features"].astype(np.float64)   # [2000, 1568]
    y = f["y"].astype(np.int64)            # [2000]
    N, D = X.shape
    C = 10

    # Helper: fit LDA with shrinkage alpha on given data
    def fit_lda(X_train, y_train, alpha):
        means = np.zeros((C, D), dtype=np.float64)
        priors = np.zeros(C, dtype=np.float64)
        for c in range(C):
            mask = (y_train == c)
            n_c = np.sum(mask)
            if n_c > 0:
                means[c] = X_train[mask].mean(axis=0)
                priors[c] = n_c / len(y_train)
            else:
                means[c] = 0.0
                priors[c] = 1.0 / C

        # Within-class scatter
        S = np.zeros((D, D), dtype=np.float64)
        for c in range(C):
            mask = (y_train == c)
            if np.sum(mask) > 1:
                Xc = X_train[mask] - means[c]
                S += Xc.T @ Xc

        # Pooled covariance
        Sigma = S / (len(y_train) - C)
        mean_diag = np.trace(Sigma) / D
        Sigma_shrunk = (1.0 - alpha) * Sigma + alpha * mean_diag * np.eye(D, dtype=np.float64)

        # Solve Sigma_shrunk @ W.T = means.T
        W = np.linalg.solve(Sigma_shrunk, means.T).T   # [C, D]
        b = -0.5 * np.sum(means * W, axis=1) + np.log(priors)
        return W, b

    # Single train/validation split for alpha selection
    rng = np.random.RandomState(42)
    perm = rng.permutation(N)
    val_size = 400
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    alphas = [0.0001, 0.001, 0.01, 0.1]
    best_alpha = 0.001
    best_acc = -1.0
    for alpha in alphas:
        W, b = fit_lda(X[train_idx], y[train_idx], alpha)
        logits = X[val_idx] @ W.T + b
        pred = np.argmax(logits, axis=1)
        acc = np.mean(pred == y[val_idx])
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha

    # Final fit on full pool
    W, b = fit_lda(X, y, best_alpha)

    # Scale logits to match current head's logit standard deviation
    current_W = w["head.weight"].astype(np.float64)
    current_b = w["head.bias"].astype(np.float64)
    current_logits = X @ current_W.T + current_b
    target_std = np.std(current_logits)

    new_logits = X @ W.T + b
    new_std = np.std(new_logits)
    if new_std > 1e-12:
        s = target_std / new_std
    else:
        s = 1.0

    W = W * s
    b = b * s

    # Write back to weights
    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

    diagnostics = {
        "best_alpha": float(best_alpha),
        "val_acc": float(best_acc),
        "scale": float(s),
        "new_std": float(new_std),
        "target_std": float(target_std),
        "W_std": float(np.std(W)),
        "b_mean": float(np.mean(b)),
    }
    return {"weights": w, "diagnostics": diagnostics}
