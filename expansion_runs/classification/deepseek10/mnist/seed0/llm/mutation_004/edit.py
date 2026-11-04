import numpy as np

def edit(w, f):
    # Extract full feedback arrays
    X = f["features"].astype(np.float64)   # (2000, 128)
    y = f["y"].astype(np.int64)            # (2000,)
    N, D = X.shape
    C = 10

    # Helper: regularized LDA on features
    def fit_lda(X_tr, y_tr, reg):
        mu = np.zeros((C, D))
        for c in range(C):
            idx = (y_tr == c)
            if np.sum(idx) > 0:
                mu[c] = X_tr[idx].mean(axis=0)
        # Within-class scatter
        Sw = np.zeros((D, D))
        for c in range(C):
            idx = (y_tr == c)
            if np.sum(idx) > 0:
                Xc = X_tr[idx] - mu[c]
                Sw += Xc.T @ Xc
        Sw /= len(y_tr)
        Sw_reg = Sw + reg * np.eye(D)
        # Solve for weights: W = Sw_reg^{-1} mu^T
        # Use solve for stability
        W = np.linalg.solve(Sw_reg, mu.T).T  # (C, D)
        # Bias: -0.5 * mu_c^T Sw_reg^{-1} mu_c + log(prior)
        # Compute quadratic term
        quad = np.zeros(C)
        for c in range(C):
            quad[c] = mu[c] @ np.linalg.solve(Sw_reg, mu[c])
        # Priors (assume uniform or use class frequencies)
        priors = np.array([np.mean(y_tr == c) for c in range(C)])
        b = -0.5 * quad + np.log(priors + 1e-12)
        return W, b

    # Helper: ridge regression on one-hot targets
    def fit_ridge(X_tr, y_tr, alpha):
        Y = np.zeros((len(y_tr), C))
        Y[np.arange(len(y_tr)), y_tr] = 1.0
        # Augment with bias
        X_aug = np.hstack([X_tr, np.ones((len(y_tr), 1))])
        # Solve (X^T X + alpha I) W = X^T Y
        A = X_aug.T @ X_aug + alpha * np.eye(D + 1)
        B = X_aug.T @ Y
        W_aug = np.linalg.solve(A, B)  # (D+1, C)
        W = W_aug[:-1, :].T  # (C, D)
        b = W_aug[-1, :]     # (C,)
        return W, b

    # Split into train/val (use 1600/400)
    rng = np.random.RandomState(42)
    perm = rng.permutation(N)
    n_val = 400
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    best_acc = -1.0
    best_method = None
    best_params = None
    best_W = None
    best_b = None

    # Try LDA with various regularization
    for reg in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]:
        W, b = fit_lda(X_tr, y_tr, reg)
        logits = X_val @ W.T + b
        pred = np.argmax(logits, axis=1)
        acc = np.mean(pred == y_val)
        if acc > best_acc:
            best_acc = acc
            best_method = "lda"
            best_params = reg
            best_W = W
            best_b = b

    # Try ridge with various alphas
    for alpha in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]:
        W, b = fit_ridge(X_tr, y_tr, alpha)
        logits = X_val @ W.T + b
        pred = np.argmax(logits, axis=1)
        acc = np.mean(pred == y_val)
        if acc > best_acc:
            best_acc = acc
            best_method = "ridge"
            best_params = alpha
            best_W = W
            best_b = b

    # Refit on full data with best method and params
    if best_method == "lda":
        W_full, b_full = fit_lda(X, y, best_params)
    else:
        W_full, b_full = fit_ridge(X, y, best_params)

    # Update weights
    w["fc2.weight"] = W_full.astype(np.float32)
    w["fc2.bias"] = b_full.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "method": best_method,
        "param": float(best_params),
        "val_acc": float(best_acc),
        "fc2_weight_mean": float(np.mean(w["fc2.weight"])),
        "fc2_weight_std": float(np.std(w["fc2.weight"])),
        "fc2_bias_mean": float(np.mean(w["fc2.bias"])),
        "fc2_bias_std": float(np.std(w["fc2.bias"])),
    }

    return {"weights": w, "diagnostics": diagnostics}
