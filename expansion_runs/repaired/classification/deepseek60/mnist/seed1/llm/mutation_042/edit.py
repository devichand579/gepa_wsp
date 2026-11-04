import numpy as np

def edit(w, f):
    # Extract data
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float32)  # (2000, 784)
    y = f["y"].astype(np.int64)                                  # (2000,)
    N, D = X.shape
    C = 10

    # Current fc1 weights as candidates
    W1_cur = w["fc1.weight"].astype(np.float32)  # (128, 784)

    # Class means
    mu = np.zeros((C, D), dtype=np.float32)
    counts = np.zeros(C, dtype=np.float32)
    for c in range(C):
        mask = (y == c)
        counts[c] = mask.sum()
        if counts[c] > 0:
            mu[c] = X[mask].mean(axis=0)
    mu0 = X.mean(axis=0)

    # Candidate pool
    candidates = []
    candidates.append(W1_cur)  # 128

    # Class means and differences
    candidates.append(mu - mu0)  # 10
    for i in range(C):
        for j in range(i+1, C):
            candidates.append((mu[i] - mu[j])[None, :])  # 45

    # LDA directions (up to 9)
    # Compute within-class scatter Sw and between-class scatter Sb
    X_centered = X - mu0
    T = X_centered.T @ X_centered  # total scatter
    Sb = np.zeros((D, D), dtype=np.float32)
    for c in range(C):
        diff = (mu[c] - mu0)[:, None]
        Sb += counts[c] * (diff @ diff.T)
    Sw = T - Sb
    # Regularize Sw
    Sw_reg = Sw + 1e-3 * np.eye(D, dtype=np.float32)
    try:
        L = np.linalg.cholesky(Sw_reg)
        # M = L^{-1} Sb L^{-T}
        A = np.linalg.solve(L, Sb)
        M = np.linalg.solve(L, A.T).T
        M = (M + M.T) / 2.0  # symmetrize
        eigvals, eigvecs = np.linalg.eigh(M)
        # Take top 9 (largest eigenvalues)
        idx = np.argsort(eigvals)[::-1][:9]
        for k in idx:
            v = eigvecs[:, k]
            lda_dir = np.linalg.solve(L.T, v)
            candidates.append(lda_dir[None, :])
    except np.linalg.LinAlgError:
        pass

    # PCA directions (top 300)
    X_cent = X - mu0
    # Use SVD on X_cent (2000 x 784)
    U, S, Vt = np.linalg.svd(X_cent, full_matrices=False)
    n_pca = min(300, Vt.shape[0])
    candidates.append(Vt[:n_pca])  # (n_pca, 784)

    # Random projections (2000)
    rng = np.random.RandomState(0)
    rand_dirs = rng.randn(2000, D).astype(np.float32)
    candidates.append(rand_dirs)

    # Ridge linear classifier on raw pixels (10 directions)
    # Solve (X^T X + lambda I) W = X^T Y_onehot
    Y_onehot = np.zeros((N, C), dtype=np.float32)
    Y_onehot[np.arange(N), y] = 1.0
    lam_lin = 1e-2
    XtX = X.T @ X + lam_lin * np.eye(D, dtype=np.float32)
    XtY = X.T @ Y_onehot
    W_lin = np.linalg.solve(XtX, XtY).T  # (10, 784)
    candidates.append(W_lin)

    # Concatenate all candidates
    C_all = np.vstack(candidates).astype(np.float32)  # (n_cand, 784)
    n_cand = C_all.shape[0]

    # Normalize each candidate to unit norm
    norms = np.linalg.norm(C_all, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    C_all = C_all / norms

    # Compute ANOVA F-value for each candidate
    P = X @ C_all.T  # (2000, n_cand)
    overall_mean = P.mean(axis=0)  # (n_cand,)
    between = np.zeros(n_cand, dtype=np.float32)
    within = np.zeros(n_cand, dtype=np.float32)
    for c in range(C):
        mask = (y == c)
        if counts[c] > 0:
            mean_c = P[mask].mean(axis=0)
            diff = mean_c - overall_mean
            between += counts[c] * (diff * diff)
            # within: sum of squared deviations from class mean
            P_c = P[mask]
            within += ((P_c - mean_c) ** 2).sum(axis=0)
    # Avoid division by zero
    within = np.maximum(within, 1e-12)
    F = (between / (C - 1)) / (within / (N - C))

    # Select top 128
    top_idx = np.argsort(F)[::-1][:128]
    W1_new = C_all[top_idx]  # (128, 784)

    # Compute projections Z = X @ W1_new.T
    Z = X @ W1_new.T  # (2000, 128)

    # Try different bias strategies and ridge lambdas via 5-fold CV
    bias_strategies = {
        "zero": np.zeros(128, dtype=np.float32),
        "mean": -Z.mean(axis=0),
        "half_std": -0.5 * Z.std(axis=0),
        "mean_plus_half_std": -Z.mean(axis=0) + 0.5 * Z.std(axis=0),
    }
    lambdas = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]

    # 5-fold CV
    n_folds = 5
    fold_size = N // n_folds
    indices = np.arange(N)
    rng_cv = np.random.RandomState(42)
    rng_cv.shuffle(indices)

    best_cv_acc = -1.0
    best_bias_name = None
    best_lambda = None
    best_bias = None

    for bias_name, b1 in bias_strategies.items():
        H = np.maximum(Z + b1, 0.0)  # (2000, 128)
        H_aug = np.concatenate([H, np.ones((N, 1), dtype=np.float32)], axis=1)  # (2000, 129)
        for lam in lambdas:
            accs = []
            for fold in range(n_folds):
                val_idx = indices[fold * fold_size:(fold + 1) * fold_size]
                train_idx = np.setdiff1d(indices, val_idx)
                H_tr = H_aug[train_idx]
                Y_tr = Y_onehot[train_idx]
                H_val = H_aug[val_idx]
                Y_val = Y_onehot[val_idx]
                # Ridge regression with no penalty on bias
                reg = lam * np.eye(129, dtype=np.float32)
                reg[-1, -1] = 0.0
                W_aug = np.linalg.solve(H_tr.T @ H_tr + reg, H_tr.T @ Y_tr)
                pred = H_val @ W_aug
                pred_class = np.argmax(pred, axis=1)
                true_class = np.argmax(Y_val, axis=1)
                acc = (pred_class == true_class).mean()
                accs.append(acc)
            mean_acc = np.mean(accs)
            if mean_acc > best_cv_acc:
                best_cv_acc = mean_acc
                best_bias_name = bias_name
                best_lambda = lam
                best_bias = b1

    # Refit on full data with best hyperparameters
    H_full = np.maximum(Z + best_bias, 0.0)
    H_full_aug = np.concatenate([H_full, np.ones((N, 1), dtype=np.float32)], axis=1)
    reg = best_lambda * np.eye(129, dtype=np.float32)
    reg[-1, -1] = 0.0
    W_aug = np.linalg.solve(H_full_aug.T @ H_full_aug + reg, H_full_aug.T @ Y_onehot)
    W2_new = W_aug[:128].T.astype(np.float32)  # (10, 128)
    b2_new = W_aug[128].astype(np.float32)     # (10,)

    # Compute training accuracy
    logits_train = H_full @ W2_new.T + b2_new
    pred_train = np.argmax(logits_train, axis=1)
    train_acc = (pred_train == y).mean()

    # Update weights
    w["fc1.weight"] = W1_new.astype(np.float32)
    w["fc1.bias"] = best_bias.astype(np.float32)
    w["fc2.weight"] = W2_new
    w["fc2.bias"] = b2_new

    # Clip to safe range
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)

    diagnostics = {
        "train_accuracy": float(train_acc),
        "cv_accuracy": float(best_cv_acc),
        "best_bias": best_bias_name,
        "best_lambda": float(best_lambda),
        "num_candidates": int(n_cand),
        "selected_indices": top_idx[:10].tolist(),
        "fc1_weight_std": float(w["fc1.weight"].std()),
        "fc2_weight_std": float(w["fc2.weight"].std()),
        "fc2_bias_mean": float(w["fc2.bias"].mean()),
    }

    return {"weights": w, "diagnostics": diagnostics}
