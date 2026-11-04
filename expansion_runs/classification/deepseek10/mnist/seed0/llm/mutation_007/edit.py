import numpy as np

def edit(w, f):
    # Extract full-pool hidden features and labels
    X = f["features"].astype(np.float64)   # (2000, 128)
    y = f["y"].astype(np.int64)            # (2000,)
    N, D = X.shape
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Add bias column for ridge
    Xb = np.hstack([X, np.ones((N, 1), dtype=np.float64)])
    Db = D + 1

    # 5-fold CV split (fixed seed for reproducibility)
    rng = np.random.RandomState(42)
    indices = np.arange(N)
    rng.shuffle(indices)
    folds = np.array_split(indices, 5)

    best_acc = -1.0
    best_method = None
    best_param = None
    best_W = None
    best_b = None

    # ---------- LDA with shrinkage ----------
    alphas_lda = np.logspace(-4, 1, 25)
    for alpha in alphas_lda:
        accs = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            X_tr = X[train_idx]
            y_tr = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]

            # Class means on training fold
            means = np.zeros((C, D), dtype=np.float64)
            for c in range(C):
                mask = (y_tr == c)
                if np.sum(mask) > 0:
                    means[c] = X_tr[mask].mean(axis=0)
            global_mean = X_tr.mean(axis=0)

            # Within-class scatter
            Sw = np.zeros((D, D), dtype=np.float64)
            for c in range(C):
                mask = (y_tr == c)
                if np.sum(mask) > 0:
                    Xc = X_tr[mask]
                    diff = Xc - means[c]
                    Sw += diff.T @ diff

            # Shrinkage
            Sw_reg = (1.0 - alpha) * Sw + alpha * (np.trace(Sw) / D) * np.eye(D)
            # Solve for weights
            # W = Sw_reg^{-1} @ (means - global_mean).T
            centered_means = means - global_mean
            W = np.linalg.solve(Sw_reg, centered_means.T)  # (D, C)
            b = -0.5 * np.sum(centered_means * W.T, axis=1)  # (C,)
            # Priors (assume uniform or use training priors)
            priors = np.array([np.mean(y_tr == c) for c in range(C)])
            b += np.log(priors + 1e-12)

            logits = X_val @ W + b
            pred = np.argmax(logits, axis=1)
            accs.append(np.mean(pred == y_val))
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_method = "lda"
            best_param = alpha
            best_W = W
            best_b = b

    # ---------- Ridge regression ----------
    alphas_ridge = np.logspace(-4, 3, 30)
    for alpha in alphas_ridge:
        accs = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            X_tr = Xb[train_idx]
            Y_tr = Y[train_idx]
            X_val = Xb[val_idx]
            y_val = y[val_idx]

            # Ridge normal equations
            A = X_tr.T @ X_tr + alpha * np.eye(Db)
            B = X_tr.T @ Y_tr
            W_full = np.linalg.solve(A, B)  # (Db, C)
            W = W_full[:D, :].T  # (C, D)
            b = W_full[D, :]     # (C,)

            logits = X_val[:, :D] @ W.T + b
            pred = np.argmax(logits, axis=1)
            accs.append(np.mean(pred == y_val))
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_method = "ridge"
            best_param = alpha
            best_W = W
            best_b = b

    # ---------- Refit best method on full data ----------
    if best_method == "lda":
        alpha = best_param
        means = np.zeros((C, D), dtype=np.float64)
        for c in range(C):
            mask = (y == c)
            if np.sum(mask) > 0:
                means[c] = X[mask].mean(axis=0)
        global_mean = X.mean(axis=0)
        Sw = np.zeros((D, D), dtype=np.float64)
        for c in range(C):
            mask = (y == c)
            if np.sum(mask) > 0:
                Xc = X[mask]
                diff = Xc - means[c]
                Sw += diff.T @ diff
        Sw_reg = (1.0 - alpha) * Sw + alpha * (np.trace(Sw) / D) * np.eye(D)
        centered_means = means - global_mean
        W = np.linalg.solve(Sw_reg, centered_means.T)  # (D, C)
        b = -0.5 * np.sum(centered_means * W.T, axis=1)
        priors = np.array([np.mean(y == c) for c in range(C)])
        b += np.log(priors + 1e-12)
        W = W.T  # (C, D)
    else:  # ridge
        alpha = best_param
        A = Xb.T @ Xb + alpha * np.eye(Db)
        B = Xb.T @ Y
        W_full = np.linalg.solve(A, B)
        W = W_full[:D, :].T
        b = W_full[D, :]

    # ---------- Write into w ----------
    w["fc2.weight"] = W.astype(np.float32)
    w["fc2.bias"] = b.astype(np.float32)

    # Diagnostics
    logits_full = X @ W.T + b
    pred_full = np.argmax(logits_full, axis=1)
    full_acc = float(np.mean(pred_full == y))
    # Cross-entropy
    probs = np.exp(logits_full - logits_full.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    ce = float(-np.mean(np.log(probs[np.arange(N), y] + 1e-12)))

    diagnostics = {
        "method": best_method,
        "param": float(best_param),
        "cv_accuracy": float(best_acc),
        "full_accuracy": full_acc,
        "full_cross_entropy": ce,
        "fc2_weight_mean": float(W.mean()),
        "fc2_weight_std": float(W.std()),
        "fc2_bias_mean": float(b.mean()),
        "fc2_bias_std": float(b.std()),
    }

    return {"weights": w, "diagnostics": diagnostics}
