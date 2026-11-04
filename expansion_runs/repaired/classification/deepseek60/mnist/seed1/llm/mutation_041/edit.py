import numpy as np

def edit(w, f):
    # Flatten images
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)
    y = f["y"].astype(np.int64)
    N = X.shape[0]
    C = 10
    D_in = X.shape[1]

    # Class means and global mean
    means = np.zeros((C, D_in))
    counts = np.zeros(C, dtype=int)
    for c in range(C):
        mask = (y == c)
        counts[c] = mask.sum()
        if counts[c] > 0:
            means[c] = X[mask].mean(0)
    mu = X.mean(0)

    # Build candidate directions
    candidates = []
    # 1. Current fc1 rows
    candidates.append(w["fc1.weight"].astype(np.float64))
    # 2. Class means
    candidates.append(means)
    # 3. Pairwise class-mean differences
    for i in range(C):
        for j in range(i + 1, C):
            candidates.append((means[i] - means[j]).reshape(1, -1))
    # 4. PCA directions (top 128)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    candidates.append(Vt[:128])
    # 5. LDA directions (up to 9)
    Sw = np.zeros((D_in, D_in))
    Sb = np.zeros((D_in, D_in))
    for c in range(C):
        Xc_c = X[y == c] - means[c]
        Sw += Xc_c.T @ Xc_c
        diff = (means[c] - mu).reshape(-1, 1)
        Sb += counts[c] * (diff @ diff.T)
    Sw_reg = Sw + 1e-3 * np.eye(D_in)
    try:
        M = np.linalg.solve(Sw_reg, Sb)
        eigvals, eigvecs = np.linalg.eig(M)
        idx = np.argsort(-np.real(eigvals))[:9]
        lda_dirs = np.real(eigvecs[:, idx]).T
        candidates.append(lda_dirs)
    except Exception:
        pass
    # 6. Random projections
    rng = np.random.RandomState(42)
    candidates.append(rng.randn(1000, D_in))

    D = np.vstack(candidates)
    # Normalize each direction to unit norm
    norms = np.linalg.norm(D, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    D = D / norms

    # Compute ANOVA F-values for all candidates
    P = X @ D.T  # (N, num_candidates)
    P_means = np.zeros((C, P.shape[1]))
    for c in range(C):
        P_means[c] = P[y == c].mean(0)
    P_global_mean = P.mean(0)
    between = np.zeros(P.shape[1])
    for c in range(C):
        diff = P_means[c] - P_global_mean
        between += counts[c] * (diff ** 2)
    total = ((P - P_global_mean) ** 2).sum(0)
    within = total - between
    within[within < 1e-12] = 1e-12
    F = (between / (C - 1)) / (within / (N - C))

    # Select top 128 directions
    top_idx = np.argsort(-F)[:128]
    W1 = D[top_idx]  # (128, D_in)

    # Projections for bias selection
    proj = X @ W1.T  # (N, 128)

    # Bias strategies
    bias_strategies = {
        'zero': np.zeros(128),
        'mean': -proj.mean(0),
        'median': -np.median(proj, axis=0),
        'q25': -np.percentile(proj, 25, axis=0),
        'q75': -np.percentile(proj, 75, axis=0),
    }

    # 5-fold CV
    n_folds = 5
    fold_size = N // n_folds
    indices = np.arange(N)
    rng_cv = np.random.RandomState(123)
    rng_cv.shuffle(indices)
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = N if i == n_folds - 1 else (i + 1) * fold_size
        val_idx = indices[start:end]
        train_idx = np.setdiff1d(indices, val_idx)
        folds.append((train_idx, val_idx))

    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    best_acc = -1.0
    best_config = None

    for bias_name, b1 in bias_strategies.items():
        H = np.maximum(proj + b1, 0)  # (N, 128)
        for lam in lambdas:
            accs = []
            for train_idx, val_idx in folds:
                H_train = H[train_idx]
                y_train = y[train_idx]
                H_val = H[val_idx]
                y_val = y[val_idx]

                # Augment with bias column
                H_train_aug = np.hstack([H_train, np.ones((len(train_idx), 1))])
                H_val_aug = np.hstack([H_val, np.ones((len(val_idx), 1))])

                # One-hot targets
                Y_train = np.zeros((len(train_idx), C))
                Y_train[np.arange(len(train_idx)), y_train] = 1.0

                # Ridge regression (do not regularize bias)
                reg = lam * np.eye(H_train_aug.shape[1])
                reg[-1, -1] = 0.0
                W_aug = np.linalg.solve(H_train_aug.T @ H_train_aug + reg,
                                        H_train_aug.T @ Y_train)

                # Predict
                logits_val = H_val_aug @ W_aug
                preds = np.argmax(logits_val, axis=1)
                acc = (preds == y_val).mean()
                accs.append(acc)
            mean_acc = np.mean(accs)
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_config = (bias_name, b1, lam)

    # Final fit on all data with best config
    bias_name, b1, lam = best_config
    H = np.maximum(proj + b1, 0)
    H_aug = np.hstack([H, np.ones((N, 1))])
    Y = np.zeros((N, C))
    Y[np.arange(N), y] = 1.0
    reg = lam * np.eye(H_aug.shape[1])
    reg[-1, -1] = 0.0
    W_aug = np.linalg.solve(H_aug.T @ H_aug + reg, H_aug.T @ Y)

    W2 = W_aug[:128, :].T  # (10, 128)
    b2 = W_aug[128, :]     # (10,)

    # Clip to safe range
    W1 = np.clip(W1, -1000, 1000).astype(np.float32)
    b1 = np.clip(b1, -1000, 1000).astype(np.float32)
    W2 = np.clip(W2, -1000, 1000).astype(np.float32)
    b2 = np.clip(b2, -1000, 1000).astype(np.float32)

    w["fc1.weight"] = W1
    w["fc1.bias"] = b1
    w["fc2.weight"] = W2
    w["fc2.bias"] = b2

    diagnostics = {
        "best_bias": bias_name,
        "best_lambda": float(lam),
        "cv_accuracy": float(best_acc),
        "num_candidates": int(D.shape[0]),
        "selected_F_mean": float(F[top_idx].mean()),
        "fc1_weight_std": float(W1.std()),
        "fc2_weight_std": float(W2.std()),
        "fc2_bias_mean": float(b2.mean()),
    }
    return {"weights": w, "diagnostics": diagnostics}
