import numpy as np

def edit(w, f):
    # Extract data
    X = f["x"].reshape(2000, -1).astype(np.float32)  # 2000x784
    y = f["y"].astype(np.int64)                       # 2000
    N = X.shape[0]
    C = 10

    # Current weights
    W1_cur = w["fc1.weight"].copy()  # 128x784
    b1_cur = w["fc1.bias"].copy()    # 128
    W2_cur = w["fc2.weight"].copy()  # 10x128
    b2_cur = w["fc2.bias"].copy()    # 10

    # ---------- Generate candidate directions ----------
    candidates = []

    # 1. Current fc1 rows
    candidates.append(W1_cur)

    # 2. Class means
    class_means = np.zeros((C, 784), dtype=np.float32)
    for c in range(C):
        class_means[c] = X[y == c].mean(axis=0)
    candidates.append(class_means)

    # 3. Pairwise class mean differences
    pairwise = []
    for i in range(C):
        for j in range(i + 1, C):
            pairwise.append(class_means[i] - class_means[j])
    candidates.append(np.array(pairwise, dtype=np.float32))

    # 4. PCA directions (top 128)
    X_centered = X - X.mean(axis=0, keepdims=True)
    # Use SVD on centered data
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    pca_dirs = Vt[:128].astype(np.float32)
    candidates.append(pca_dirs)

    # 5. Ridge regression directions for several lambdas
    Y_onehot = np.zeros((N, C), dtype=np.float32)
    Y_onehot[np.arange(N), y] = 1.0
    X_aug = np.hstack([X, np.ones((N, 1), dtype=np.float32)])  # 2000x785
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    for lam in lambdas:
        A = X_aug.T @ X_aug + lam * np.eye(785, dtype=np.float32)
        B = X_aug.T @ Y_onehot
        W = np.linalg.solve(A, B)  # 785x10
        ridge_dirs = W[:784].T     # 10x784
        candidates.append(ridge_dirs)

    # 6. Random projections
    rng = np.random.RandomState(42)
    num_random = 200
    random_dirs = rng.randn(num_random, 784).astype(np.float32)
    random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)
    candidates.append(random_dirs)

    # 7. LDA directions
    mu = X.mean(axis=0)
    Sw = np.zeros((784, 784), dtype=np.float32)
    Sb = np.zeros((784, 784), dtype=np.float32)
    for c in range(C):
        Xc = X[y == c]
        mu_c = Xc.mean(axis=0)
        diff = mu_c - mu
        Sb += len(Xc) * np.outer(diff, diff)
        Xc_centered = Xc - mu_c
        Sw += Xc_centered.T @ Xc_centered
    Sw += 1e-3 * np.eye(784, dtype=np.float32)
    M = np.zeros((784, C), dtype=np.float32)
    for c in range(C):
        n_c = np.sum(y == c)
        M[:, c] = np.sqrt(n_c) * (class_means[c] - mu)
    Sw_inv_M = np.linalg.solve(Sw, M)
    A_lda = M.T @ Sw_inv_M
    eigvals, eigvecs = np.linalg.eigh(A_lda)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    lda_dirs = (Sw_inv_M @ eigvecs)[:, :9].T.astype(np.float32)  # 9x784
    candidates.append(lda_dirs)

    all_candidates = np.vstack(candidates)  # (total, 784)
    num_cand = all_candidates.shape[0]

    # ---------- Compute ANOVA F-value for each candidate ----------
    Z = X @ all_candidates.T  # 2000 x num_cand
    mu_z = Z.mean(axis=0)
    between = np.zeros(num_cand, dtype=np.float32)
    within = np.zeros(num_cand, dtype=np.float32)
    for c in range(C):
        mask = (y == c)
        Z_c = Z[mask]
        mu_c = Z_c.mean(axis=0)
        between += mask.sum() * (mu_c - mu_z) ** 2
        within += ((Z_c - mu_c) ** 2).sum(axis=0)
    F = (between / (C - 1)) / (within / (N - C) + 1e-12)

    # Select top 128
    top_idx = np.argsort(F)[::-1][:128]
    W1 = all_candidates[top_idx].copy()  # 128x784

    # Set bias to center the projections (median)
    proj = X @ W1.T  # 2000x128
    b1 = -np.median(proj, axis=0).astype(np.float32)

    # Compute hidden features
    H = np.maximum(proj + b1, 0.0).astype(np.float32)  # 2000x128

    # ---------- Fit fc2 via ridge regression with 5-fold CV ----------
    # Augment H with ones
    H_aug = np.hstack([H, np.ones((N, 1), dtype=np.float32)])  # 2000x129

    # 5-fold CV
    fold_size = N // 5
    indices = np.arange(N)
    rng_cv = np.random.RandomState(123)
    rng_cv.shuffle(indices)
    lambdas_cv = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    best_lambda = 1e-2
    best_score = -np.inf

    for lam in lambdas_cv:
        accs = []
        ces = []
        for fold in range(5):
            val_idx = indices[fold * fold_size:(fold + 1) * fold_size]
            train_idx = np.setdiff1d(indices, val_idx)
            H_train = H_aug[train_idx]
            H_val = H_aug[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]
            Y_train = np.zeros((len(train_idx), C), dtype=np.float32)
            Y_train[np.arange(len(train_idx)), y_train] = 1.0
            A = H_train.T @ H_train + lam * np.eye(129, dtype=np.float32)
            B = H_train.T @ Y_train
            W_aug = np.linalg.solve(A, B)  # 129x10
            logits_val = H_val @ W_aug
            # Accuracy
            preds = np.argmax(logits_val, axis=1)
            acc = np.mean(preds == y_val)
            # Cross-entropy
            logits_val -= logits_val.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits_val)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            ce = -np.mean(np.log(probs[np.arange(len(y_val)), y_val] + 1e-12))
            accs.append(acc)
            ces.append(ce)
        mean_acc = np.mean(accs)
        mean_ce = np.mean(ces)
        # Score: prioritize accuracy, but penalize high CE slightly
        score = mean_acc - 0.05 * mean_ce
        if score > best_score:
            best_score = score
            best_lambda = lam

    # Fit on full data with best lambda
    A_full = H_aug.T @ H_aug + best_lambda * np.eye(129, dtype=np.float32)
    B_full = H_aug.T @ Y_onehot
    W_aug_full = np.linalg.solve(A_full, B_full)  # 129x10
    W2 = W_aug_full[:128].T.astype(np.float32)    # 10x128
    b2 = W_aug_full[128].astype(np.float32)       # 10

    # ---------- Temperature calibration ----------
    logits = H @ W2.T + b2  # 2000x10
    # Grid search over scale s
    best_s = 1.0
    best_ce = np.inf
    for s in np.linspace(0.1, 5.0, 100):
        scaled = s * logits
        scaled -= scaled.max(axis=1, keepdims=True)
        exp_scaled = np.exp(scaled)
        probs = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)
        ce = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))
        if ce < best_ce:
            best_ce = ce
            best_s = s

    # Apply scale
    W2 *= best_s
    b2 *= best_s

    # ---------- Update weights ----------
    w["fc1.weight"] = W1.astype(np.float32)
    w["fc1.bias"] = b1.astype(np.float32)
    w["fc2.weight"] = W2.astype(np.float32)
    w["fc2.bias"] = b2.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "num_candidates": int(num_cand),
        "selected_indices": top_idx[:10].tolist(),
        "best_lambda": float(best_lambda),
        "best_scale": float(best_s),
        "train_accuracy": float(np.mean(np.argmax(logits * best_s, axis=1) == y)),
        "train_cross_entropy": float(best_ce),
        "fc1_weight_mean": float(W1.mean()),
        "fc1_weight_std": float(W1.std()),
        "fc2_weight_mean": float(W2.mean()),
        "fc2_weight_std": float(W2.std()),
        "fc2_bias_mean": float(b2.mean()),
        "fc2_bias_std": float(b2.std()),
    }

    return {"weights": w, "diagnostics": diagnostics}
