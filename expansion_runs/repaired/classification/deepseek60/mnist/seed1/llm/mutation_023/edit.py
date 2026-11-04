import numpy as np

def edit(w, f):
    # Extract data
    X = f["x"].reshape(2000, 784).astype(np.float32)  # already in [0,1]
    y = f["y"].astype(np.int64)
    N = X.shape[0]
    H_cur = f["features"].astype(np.float32)  # current fc1 outputs (2000,128)

    # Current weights
    W1_cur = w["fc1.weight"].astype(np.float32)  # (128,784)
    b1_cur = w["fc1.bias"].astype(np.float32)    # (128,)
    W2_cur = w["fc2.weight"].astype(np.float32)  # (10,128)
    b2_cur = w["fc2.bias"].astype(np.float32)    # (10,)

    # One-hot targets
    Y = np.zeros((N, 10), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    # Helper: ridge regression fit
    def ridge_fit(H, Y, lam=1e-3):
        # H: (N, D), Y: (N, 10)
        # Add bias column
        H_aug = np.concatenate([H, np.ones((N, 1), dtype=np.float32)], axis=1)
        D = H_aug.shape[1]
        # Solve (H^T H + lam I) W = H^T Y
        A = H_aug.T @ H_aug + lam * np.eye(D, dtype=np.float32)
        B = H_aug.T @ Y
        W_aug = np.linalg.solve(A, B)  # (D, 10)
        W = W_aug[:-1].T  # (10, D-1)
        b = W_aug[-1]     # (10,)
        return W, b

    # Helper: evaluate accuracy
    def accuracy(H, y, W, b):
        logits = H @ W.T + b
        pred = np.argmax(logits, axis=1)
        return np.mean(pred == y)

    # Split for validation
    rng = np.random.RandomState(42)
    idx = rng.permutation(N)
    n_train = 1600
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    H_cur_train, H_cur_val = H_cur[train_idx], H_cur[val_idx]

    # Compute linear classifier on raw pixels using ridge
    # Use a small lambda, can tune later
    lam_lin = 1e-2
    W_lin, b_lin = ridge_fit(X_train, Y[train_idx], lam=lam_lin)  # (10,784), (10,)

    # Compute PCA on raw pixels (train only)
    # Center X_train
    X_mean = X_train.mean(axis=0, keepdims=True)
    Xc = X_train - X_mean
    # SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # Vt: (min(N,D), 784). Take top 128 components
    n_pca = min(128, Vt.shape[0])
    W_pca = Vt[:n_pca]  # (n_pca, 784)
    b_pca = - (X_mean @ W_pca.T).flatten()  # (n_pca,) to center

    # Random projections
    rng_rand = np.random.RandomState(123)
    W_rand = rng_rand.randn(128, 784).astype(np.float32) * 0.05
    b_rand = rng_rand.uniform(-1, 1, size=128).astype(np.float32)

    # We will try several configurations:
    # Each config: (n_existing, n_lin_pos_neg, n_pca, n_rand)
    # n_lin_pos_neg is number of units for linear classifier (should be 0, 10, or 20)
    # We'll allocate the rest to existing, pca, rand as specified.
    configs = []
    # 1. All existing
    configs.append(("all_existing", 128, 0, 0, 0))
    # 2. 108 existing + 20 linear (pos/neg)
    configs.append(("108_exist_20_lin", 108, 20, 0, 0))
    # 3. 118 existing + 10 linear (with large bias)
    configs.append(("118_exist_10_lin", 118, 10, 0, 0))
    # 4. All PCA
    configs.append(("all_pca", 0, 0, 128, 0))
    # 5. All random
    configs.append(("all_rand", 0, 0, 0, 128))
    # 6. 64 existing + 64 PCA
    configs.append(("64_exist_64_pca", 64, 0, 64, 0))
    # 7. 64 existing + 64 random
    configs.append(("64_exist_64_rand", 64, 0, 0, 64))
    # 8. 108 existing + 10 PCA + 10 linear
    configs.append(("108_exist_10_pca_10_lin", 108, 10, 10, 0))
    # 9. 88 existing + 20 linear + 20 PCA
    configs.append(("88_exist_20_lin_20_pca", 88, 20, 20, 0))

    best_acc = -1
    best_config = None
    best_H_train = None
    best_H_val = None
    best_W2 = None
    best_b2 = None

    for name, n_exist, n_lin, n_pca, n_rand in configs:
        total = n_exist + n_lin + n_pca + n_rand
        if total > 128:
            continue
        # Build feature matrix for train and val
        H_train_parts = []
        H_val_parts = []
        # Existing
        if n_exist > 0:
            H_train_parts.append(H_cur_train[:, :n_exist])
            H_val_parts.append(H_cur_val[:, :n_exist])
        # Linear classifier units
        if n_lin > 0:
            # Compute linear scores on train and val
            S_train = X_train @ W_lin.T + b_lin  # (N_train, 10)
            S_val = X_val @ W_lin.T + b_lin
            if n_lin == 20:
                # pos and neg parts
                H_lin_train = np.concatenate([np.maximum(S_train, 0), np.maximum(-S_train, 0)], axis=1)  # (N,20)
                H_lin_val = np.concatenate([np.maximum(S_val, 0), np.maximum(-S_val, 0)], axis=1)
            elif n_lin == 10:
                # Use large bias to avoid clipping
                M = 100.0
                H_lin_train = S_train + M
                H_lin_val = S_val + M
            else:
                # If n_lin is not 10 or 20, just use first n_lin classes with pos/neg? Not needed.
                H_lin_train = np.maximum(S_train[:, :n_lin], 0)
                H_lin_val = np.maximum(S_val[:, :n_lin], 0)
            H_train_parts.append(H_lin_train)
            H_val_parts.append(H_lin_val)
        # PCA
        if n_pca > 0:
            # Project X_train and X_val using W_pca and b_pca
            H_pca_train = np.maximum(X_train @ W_pca[:n_pca].T + b_pca[:n_pca], 0)
            H_pca_val = np.maximum(X_val @ W_pca[:n_pca].T + b_pca[:n_pca], 0)
            H_train_parts.append(H_pca_train)
            H_val_parts.append(H_pca_val)
        # Random
        if n_rand > 0:
            H_rand_train = np.maximum(X_train @ W_rand[:n_rand].T + b_rand[:n_rand], 0)
            H_rand_val = np.maximum(X_val @ W_rand[:n_rand].T + b_rand[:n_rand], 0)
            H_train_parts.append(H_rand_train)
            H_val_parts.append(H_rand_val)
        # Concatenate
        if len(H_train_parts) == 0:
            continue
        H_train = np.concatenate(H_train_parts, axis=1)
        H_val = np.concatenate(H_val_parts, axis=1)
        # Pad with zeros if less than 128? Not needed, we can just use the available features.
        # Fit fc2 on train
        W2, b2 = ridge_fit(H_train, Y[train_idx], lam=1e-3)
        acc = accuracy(H_val, y_val, W2, b2)
        if acc > best_acc:
            best_acc = acc
            best_config = (name, n_exist, n_lin, n_pca, n_rand)
            best_H_train = H_train
            best_H_val = H_val
            best_W2 = W2
            best_b2 = b2

    # Now build final W1, b1, W2, b2 using best_config on full data
    name, n_exist, n_lin, n_pca, n_rand = best_config
    # We need to construct W1 (128,784) and b1 (128,)
    # We'll fill the first n_exist rows with current fc1 weights, then n_lin rows with linear, etc.
    # But we need to ensure the total is 128. We can pad remaining with zeros or random? Better to use all 128.
    # We'll allocate exactly as in best_config, and if total < 128, we can fill the rest with existing features (or zeros).
    # Let's just use the best_config and pad with existing features if needed.
    total = n_exist + n_lin + n_pca + n_rand
    if total < 128:
        # Add more existing features
        n_exist += 128 - total
        total = 128

    W1_new = np.zeros((128, 784), dtype=np.float32)
    b1_new = np.zeros(128, dtype=np.float32)

    idx = 0
    # Existing
    if n_exist > 0:
        W1_new[idx:idx+n_exist] = W1_cur[:n_exist]
        b1_new[idx:idx+n_exist] = b1_cur[:n_exist]
        idx += n_exist
    # Linear
    if n_lin > 0:
        if n_lin == 20:
            # pos and neg parts: for each class, one unit for positive score, one for negative
            # W1 rows: W_lin[c] for pos, -W_lin[c] for neg
            # b1 rows: b_lin[c] for pos, -b_lin[c] for neg
            for c in range(10):
                W1_new[idx] = W_lin[c]
                b1_new[idx] = b_lin[c]
                idx += 1
                W1_new[idx] = -W_lin[c]
                b1_new[idx] = -b_lin[c]
                idx += 1
        elif n_lin == 10:
            M = 100.0
            for c in range(10):
                W1_new[idx] = W_lin[c]
                b1_new[idx] = b_lin[c] + M
                idx += 1
        else:
            # fallback: just use first n_lin classes with pos
            for c in range(n_lin):
                W1_new[idx] = W_lin[c]
                b1_new[idx] = b_lin[c]
                idx += 1
    # PCA
    if n_pca > 0:
        W1_new[idx:idx+n_pca] = W_pca[:n_pca]
        b1_new[idx:idx+n_pca] = b_pca[:n_pca]
        idx += n_pca
    # Random
    if n_rand > 0:
        W1_new[idx:idx+n_rand] = W_rand[:n_rand]
        b1_new[idx:idx+n_rand] = b_rand[:n_rand]
        idx += n_rand

    # Now compute new hidden features for full data
    H_new = np.maximum(X @ W1_new.T + b1_new, 0)  # (2000,128)

    # Fit fc2 on full data
    W2_new, b2_new = ridge_fit(H_new, Y, lam=1e-3)

    # Update weights
    w["fc1.weight"] = W1_new.astype(np.float32)
    w["fc1.bias"] = b1_new.astype(np.float32)
    w["fc2.weight"] = W2_new.astype(np.float32)
    w["fc2.bias"] = b2_new.astype(np.float32)

    # Diagnostics
    # Compute accuracy on full feedback
    logits = H_new @ W2_new.T + b2_new
    pred = np.argmax(logits, axis=1)
    acc_full = np.mean(pred == y)
    # Cross-entropy
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    ce = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))

    diagnostics = {
        "best_config": name,
        "n_exist": int(n_exist),
        "n_lin": int(n_lin),
        "n_pca": int(n_pca),
        "n_rand": int(n_rand),
        "val_acc": float(best_acc),
        "full_acc": float(acc_full),
        "full_ce": float(ce),
        "fc1_weight_mean": float(W1_new.mean()),
        "fc1_weight_std": float(W1_new.std()),
        "fc2_weight_mean": float(W2_new.mean()),
        "fc2_weight_std": float(W2_new.std()),
        "fc2_bias_mean": float(b2_new.mean()),
        "fc2_bias_std": float(b2_new.std()),
    }

    return {"weights": w, "diagnostics": diagnostics}
