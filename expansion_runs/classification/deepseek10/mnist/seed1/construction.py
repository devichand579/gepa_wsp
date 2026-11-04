import numpy as np

def edit(w, f):
    # Extract full training pool
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)  # (2000, 784)
    y = f["y"].astype(np.int64)
    N = X.shape[0]
    Y = np.eye(10, dtype=np.float64)[y]

    # Global mean and PCA
    mean = X.mean(axis=0)
    Xc = X - mean
    # SVD of centered data: top 128 principal components
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V_pca = Vt[:128]  # (128, 784)

    # Class means
    class_means = np.zeros((10, 784), dtype=np.float64)
    for c in range(10):
        class_means[c] = X[y == c].mean(axis=0)

    # Build candidate feature maps (W1, b1)
    candidates = []

    # 1. Pure PCA features
    candidates.append(("PCA", V_pca.copy(), -V_pca @ mean))

    # 2. Class means + top 118 PCA components
    W1 = np.vstack([class_means, V_pca[:118]])
    b1 = np.concatenate([-class_means @ mean, -V_pca[:118] @ mean])
    candidates.append(("CM+PCA", W1, b1))

    # 3. Random ReLU features (He initialization)
    for seed in [0, 1, 2]:
        rng = np.random.RandomState(seed)
        W1 = rng.randn(128, 784) * np.sqrt(2.0 / 784)
        b1 = rng.randn(128) * 0.1
        candidates.append((f"Rand{seed}", W1, b1))

    # 4. Random ReLU features with larger scale
    for seed in [3, 4]:
        rng = np.random.RandomState(seed)
        W1 = rng.randn(128, 784) * 0.1
        b1 = rng.randn(128) * 0.1
        candidates.append((f"RandLarge{seed}", W1, b1))

    # Validation split for hyperparameter selection
    np.random.seed(42)
    indices = np.random.permutation(N)
    train_idx = indices[:1600]
    val_idx = indices[1600:]

    X_train, y_train = X[train_idx], y[train_idx]
    Y_train = Y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    def compute_H(X, W1, b1):
        return np.maximum(0.0, X @ W1.T + b1)

    def solve_ridge(H, Y, lam):
        H_aug = np.hstack([H, np.ones((H.shape[0], 1), dtype=np.float64)])
        A = H_aug.T @ H_aug + lam * np.eye(H_aug.shape[1], dtype=np.float64)
        B = H_aug.T @ Y
        W_aug = np.linalg.solve(A, B)
        W2 = W_aug[:-1].T
        b2 = W_aug[-1]
        return W2, b2

    def evaluate(W1, b1, W2, b2, X, y):
        H = compute_H(X, W1, b1)
        logits = H @ W2.T + b2
        pred = np.argmax(logits, axis=1)
        return np.mean(pred == y)

    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    best_val = -1.0
    best_config = None

    for name, W1, b1 in candidates:
        H_train = compute_H(X_train, W1, b1)
        for lam in lambdas:
            W2, b2 = solve_ridge(H_train, Y_train, lam)
            val_acc = evaluate(W1, b1, W2, b2, X_val, y_val)
            if val_acc > best_val:
                best_val = val_acc
                best_config = (name, W1, b1, lam)

    # Retrain on full training pool with the best configuration
    name, W1_best, b1_best, lam_best = best_config
    H_full = compute_H(X, W1_best, b1_best)
    W2_best, b2_best = solve_ridge(H_full, Y, lam_best)

    # Write back to w (float32)
    w["fc1.weight"] = W1_best.astype(np.float32)
    w["fc1.bias"]   = b1_best.astype(np.float32)
    w["fc2.weight"] = W2_best.astype(np.float32)
    w["fc2.bias"]   = b2_best.astype(np.float32)

    # Diagnostics
    train_acc = evaluate(W1_best, b1_best, W2_best, b2_best, X, y)
    diagnostics = {
        "best_feature": name,
        "best_lambda": lam_best,
        "val_acc": float(best_val),
        "train_acc": float(train_acc),
        "fc1_weight_norm": float(np.linalg.norm(w["fc1.weight"])),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
    }

    return {"weights": w, "diagnostics": diagnostics}
