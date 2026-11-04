import numpy as np

def edit(w, f):
    # Extract training data
    X = f["x"].reshape(-1, 784).astype(np.float32)
    y = f["y"].astype(np.int64)
    N = X.shape[0]
    classes = np.unique(y)
    n_classes = len(classes)

    # Global mean
    mu = X.mean(axis=0)

    # Class means and counts
    mu_c = np.zeros((n_classes, 784), dtype=np.float32)
    counts = np.zeros(n_classes, dtype=np.float32)
    for i, c in enumerate(classes):
        mask = (y == c)
        counts[i] = mask.sum()
        mu_c[i] = X[mask].mean(axis=0)

    # Centered data and class-mean differences
    Xc = X - mu
    D = mu_c - mu  # (10, 784)

    # Scatter matrices
    St = Xc.T @ Xc
    Sb = D.T @ (counts[:, None] * D)
    Sw = St - Sb

    # Ridge stabilize Sw
    Sw += 1e-3 * np.eye(784, dtype=np.float32) * (np.trace(Sw) / 784.0)

    # LDA directions: solve Sw^{-1} D
    # Use solve for stability
    M = np.linalg.solve(Sw, D.T)  # (784, 10)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    # Take up to 9 LDA directions (rank at most n_classes-1)
    n_lda = min(n_classes - 1, U.shape[1])
    lda_dirs = U[:, :n_lda]  # (784, n_lda)

    # PCA on centered data
    # We need enough components to fill remaining units
    # Total hidden units = 128. LDA gives n_lda directions, we use both signs -> 2*n_lda units.
    # Remaining units = 128 - 2*n_lda. Each PCA component gives 2 units (pos/neg).
    n_pca = (128 - 2 * n_lda) // 2
    # Compute top n_pca PCA components
    # Use SVD on Xc (N x 784). We only need first n_pca right singular vectors.
    # To save memory/time, compute covariance and eigen-decompose? N=2000, 784 features.
    # SVD on Xc is fine.
    U_pca, S_pca, Vt_pca = np.linalg.svd(Xc, full_matrices=False)
    pca_dirs = Vt_pca[:n_pca, :]  # (n_pca, 784)

    # Build fc1 weights and biases
    W1 = np.zeros((128, 784), dtype=np.float32)
    b1 = np.zeros(128, dtype=np.float32)
    idx = 0

    # LDA units (positive and negative)
    for i in range(n_lda):
        v = lda_dirs[:, i]
        # positive
        W1[idx] = v
        b1[idx] = -np.dot(v, mu)
        idx += 1
        # negative
        W1[idx] = -v
        b1[idx] = np.dot(v, mu)
        idx += 1

    # PCA units (positive and negative)
    for i in range(n_pca):
        v = pca_dirs[i]
        W1[idx] = v
        b1[idx] = -np.dot(v, mu)
        idx += 1
        W1[idx] = -v
        b1[idx] = np.dot(v, mu)
        idx += 1

    # If we have leftover units (due to odd numbers), fill with zeros or random small
    while idx < 128:
        W1[idx] = 0.0
        b1[idx] = 0.0
        idx += 1

    # Compute new hidden features for all data
    H = np.maximum(0, X @ W1.T + b1)  # (N, 128)

    # Prepare one-hot targets
    Y = np.zeros((N, n_classes), dtype=np.float32)
    for i, c in enumerate(classes):
        Y[y == c, i] = 1.0

    # Validation split for choosing ridge alpha
    # Use a simple holdout: first 1600 train, last 400 val
    # But we can also just use all data for final fit. To choose alpha, do a quick holdout.
    n_train = int(0.8 * N)
    # Shuffle? Data may be ordered. Let's shuffle deterministically.
    rng = np.random.RandomState(42)
    perm = rng.permutation(N)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    H_train = H[train_idx]
    Y_train = Y[train_idx]
    H_val = H[val_idx]
    Y_val = Y[val_idx]

    # Augment with bias
    H_train_aug = np.hstack([H_train, np.ones((H_train.shape[0], 1), dtype=np.float32)])
    H_val_aug = np.hstack([H_val, np.ones((H_val.shape[0], 1), dtype=np.float32)])

    best_alpha = 1e-2
    best_acc = -1.0
    for alpha in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        # Solve ridge: (H^T H + alpha I) W = H^T Y
        A = H_train_aug.T @ H_train_aug + alpha * np.eye(H_train_aug.shape[1], dtype=np.float32)
        B = H_train_aug.T @ Y_train
        W_aug = np.linalg.solve(A, B)  # (129, 10)
        # Predict on val
        logits_val = H_val_aug @ W_aug
        preds = np.argmax(logits_val, axis=1)
        acc = np.mean(preds == y[val_idx])
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha

    # Final fit on all data with best alpha
    H_aug = np.hstack([H, np.ones((N, 1), dtype=np.float32)])
    A = H_aug.T @ H_aug + best_alpha * np.eye(H_aug.shape[1], dtype=np.float32)
    B = H_aug.T @ Y
    W_aug = np.linalg.solve(A, B)  # (129, 10)

    W2 = W_aug[:128, :].T  # (10, 128)
    b2 = W_aug[128, :]     # (10,)

    # Update weights
    w["fc1.weight"] = W1.astype(np.float32)
    w["fc1.bias"] = b1.astype(np.float32)
    w["fc2.weight"] = W2.astype(np.float32)
    w["fc2.bias"] = b2.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "method": "LDA+PCA features, ridge fc2",
        "best_alpha": float(best_alpha),
        "val_acc": float(best_acc),
        "n_lda": int(n_lda),
        "n_pca": int(n_pca),
        "fc1_weight_mean": float(W1.mean()),
        "fc1_weight_std": float(W1.std()),
        "fc2_weight_mean": float(W2.mean()),
        "fc2_weight_std": float(W2.std()),
    }

    return {"weights": w, "diagnostics": diagnostics}
