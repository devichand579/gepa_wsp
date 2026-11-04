import numpy as np

def edit(w, f):
    # Extract full training pool
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)  # (2000, 784)
    y = f["y"].astype(int)                                      # (2000,)
    N = X.shape[0]
    classes = np.unique(y)
    C = len(classes)  # 10

    # Global mean
    mu = X.mean(axis=0)

    # Class means and differences
    D = []
    for c in classes:
        mu_c = X[y == c].mean(axis=0)
        D.append(mu_c - mu)
    D = np.array(D)  # (10, 784)
    # Normalize each class-mean direction to unit norm
    D_norm = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)

    # PCA on centered data
    Xc = X - mu
    # Use SVD; we only need the right singular vectors
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    n_pca = 128 - C  # 118
    V = Vt[:n_pca]   # (118, 784)

    # Combine class-mean directions and PCA directions
    W1 = np.vstack([D_norm, V])  # (128, 784)
    # Normalize all rows to unit norm for stable activations
    W1 = W1 / (np.linalg.norm(W1, axis=1, keepdims=True) + 1e-12)
    b1 = -W1 @ mu

    # Compute hidden features
    H = np.maximum(0.0, X @ W1.T + b1)  # (2000, 128)

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Augment hidden features with bias term
    H_aug = np.hstack([H, np.ones((N, 1))])  # (2000, 129)

    # Choose ridge lambda via 5-fold cross-validation on the pool
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    best_lam = 1e-3
    best_acc = -1.0

    # Manual 5-fold split
    rng = np.random.RandomState(0)
    indices = np.arange(N)
    rng.shuffle(indices)
    folds = np.array_split(indices, 5)

    for lam in lambdas:
        accs = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            H_tr = H_aug[train_idx]
            H_va = H_aug[val_idx]
            Y_tr = Y[train_idx]
            Y_va = Y[val_idx]

            A = H_tr.T @ H_tr + lam * np.eye(H_tr.shape[1])
            B = H_tr.T @ Y_tr
            W_aug = np.linalg.solve(A, B)  # (129, 10)

            logits_va = H_va @ W_aug
            pred_va = np.argmax(logits_va, axis=1)
            true_va = np.argmax(Y_va, axis=1)
            accs.append(np.mean(pred_va == true_va))
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lam = lam

    # Final fit on all data with best lambda
    lam = best_lam
    A = H_aug.T @ H_aug + lam * np.eye(H_aug.shape[1])
    B = H_aug.T @ Y
    W_aug = np.linalg.solve(A, B)  # (129, 10)

    W2 = W_aug[:128].T  # (10, 128)
    b2 = W_aug[128]     # (10,)

    # Write back to w (preserve float32 and shapes)
    w["fc1.weight"] = W1.astype(np.float32)
    w["fc1.bias"]   = b1.astype(np.float32)
    w["fc2.weight"] = W2.astype(np.float32)
    w["fc2.bias"]   = b2.astype(np.float32)

    # Diagnostics
    logits_full = H @ W2.T + b2
    pred_full = np.argmax(logits_full, axis=1)
    train_acc = float(np.mean(pred_full == y))
    diag = {
        "train_accuracy": train_acc,
        "cv_accuracy": float(best_acc),
        "chosen_lambda": float(lam),
        "n_pca": int(n_pca),
        "n_class_mean": int(C),
        "fc1_weight_norm": float(np.linalg.norm(w["fc1.weight"])),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
    }
    return {"weights": w, "diagnostics": diag}
