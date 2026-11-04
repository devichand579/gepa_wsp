import numpy as np

def edit(w, f):
    # Extract data
    x = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)  # (2000, 784)
    y = f["y"].astype(np.int64)
    N = x.shape[0]
    num_classes = 10

    # Global mean and centered data
    mu = x.mean(axis=0)
    Xc = x - mu

    # PCA via SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T  # (784, 784)

    # Use top 128 PCA directions for LDA
    n_pca_lda = min(128, V.shape[1])
    V128 = V[:, :n_pca_lda]
    Z = Xc @ V128  # (2000, 128)

    # Class means in Z
    mu_c_z = np.zeros((num_classes, n_pca_lda))
    for c in range(num_classes):
        mu_c_z[c] = Z[y == c].mean(axis=0)
    mu_z = Z.mean(axis=0)

    # Within-class scatter
    Sw = np.zeros((n_pca_lda, n_pca_lda))
    for c in range(num_classes):
        Zc = Z[y == c] - mu_c_z[c]
        Sw += Zc.T @ Zc
    Sw /= N

    # Between-class scatter
    Sb = np.zeros((n_pca_lda, n_pca_lda))
    for c in range(num_classes):
        n_c = np.sum(y == c)
        diff = (mu_c_z[c] - mu_z).reshape(-1, 1)
        Sb += n_c * (diff @ diff.T)
    Sb /= N

    # Regularized LDA
    Sw_reg = Sw + 1e-3 * np.eye(n_pca_lda)
    try:
        eigvals, eigvecs = np.linalg.eig(np.linalg.solve(Sw_reg, Sb))
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(Sw_reg) @ Sb)

    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    n_lda = min(num_classes - 1, eigvecs.shape[1])
    W_lda_z = eigvecs[:, :n_lda]
    W_lda = V128 @ W_lda_z  # (784, 9)

    # Normalize LDA directions
    norms = np.linalg.norm(W_lda, axis=0, keepdims=True)
    norms[norms < 1e-8] = 1.0
    W_lda = W_lda / norms

    # Fill remaining directions with PCA
    n_pca_extra = 64 - n_lda
    if n_pca_extra > 0:
        W_pca = V[:, :n_pca_extra]
        D = np.hstack([W_lda, W_pca])
    else:
        D = W_lda[:, :64]

    if D.shape[1] < 64:
        extra = 64 - D.shape[1]
        W_extra = V[:, n_pca_extra:n_pca_extra + extra]
        D = np.hstack([D, W_extra])
    D = D[:, :64]

    # Build fc1 weights and biases (duplicate each direction with its negative)
    W1 = np.zeros((128, 784), dtype=np.float64)
    b1 = np.zeros(128, dtype=np.float64)
    for i in range(64):
        d = D[:, i]
        W1[2 * i] = d
        b1[2 * i] = -mu @ d
        W1[2 * i + 1] = -d
        b1[2 * i + 1] = mu @ d

    # Compute hidden features
    H = x @ W1.T + b1
    H = np.maximum(H, 0)

    # Standardize hidden features
    H_mean = H.mean(axis=0)
    H_std = H.std(axis=0)
    H_std[H_std < 1e-8] = 1.0
    H_norm = (H - H_mean) / H_std

    # One-hot targets
    Y = np.zeros((N, num_classes))
    Y[np.arange(N), y] = 1.0

    # Cross-validation for ridge lambda
    lambdas = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
    best_lambda = 1e-2
    best_acc = -1.0
    n_folds = 5
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = N // n_folds

    for lam in lambdas:
        accs = []
        for fold in range(n_folds):
            val_idx = indices[fold * fold_size:(fold + 1) * fold_size]
            train_idx = np.setdiff1d(indices, val_idx)
            H_train = H_norm[train_idx]
            Y_train = Y[train_idx]
            H_val = H_norm[val_idx]
            H_train_aug = np.hstack([H_train, np.ones((len(train_idx), 1))])
            H_val_aug = np.hstack([H_val, np.ones((len(val_idx), 1))])
            A = H_train_aug.T @ H_train_aug + lam * np.eye(H_train_aug.shape[1])
            b = H_train_aug.T @ Y_train
            W_aug = np.linalg.solve(A, b)
            logits_val = H_val_aug @ W_aug
            preds = np.argmax(logits_val, axis=1)
            acc = np.mean(preds == y[val_idx])
            accs.append(acc)
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lambda = lam

    # Fit on full data
    H_aug = np.hstack([H_norm, np.ones((N, 1))])
    A = H_aug.T @ H_aug + best_lambda * np.eye(H_aug.shape[1])
    b = H_aug.T @ Y
    W_aug = np.linalg.solve(A, b)

    W2_std = W_aug[:128, :].T
    b2_std = W_aug[128, :]

    # Unstandardize
    W2 = W2_std / H_std
    b2 = b2_std - W2_std @ (H_mean / H_std)

    # Update weights
    w["fc1.weight"] = W1.astype(np.float32)
    w["fc1.bias"] = b1.astype(np.float32)
    w["fc2.weight"] = W2.astype(np.float32)
    w["fc2.bias"] = b2.astype(np.float32)

    # Diagnostics
    logits_full = H @ W2.T + b2
    preds_full = np.argmax(logits_full, axis=1)
    acc_full = np.mean(preds_full == y)
    probs_full = np.exp(logits_full - logits_full.max(axis=1, keepdims=True))
    probs_full /= probs_full.sum(axis=1, keepdims=True)
    ce_full = -np.mean(np.log(probs_full[np.arange(N), y] + 1e-12))

    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_acc),
        "full_accuracy": float(acc_full),
        "full_cross_entropy": float(ce_full),
        "fc1_weight_norm": float(np.linalg.norm(W1)),
        "fc1_bias_norm": float(np.linalg.norm(b1)),
        "fc2_weight_norm": float(np.linalg.norm(W2)),
        "fc2_bias_norm": float(np.linalg.norm(b2)),
        "hidden_zero_fraction": float(np.mean(H == 0)),
    }

    return {"weights": w, "diagnostics": diagnostics}
