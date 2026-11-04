import numpy as np
import math

def edit(w, f):
    # Extract data
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)  # N x 784
    y = f["y"].astype(np.int64)
    N, D = X.shape
    C = 10

    # Current weights
    W1 = w["fc1.weight"].astype(np.float64)  # 128 x 784
    b1 = w["fc1.bias"].astype(np.float64)    # 128
    W2 = w["fc2.weight"].astype(np.float64)  # 10 x 128
    b2 = w["fc2.bias"].astype(np.float64)    # 10

    # Class means and overall mean
    mu = X.mean(axis=0)
    mu_c = np.zeros((C, D))
    for c in range(C):
        mu_c[c] = X[y == c].mean(axis=0)

    # Within-class scatter
    Sw = np.zeros((D, D))
    for c in range(C):
        Xc = X[y == c]
        Xc_centered = Xc - mu_c[c]
        Sw += Xc_centered.T @ Xc_centered
    Sw /= N
    reg = 1e-3 * np.trace(Sw) / D
    Sw_reg = Sw + reg * np.eye(D)

    # Between-class scatter
    Sb = np.zeros((D, D))
    for c in range(C):
        diff = (mu_c[c] - mu).reshape(-1, 1)
        Sb += (y == c).sum() * (diff @ diff.T)
    Sb /= N

    # LDA directions (top 9)
    Sw_inv = np.linalg.inv(Sw_reg)
    M = Sw_inv @ Sb
    eigvals, eigvecs = np.linalg.eig(M)
    idx = np.argsort(np.real(eigvals))[::-1]
    lda_dirs = np.real(eigvecs[:, idx[:9]]).T  # 9 x 784
    for i in range(lda_dirs.shape[0]):
        norm = np.linalg.norm(lda_dirs[i])
        if norm > 1e-12:
            lda_dirs[i] /= norm

    # Class mean directions (10)
    mean_dirs = mu_c - mu  # 10 x 784
    for i in range(mean_dirs.shape[0]):
        norm = np.linalg.norm(mean_dirs[i])
        if norm > 1e-12:
            mean_dirs[i] /= norm

    # PCA directions
    X_centered = X - mu
    Cov = X_centered.T @ X_centered / (N - 1)
    eigvals_pca, eigvecs_pca = np.linalg.eigh(Cov)
    idx_pca = np.argsort(eigvals_pca)[::-1]
    pca_dirs = eigvecs_pca[:, idx_pca].T  # 784 x 784, take top

    # Candidate directions pool: LDA, then class means, then PCA
    pool = []
    pool.extend(lda_dirs)
    pool.extend(mean_dirs)
    pool.extend(pca_dirs)
    pool = np.array(pool)  # up to 803 x 784

    # Identify dead units (most negative biases)
    dead_indices = np.argsort(b1)  # ascending

    # Mean norm of existing fc1 rows for scaling new directions
    mean_norm = np.mean(np.linalg.norm(W1, axis=1))

    # Cross-validation setup
    n_folds = 5
    rng = np.random.RandomState(42)
    indices = np.arange(N)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    K_list = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119, 128]
    alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    best_cv_acc = -1.0
    best_K = 0
    best_alpha = 0.01

    for K in K_list:
        W1_new = W1.copy()
        b1_new = b1.copy()
        for i in range(min(K, len(dead_indices))):
            idx_unit = dead_indices[i]
            if i < len(pool):
                d = pool[i] * mean_norm
                W1_new[idx_unit] = d
                # Center activations: bias = -mean(X @ d)
                b1_new[idx_unit] = -np.mean(X @ d)
            else:
                W1_new[idx_unit] = 0.0
                b1_new[idx_unit] = 0.0

        H = np.maximum(0, X @ W1_new.T + b1_new)  # N x 128

        for alpha in alphas:
            acc_sum = 0.0
            for fold in folds:
                train_idx = np.setdiff1d(indices, fold)
                test_idx = fold
                H_train = H[train_idx]
                y_train = y[train_idx]
                H_test = H[test_idx]
                y_test = y[test_idx]

                H_train_aug = np.hstack([H_train, np.ones((len(train_idx), 1))])
                H_test_aug = np.hstack([H_test, np.ones((len(test_idx), 1))])

                Y_train = np.zeros((len(train_idx), C))
                Y_train[np.arange(len(train_idx)), y_train] = 1.0

                A = H_train_aug.T @ H_train_aug + alpha * np.eye(H_train_aug.shape[1])
                B = H_train_aug.T @ Y_train
                W_aug = np.linalg.solve(A, B).T  # C x (128+1)

                logits = H_test_aug @ W_aug.T
                pred = np.argmax(logits, axis=1)
                acc_sum += np.mean(pred == y_test)
            cv_acc = acc_sum / n_folds
            if cv_acc > best_cv_acc:
                best_cv_acc = cv_acc
                best_K = K
                best_alpha = alpha

    # Refit on full pool with best hyperparameters
    K = best_K
    alpha = best_alpha
    W1_new = W1.copy()
    b1_new = b1.copy()
    for i in range(min(K, len(dead_indices))):
        idx_unit = dead_indices[i]
        if i < len(pool):
            d = pool[i] * mean_norm
            W1_new[idx_unit] = d
            b1_new[idx_unit] = -np.mean(X @ d)
        else:
            W1_new[idx_unit] = 0.0
            b1_new[idx_unit] = 0.0

    H = np.maximum(0, X @ W1_new.T + b1_new)
    H_aug = np.hstack([H, np.ones((N, 1))])
    Y = np.zeros((N, C))
    Y[np.arange(N), y] = 1.0
    A = H_aug.T @ H_aug + alpha * np.eye(H_aug.shape[1])
    B = H_aug.T @ Y
    W_aug = np.linalg.solve(A, B).T  # C x (128+1)

    W2_new = W_aug[:, :128]
    b2_new = W_aug[:, 128]

    # Clip to safe range
    W1_new = np.clip(W1_new, -1000.0, 1000.0)
    b1_new = np.clip(b1_new, -1000.0, 1000.0)
    W2_new = np.clip(W2_new, -1000.0, 1000.0)
    b2_new = np.clip(b2_new, -1000.0, 1000.0)

    # Update weights
    w["fc1.weight"] = W1_new.astype(np.float32)
    w["fc1.bias"] = b1_new.astype(np.float32)
    w["fc2.weight"] = W2_new.astype(np.float32)
    w["fc2.bias"] = b2_new.astype(np.float32)

    # Diagnostics
    logits_full = H_aug @ W_aug.T
    pred_full = np.argmax(logits_full, axis=1)
    full_acc = np.mean(pred_full == y)

    diagnostics = {
        "best_K": int(best_K),
        "best_alpha": float(best_alpha),
        "best_cv_acc": float(best_cv_acc),
        "full_pool_acc": float(full_acc),
        "replaced_units": [int(dead_indices[i]) for i in range(min(K, len(dead_indices)))],
        "fc1_weight_norm": float(np.linalg.norm(W1_new)),
        "fc2_weight_norm": float(np.linalg.norm(W2_new)),
        "fc2_bias_norm": float(np.linalg.norm(b2_new))
    }

    return {"weights": w, "diagnostics": diagnostics}
