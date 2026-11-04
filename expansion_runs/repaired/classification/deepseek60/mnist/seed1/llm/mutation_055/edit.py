import numpy as np

def edit(w, f):
    # Load feedback data
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float32)  # (2000, 784)
    y = f["y"].astype(np.int64)
    N = X.shape[0]
    C = 10
    Y = np.zeros((N, C), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    # Current features from fc1 (already ReLU output)
    H_cur = f["features"].astype(np.float32)  # (2000, 128)

    # Helper: ridge regression with 5-fold CV to choose lambda
    def ridge_cv(H, Y, lambdas):
        N, D = H.shape
        C = Y.shape[1]
        # Augment with bias column
        H_aug = np.hstack([H, np.ones((N, 1), dtype=np.float32)])  # (N, D+1)
        # Precompute for full data
        HtH = H_aug.T @ H_aug
        HtY = H_aug.T @ Y
        # Identity for regularization (do not regularize bias)
        reg_mask = np.ones(D + 1, dtype=np.float32)
        reg_mask[-1] = 0.0

        best_lambda = None
        best_cv_acc = -1.0
        best_W = None
        best_b = None

        # 5-fold CV indices
        indices = np.arange(N)
        np.random.seed(42)
        np.random.shuffle(indices)
        folds = np.array_split(indices, 5)

        for lam in lambdas:
            accs = []
            for i in range(5):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
                H_train = H_aug[train_idx]
                Y_train = Y[train_idx]
                H_val = H_aug[val_idx]
                Y_val = Y[val_idx]
                # Solve (H^T H + lam * diag(reg_mask)) W = H^T Y
                HtH_train = H_train.T @ H_train
                HtY_train = H_train.T @ Y_train
                A = HtH_train + lam * np.diag(reg_mask)
                W_aug = np.linalg.solve(A, HtY_train)  # (D+1, C)
                # Predict on validation
                logits_val = H_val @ W_aug  # (N_val, C)
                preds = np.argmax(logits_val, axis=1)
                acc = np.mean(preds == np.argmax(Y_val, axis=1))
                accs.append(acc)
            mean_acc = np.mean(accs)
            if mean_acc > best_cv_acc:
                best_cv_acc = mean_acc
                best_lambda = lam
                # Refit on full data with this lambda
                A_full = HtH + lam * np.diag(reg_mask)
                W_aug_full = np.linalg.solve(A_full, HtY)
                best_W = W_aug_full[:-1, :].T  # (C, D)
                best_b = W_aug_full[-1, :]     # (C,)
        return best_lambda, best_W, best_b, best_cv_acc

    # Option A: keep fc1, refit fc2 on H_cur
    lambdas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    lam_A, W_A, b_A, acc_A = ridge_cv(H_cur, Y, lambdas)

    # Option B: LDA + PCA projection for fc1
    # Compute class means and overall mean
    mu = X.mean(axis=0)  # (784,)
    mu_c = np.zeros((C, X.shape[1]), dtype=np.float32)
    for c in range(C):
        mu_c[c] = X[y == c].mean(axis=0)
    # Within-class scatter Sw
    Sw = np.zeros((X.shape[1], X.shape[1]), dtype=np.float32)
    for c in range(C):
        Xc = X[y == c] - mu_c[c]
        Sw += Xc.T @ Xc
    # Add small ridge to Sw for numerical stability
    Sw += 1e-3 * np.trace(Sw) / X.shape[1] * np.eye(X.shape[1], dtype=np.float32)
    # Cholesky of Sw
    L = np.linalg.cholesky(Sw)
    # Between-class scatter via B matrix
    B = np.zeros((C, X.shape[1]), dtype=np.float32)
    for c in range(C):
        n_c = np.sum(y == c)
        B[c] = np.sqrt(n_c) * (mu_c[c] - mu)
    # C = B @ L^{-T}
    # Solve L^T C^T = B^T  => C^T = solve(L^T, B^T)
    C_mat = np.linalg.solve(L.T, B.T).T  # (C, 784)
    # SVD of C_mat
    U, S, Vt = np.linalg.svd(C_mat, full_matrices=False)  # Vt: (C, 784)
    # LDA directions in original space: v = L^{-T} Vt.T
    # Solve L^T V_orig = Vt.T
    V_lda = np.linalg.solve(L.T, Vt.T)  # (784, C)
    # Take first 9 LDA directions (10th is zero)
    V_lda = V_lda[:, :9]  # (784, 9)

    # PCA directions: top 119 principal components of centered X
    X_centered = X - mu
    # Use SVD on X_centered (2000x784)
    U_pca, S_pca, Vt_pca = np.linalg.svd(X_centered, full_matrices=False)
    V_pca = Vt_pca[:119].T  # (784, 119)

    # Combine
    V = np.hstack([V_lda, V_pca])  # (784, 128)
    # New fc1 weights: (128, 784)
    fc1_weight_new = V.T.astype(np.float32)
    fc1_bias_new = np.zeros(128, dtype=np.float32)

    # Compute new features
    H_new = np.maximum(0, X @ V + fc1_bias_new)  # (2000, 128)

    # Fit fc2 on H_new
    lam_B, W_B, b_B, acc_B = ridge_cv(H_new, Y, lambdas)

    # Choose better option
    if acc_B > acc_A:
        # Use Option B
        w["fc1.weight"] = fc1_weight_new
        w["fc1.bias"] = fc1_bias_new
        w["fc2.weight"] = W_B.astype(np.float32)
        w["fc2.bias"] = b_B.astype(np.float32)
        chosen = "B"
        best_acc = acc_B
        best_lam = lam_B
    else:
        # Use Option A
        w["fc2.weight"] = W_A.astype(np.float32)
        w["fc2.bias"] = b_A.astype(np.float32)
        chosen = "A"
        best_acc = acc_A
        best_lam = lam_A

    # Ensure finite and within bounds
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)

    diagnostics = {
        "chosen_option": chosen,
        "cv_accuracy_A": float(acc_A),
        "cv_accuracy_B": float(acc_B),
        "best_lambda_A": float(lam_A),
        "best_lambda_B": float(lam_B),
        "fc2_weight_mean": float(np.mean(w["fc2.weight"])),
        "fc2_weight_std": float(np.std(w["fc2.weight"])),
        "fc2_bias_mean": float(np.mean(w["fc2.bias"])),
        "fc2_bias_std": float(np.std(w["fc2.bias"])),
    }
    return {"weights": w, "diagnostics": diagnostics}
