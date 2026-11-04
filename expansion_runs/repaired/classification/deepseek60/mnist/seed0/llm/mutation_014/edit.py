import numpy as np

def edit(w, f):
    # Extract full feedback pool
    X = f["x"].reshape(2000, 784).astype(np.float64)
    y = f["y"].astype(np.int64)
    n, d = X.shape
    C = 10

    # Class means and global mean
    classes = np.arange(C)
    mu_c = np.array([X[y == c].mean(axis=0) for c in classes])  # (10,784)
    mu = X.mean(axis=0)  # (784,)

    # Within-class centered data
    Xc = X - mu_c[y]  # (2000,784)

    # SVD of within-class centered data
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T  # (784,2000)
    S_inv = 1.0 / (S + 1e-6)

    # LDA directions: whiten within-class covariance, then PCA on class means
    D = mu_c - mu  # (10,784)
    M = (D @ V) * S_inv  # (10,2000)
    U_m, S_m, Vt_m = np.linalg.svd(M, full_matrices=False)
    # Top 9 LDA directions in original space
    LDA_dirs = V @ (Vt_m[:9].T * S_inv[:, None])  # (784,9)

    # Within-class PCA directions (top 109)
    within_pca = V[:, :109].T  # (109,784)

    # Combine: 10 class means + 9 LDA + 109 within-class PCA = 128 rows
    W1 = np.vstack([mu_c, LDA_dirs.T, within_pca])  # (128,784)

    # Bias: center the projections
    mean_proj = (X @ W1.T).mean(axis=0)
    b1 = -mean_proj  # (128,)

    # Compute hidden features
    H = np.maximum(0, X @ W1.T + b1)  # (2000,128)

    # 5-fold CV to choose between ridge and LDA, and ridge lambda
    indices = np.arange(n)
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    folds = np.array_split(indices, 5)

    def fit_ridge(H_tr, y_tr, H_va, y_va, lam):
        n_tr, d_tr = H_tr.shape
        H_aug = np.hstack([H_tr, np.ones((n_tr, 1))])
        Y = np.zeros((n_tr, C))
        Y[np.arange(n_tr), y_tr] = 1.0
        A = H_aug.T @ H_aug + lam * np.eye(d_tr + 1)
        B = H_aug.T @ Y
        W_aug = np.linalg.solve(A, B)
        H_va_aug = np.hstack([H_va, np.ones((H_va.shape[0], 1))])
        scores = H_va_aug @ W_aug
        pred = np.argmax(scores, axis=1)
        return np.mean(pred == y_va), W_aug

    def fit_lda(H_tr, y_tr, H_va, y_va, lam):
        n_tr, d_tr = H_tr.shape
        mu_c_tr = np.array([H_tr[y_tr == c].mean(axis=0) for c in classes])
        Sw = np.zeros((d_tr, d_tr))
        for c in classes:
            Hc = H_tr[y_tr == c] - mu_c_tr[c]
            Sw += Hc.T @ Hc
        Sw /= n_tr
        Sw += lam * np.eye(d_tr)
        Sw_inv = np.linalg.inv(Sw)
        W2 = Sw_inv @ mu_c_tr.T  # (d,10)
        b2 = -0.5 * np.sum(mu_c_tr * W2.T, axis=1) + np.log(1.0 / C)
        scores = H_va @ W2 + b2
        pred = np.argmax(scores, axis=1)
        return np.mean(pred == y_va), W2, b2

    # Evaluate ridge with several lambdas
    ridge_lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    best_ridge_acc = -1.0
    best_ridge_lam = None
    for lam in ridge_lambdas:
        accs = []
        for i in range(5):
            va_idx = folds[i]
            tr_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            acc, _ = fit_ridge(H[tr_idx], y[tr_idx], H[va_idx], y[va_idx], lam)
            accs.append(acc)
        mean_acc = np.mean(accs)
        if mean_acc > best_ridge_acc:
            best_ridge_acc = mean_acc
            best_ridge_lam = lam

    # Evaluate LDA with several lambdas
    lda_lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    best_lda_acc = -1.0
    best_lda_lam = None
    for lam in lda_lambdas:
        accs = []
        for i in range(5):
            va_idx = folds[i]
            tr_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            acc, _, _ = fit_lda(H[tr_idx], y[tr_idx], H[va_idx], y[va_idx], lam)
            accs.append(acc)
        mean_acc = np.mean(accs)
        if mean_acc > best_lda_acc:
            best_lda_acc = mean_acc
            best_lda_lam = lam

    # Choose best method
    if best_ridge_acc >= best_lda_acc:
        method = "ridge"
        best_lam = best_ridge_lam
        best_cv_acc = best_ridge_acc
    else:
        method = "lda"
        best_lam = best_lda_lam
        best_cv_acc = best_lda_acc

    # Fit final fc2 on all data using best method
    if method == "ridge":
        _, W_aug = fit_ridge(H, y, H, y, best_lam)
        W2 = W_aug[:-1].T  # (10,128)
        b2 = W_aug[-1]     # (10,)
    else:
        _, W2, b2 = fit_lda(H, y, H, y, best_lam)
        W2 = W2.T  # (10,128)

    # Update weights
    w["fc1.weight"] = W1.astype(np.float32)
    w["fc1.bias"] = b1.astype(np.float32)
    w["fc2.weight"] = W2.astype(np.float32)
    w["fc2.bias"] = b2.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "method": method,
        "best_lambda": float(best_lam),
        "cv_accuracy": float(best_cv_acc),
        "ridge_cv_accuracy": float(best_ridge_acc),
        "lda_cv_accuracy": float(best_lda_acc),
        "fc1_weight_norm": float(np.linalg.norm(w["fc1.weight"])),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
    }
    return {"weights": w, "diagnostics": diagnostics}
