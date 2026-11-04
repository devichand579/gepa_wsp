import numpy as np

def edit(w, f):
    # Helper: fit fc2 (10x128 weight, 10 bias) on given features X and labels y.
    # Uses closed-form ridge regression and LDA with shrinkage, picking the best
    # training accuracy among a small set of fixed shrinkage values.
    def fit_fc2(X, y):
        N, D = X.shape
        C = 10
        Y = np.zeros((N, C))
        Y[np.arange(N), y] = 1.0

        best_acc = -1.0
        best_W = None
        best_b = None

        # --- Ridge regression ---
        X_aug = np.hstack([X, np.ones((N, 1))])
        for lam in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
            A = X_aug.T @ X_aug + lam * np.eye(D + 1)
            B = X_aug.T @ Y
            W_aug = np.linalg.solve(A, B)
            W = W_aug[:D].T          # (10, D)
            b = W_aug[D]             # (10,)
            logits = X @ W.T + b
            pred = np.argmax(logits, axis=1)
            acc = np.mean(pred == y)
            if acc > best_acc:
                best_acc = acc
                best_W = W
                best_b = b

        # --- LDA with shrinkage ---
        means = np.zeros((C, D))
        for c in range(C):
            means[c] = X[y == c].mean(axis=0)
        Sigma = np.zeros((D, D))
        for c in range(C):
            Xc = X[y == c]
            Xc_centered = Xc - means[c]
            Sigma += Xc_centered.T @ Xc_centered
        Sigma /= (N - C)
        for lam in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
            Sigma_shrunk = Sigma + lam * np.eye(D)
            W_lda = np.linalg.solve(Sigma_shrunk, means.T)   # (D, C)
            b_lda = -0.5 * np.sum(means.T * W_lda, axis=0) + np.log(np.bincount(y, minlength=C) / N)
            W = W_lda.T          # (10, D)
            b = b_lda
            logits = X @ W.T + b
            pred = np.argmax(logits, axis=1)
            acc = np.mean(pred == y)
            if acc > best_acc:
                best_acc = acc
                best_W = W
                best_b = b

        return best_W, best_b, best_acc

    # Current hidden features
    X_curr = f["features"].astype(np.float64)
    y = f["y"].astype(np.int64)

    W_curr, b_curr, acc_curr = fit_fc2(X_curr, y)

    # --- Optional: PCA-based fc1 alternative ---
    x_flat = f["x"].reshape(2000, 784).astype(np.float64)
    mean_x = x_flat.mean(axis=0)
    x_centered = x_flat - mean_x
    # SVD to get top 128 principal components
    U, S, Vt = np.linalg.svd(x_centered, full_matrices=False)
    components = Vt[:128]                     # (128, 784)
    fc1_w_pca = components
    fc1_b_pca = -components @ mean_x
    X_pca = np.maximum(0, x_flat @ fc1_w_pca.T + fc1_b_pca)
    W_pca, b_pca, acc_pca = fit_fc2(X_pca, y)

    # Choose the better closed-form solution
    if acc_pca > acc_curr:
        w["fc1.weight"] = fc1_w_pca.astype(np.float32)
        w["fc1.bias"] = fc1_b_pca.astype(np.float32)
        w["fc2.weight"] = W_pca.astype(np.float32)
        w["fc2.bias"] = b_pca.astype(np.float32)
        chosen = "pca"
        best_acc = acc_pca
    else:
        w["fc2.weight"] = W_curr.astype(np.float32)
        w["fc2.bias"] = b_curr.astype(np.float32)
        chosen = "current"
        best_acc = acc_curr

    diagnostics = {
        "chosen": chosen,
        "best_train_acc": float(best_acc),
        "acc_current_features": float(acc_curr),
        "acc_pca_features": float(acc_pca),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
        "fc2_weight_mean": float(np.mean(w["fc2.weight"])),
        "fc2_weight_std": float(np.std(w["fc2.weight"])),
    }
    return {"weights": w, "diagnostics": diagnostics}
