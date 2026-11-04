import numpy as np

def edit(w, f):
    # Extract data
    X_img = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)  # (N, 784)
    y = f["y"].astype(np.int64)
    N = X_img.shape[0]
    X_feat = f["features"].astype(np.float64)  # (N, 128)

    # Helper: ridge regression fit and predict
    def ridge_fit_predict(X_train, y_train, X_val, lam):
        n, d = X_train.shape
        # augment with bias
        X_train_aug = np.hstack([X_train, np.ones((n, 1))])
        X_val_aug = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
        # one-hot targets
        Y_train = np.zeros((n, 10))
        Y_train[np.arange(n), y_train] = 1.0
        # solve (X^T X + lam I) W = X^T Y
        A = X_train_aug.T @ X_train_aug + lam * np.eye(d + 1)
        B = X_train_aug.T @ Y_train
        W = np.linalg.solve(A, B)  # (d+1, 10)
        logits = X_val_aug @ W
        pred = np.argmax(logits, axis=1)
        return pred, W

    # Helper: LDA fit and predict
    def lda_fit_predict(X_train, y_train, X_val, lam):
        n, d = X_train.shape
        classes = np.unique(y_train)
        means = np.zeros((len(classes), d))
        priors = np.zeros(len(classes))
        for i, c in enumerate(classes):
            idx = (y_train == c)
            means[i] = X_train[idx].mean(axis=0)
            priors[i] = idx.mean()
        # pooled covariance
        Sigma = np.zeros((d, d))
        for i, c in enumerate(classes):
            idx = (y_train == c)
            diff = X_train[idx] - means[i]
            Sigma += diff.T @ diff
        Sigma /= (n - len(classes))
        # regularize
        Sigma += lam * np.eye(d)
        # weights: solve(Sigma, means^T) -> (d, C)
        W = np.linalg.solve(Sigma, means.T)  # (d, C)
        # bias: -0.5 * mu_c^T W_c + log prior
        b = -0.5 * np.sum(means.T * W, axis=0) + np.log(priors)
        logits = X_val @ W + b
        pred = np.argmax(logits, axis=1)
        return pred, (W, b)

    # 5-fold CV evaluation
    def evaluate_cv(X, y, method, lam):
        n = X.shape[0]
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)
        folds = np.array_split(indices, 5)
        accs = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            if method == "ridge":
                pred, _ = ridge_fit_predict(X_train, y_train, X_val, lam)
            else:
                pred, _ = lda_fit_predict(X_train, y_train, X_val, lam)
            accs.append(np.mean(pred == y_val))
        return np.mean(accs)

    # Candidate feature sets
    candidates = {}
    candidates["current"] = X_feat

    # Candidate: current fc1 with bias=0
    fc1_w = w["fc1.weight"].astype(np.float64)
    fc1_b = w["fc1.bias"].astype(np.float64)
    H_zero = np.maximum(0, X_img @ fc1_w.T + 0.0)
    candidates["fc1_bias0"] = H_zero

    # Candidate: current fc1 with centered bias
    pre = X_img @ fc1_w.T + fc1_b
    mean_pre = pre.mean(axis=0)
    H_centered = np.maximum(0, X_img @ fc1_w.T - mean_pre)
    candidates["fc1_centered"] = H_centered

    # Candidate: PCA features
    X_centered = X_img - X_img.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    V = Vt[:128]  # (128, 784)
    mean_img = X_img.mean(axis=0)
    H_pca = np.maximum(0, X_img @ V.T - V @ mean_img)
    candidates["pca"] = H_pca

    # Evaluate candidates with CV
    best_acc = -1
    best_candidate = None
    best_method = None
    best_lam = None
    best_params = None

    # Try ridge and LDA for each candidate
    lambdas_ridge = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    lambdas_lda = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    for name, H in candidates.items():
        for lam in lambdas_ridge:
            acc = evaluate_cv(H, y, "ridge", lam)
            if acc > best_acc:
                best_acc = acc
                best_candidate = name
                best_method = "ridge"
                best_lam = lam
        for lam in lambdas_lda:
            acc = evaluate_cv(H, y, "lda", lam)
            if acc > best_acc:
                best_acc = acc
                best_candidate = name
                best_method = "lda"
                best_lam = lam

    # Refit best on all data
    H_best = candidates[best_candidate]
    if best_method == "ridge":
        pred, W_full = ridge_fit_predict(H_best, y, H_best, best_lam)
        # W_full shape (d+1, 10)
        fc2_w = W_full[:128, :].T  # (10, 128)
        fc2_b = W_full[128, :]     # (10,)
    else:
        pred, (W, b) = lda_fit_predict(H_best, y, H_best, best_lam)
        fc2_w = W.T  # (10, 128)
        fc2_b = b

    # Update weights
    if best_candidate == "pca":
        w["fc1.weight"] = V.astype(np.float32)
        w["fc1.bias"] = (-V @ mean_img).astype(np.float32)
    elif best_candidate == "fc1_bias0":
        w["fc1.weight"] = fc1_w.astype(np.float32)
        w["fc1.bias"] = np.zeros_like(fc1_b, dtype=np.float32)
    elif best_candidate == "fc1_centered":
        w["fc1.weight"] = fc1_w.astype(np.float32)
        w["fc1.bias"] = (-mean_pre).astype(np.float32)
    # else current: keep fc1 as is

    w["fc2.weight"] = fc2_w.astype(np.float32)
    w["fc2.bias"] = fc2_b.astype(np.float32)

    # Clip to safe range
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)

    # Diagnostics
    diagnostics = {
        "best_candidate": best_candidate,
        "best_method": best_method,
        "best_lambda": float(best_lam),
        "cv_accuracy": float(best_acc),
        "train_accuracy": float(np.mean(pred == y)),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
    }

    return {"weights": w, "diagnostics": diagnostics}
