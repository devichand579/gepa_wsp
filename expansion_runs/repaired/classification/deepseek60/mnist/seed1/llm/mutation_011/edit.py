import numpy as np
import math

def edit(w, f):
    # Extract full training pool
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)  # (2000, 784)
    y = f["y"].astype(np.int64)
    N, D = X.shape
    C = 10

    # Helper: ridge regression with bias
    def fit_ridge(H, y, lam):
        H_aug = np.concatenate([H, np.ones((H.shape[0], 1))], axis=1)
        Y = np.zeros((H.shape[0], C))
        Y[np.arange(H.shape[0]), y] = 1.0
        K = H_aug.shape[1]
        A = H_aug.T @ H_aug + lam * np.eye(K)
        B = H_aug.T @ Y
        W = np.linalg.solve(A, B)  # (K, C)
        return W

    def predict(H, W):
        H_aug = np.concatenate([H, np.ones((H.shape[0], 1))], axis=1)
        logits = H_aug @ W
        return np.argmax(logits, axis=1)

    # Feature construction functions
    def pca_features(X_train, X_test, n_components=128, use_relu=True, linear_bias=None):
        mu = X_train.mean(axis=0)
        Xc_train = X_train - mu
        # SVD on centered training data
        U, S, Vt = np.linalg.svd(Xc_train, full_matrices=False)
        P = Vt[:n_components]  # (n_components, D)
        # Project
        H_train = Xc_train @ P.T
        H_test = (X_test - mu) @ P.T
        if use_relu:
            if linear_bias is not None:
                H_train = np.maximum(H_train + linear_bias, 0.0)
                H_test = np.maximum(H_test + linear_bias, 0.0)
            else:
                H_train = np.maximum(H_train, 0.0)
                H_test = np.maximum(H_test, 0.0)
        else:
            # linear: just use centered projections (bias can be absorbed)
            pass
        return H_train, H_test

    def lda_pca_features(X_train, X_test, n_lda=9, n_pca=119, use_relu=True, linear_bias=None):
        mu = X_train.mean(axis=0)
        Xc_train = X_train - mu
        # PCA part
        U, S, Vt = np.linalg.svd(Xc_train, full_matrices=False)
        P_pca = Vt[:n_pca]  # (n_pca, D)
        # LDA part
        Sw = np.zeros((D, D))
        Sb = np.zeros((D, D))
        overall_mean = Xc_train.mean(axis=0)
        for c in range(C):
            idx = (y == c)
            if idx.sum() == 0:
                continue
            Xc_c = Xc_train[idx]
            mu_c = Xc_c.mean(axis=0)
            Sw += (Xc_c - mu_c).T @ (Xc_c - mu_c)
            n_c = idx.sum()
            diff = mu_c - overall_mean
            Sb += n_c * np.outer(diff, diff)
        # Regularize Sw
        Sw_reg = Sw + 1e-3 * np.eye(D)
        # Solve generalized eigenvalue problem: Sw^{-1} Sb v = lambda v
        # Use scipy? Only numpy allowed. We can solve via eig of Sw^{-1} Sb.
        # Sw is symmetric positive definite, so we can use Cholesky.
        try:
            L = np.linalg.cholesky(Sw_reg)
            L_inv = np.linalg.inv(L)
            M = L_inv @ Sb @ L_inv.T
            eigvals, eigvecs = np.linalg.eigh(M)
            # Sort descending
            idx = np.argsort(eigvals)[::-1]
            eigvecs = eigvecs[:, idx]
            # Convert back: v = L_inv.T @ eigvecs
            V_lda = L_inv.T @ eigvecs[:, :n_lda]  # (D, n_lda)
            V_lda = V_lda.T  # (n_lda, D)
        except np.linalg.LinAlgError:
            # Fallback: use PCA only
            V_lda = np.zeros((0, D))
        # Combine
        if V_lda.shape[0] > 0:
            P_combined = np.concatenate([V_lda, P_pca], axis=0)
        else:
            P_combined = P_pca
        # Ensure exactly 128 components (pad with zeros if needed)
        if P_combined.shape[0] < 128:
            pad = np.zeros((128 - P_combined.shape[0], D))
            P_combined = np.concatenate([P_combined, pad], axis=0)
        elif P_combined.shape[0] > 128:
            P_combined = P_combined[:128]
        # Normalize each row to unit norm
        norms = np.linalg.norm(P_combined, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        P_combined = P_combined / norms
        # Project
        H_train = Xc_train @ P_combined.T
        H_test = (X_test - mu) @ P_combined.T
        if use_relu:
            if linear_bias is not None:
                H_train = np.maximum(H_train + linear_bias, 0.0)
                H_test = np.maximum(H_test + linear_bias, 0.0)
            else:
                H_train = np.maximum(H_train, 0.0)
                H_test = np.maximum(H_test, 0.0)
        else:
            pass
        return H_train, H_test

    # Cross-validation to choose method and lambda
    np.random.seed(0)
    indices = np.random.permutation(N)
    folds = np.array_split(indices, 5)

    methods = []
    # Method 0: current fc1 (use existing weights to compute features)
    # We'll handle this separately because it uses w["fc1.weight"] and w["fc1.bias"]
    # For CV, we need to compute features using current fc1 on each fold.
    # But current fc1 is fixed, so we can just compute features for all X once.
    W1 = w["fc1.weight"].astype(np.float64)
    b1 = w["fc1.bias"].astype(np.float64)
    H_current = np.maximum(X @ W1.T + b1, 0.0)  # (N, 128)

    # For other methods, we need to compute features per fold.
    # We'll define a list of (name, feature_func) where feature_func(X_train, X_test) returns H_train, H_test.
    # feature_func will be called with the training and validation parts of the fold.
    # We'll also try different lambdas.

    lambdas = [1e-3, 1e-2, 1e-1, 1.0, 10.0]

    best_score = -1
    best_method = None
    best_lambda = None

    # Evaluate current fc1 features
    for lam in lambdas:
        scores = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            H_train = H_current[train_idx]
            H_val = H_current[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]
            W = fit_ridge(H_train, y_train, lam)
            pred = predict(H_val, W)
            acc = np.mean(pred == y_val)
            scores.append(acc)
        mean_acc = np.mean(scores)
        if mean_acc > best_score:
            best_score = mean_acc
            best_method = "current"
            best_lambda = lam

    # Evaluate PCA+ReLU
    for lam in lambdas:
        scores = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            H_train, H_val = pca_features(X[train_idx], X[val_idx], n_components=128, use_relu=True)
            y_train = y[train_idx]
            y_val = y[val_idx]
            W = fit_ridge(H_train, y_train, lam)
            pred = predict(H_val, W)
            acc = np.mean(pred == y_val)
            scores.append(acc)
        mean_acc = np.mean(scores)
        if mean_acc > best_score:
            best_score = mean_acc
            best_method = "pca_relu"
            best_lambda = lam

    # Evaluate PCA+linear (no ReLU, just centered projections)
    for lam in lambdas:
        scores = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            H_train, H_val = pca_features(X[train_idx], X[val_idx], n_components=128, use_relu=False)
            y_train = y[train_idx]
            y_val = y[val_idx]
            W = fit_ridge(H_train, y_train, lam)
            pred = predict(H_val, W)
            acc = np.mean(pred == y_val)
            scores.append(acc)
        mean_acc = np.mean(scores)
        if mean_acc > best_score:
            best_score = mean_acc
            best_method = "pca_linear"
            best_lambda = lam

    # Evaluate LDA+PCA+ReLU
    for lam in lambdas:
        scores = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            H_train, H_val = lda_pca_features(X[train_idx], X[val_idx], n_lda=9, n_pca=119, use_relu=True)
            y_train = y[train_idx]
            y_val = y[val_idx]
            W = fit_ridge(H_train, y_train, lam)
            pred = predict(H_val, W)
            acc = np.mean(pred == y_val)
            scores.append(acc)
        mean_acc = np.mean(scores)
        if mean_acc > best_score:
            best_score = mean_acc
            best_method = "lda_pca_relu"
            best_lambda = lam

    # Evaluate LDA+PCA+linear
    for lam in lambdas:
        scores = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            H_train, H_val = lda_pca_features(X[train_idx], X[val_idx], n_lda=9, n_pca=119, use_relu=False)
            y_train = y[train_idx]
            y_val = y[val_idx]
            W = fit_ridge(H_train, y_train, lam)
            pred = predict(H_val, W)
            acc = np.mean(pred == y_val)
            scores.append(acc)
        mean_acc = np.mean(scores)
        if mean_acc > best_score:
            best_score = mean_acc
            best_method = "lda_pca_linear"
            best_lambda = lam

    # Now fit the best method on the full training set
    if best_method == "current":
        # Use current fc1 weights, just refit fc2
        H_full = H_current
        W = fit_ridge(H_full, y, best_lambda)
        # W is (129, 10) because of bias
        new_fc2_weight = W[:-1, :].T  # (10, 128)
        new_fc2_bias = W[-1, :]       # (10,)
        # Keep fc1 as is
        new_fc1_weight = w["fc1.weight"]
        new_fc1_bias = w["fc1.bias"]
    else:
        # Recompute features on full data using the chosen method
        if best_method == "pca_relu":
            H_full, _ = pca_features(X, X, n_components=128, use_relu=True)
            # We need the actual fc1 weights and bias to reproduce these features.
            # pca_features computes mu and P internally. We need to extract them.
            mu = X.mean(axis=0)
            Xc = X - mu
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            P = Vt[:128]
            new_fc1_weight = P.astype(np.float32)
            new_fc1_bias = (-mu @ P.T).astype(np.float32)
        elif best_method == "pca_linear":
            H_full, _ = pca_features(X, X, n_components=128, use_relu=False)
            mu = X.mean(axis=0)
            Xc = X - mu
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            P = Vt[:128]
            new_fc1_weight = P.astype(np.float32)
            new_fc1_bias = (-mu @ P.T).astype(np.float32)
        elif best_method == "lda_pca_relu":
            H_full, _ = lda_pca_features(X, X, n_lda=9, n_pca=119, use_relu=True)
            # Recompute the combined projection
            mu = X.mean(axis=0)
            Xc = X - mu
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            P_pca = Vt[:119]
            # LDA part
            Sw = np.zeros((D, D))
            Sb = np.zeros((D, D))
            overall_mean = Xc.mean(axis=0)
            for c in range(C):
                idx = (y == c)
                if idx.sum() == 0:
                    continue
                Xc_c = Xc[idx]
                mu_c = Xc_c.mean(axis=0)
                Sw += (Xc_c - mu_c).T @ (Xc_c - mu_c)
                n_c = idx.sum()
                diff = mu_c - overall_mean
                Sb += n_c * np.outer(diff, diff)
            Sw_reg = Sw + 1e-3 * np.eye(D)
            try:
                L = np.linalg.cholesky(Sw_reg)
                L_inv = np.linalg.inv(L)
                M = L_inv @ Sb @ L_inv.T
                eigvals, eigvecs = np.linalg.eigh(M)
                idx = np.argsort(eigvals)[::-1]
                eigvecs = eigvecs[:, idx]
                V_lda = L_inv.T @ eigvecs[:, :9]
                V_lda = V_lda.T
            except np.linalg.LinAlgError:
                V_lda = np.zeros((0, D))
            if V_lda.shape[0] > 0:
                P_combined = np.concatenate([V_lda, P_pca], axis=0)
            else:
                P_combined = P_pca
            if P_combined.shape[0] < 128:
                pad = np.zeros((128 - P_combined.shape[0], D))
                P_combined = np.concatenate([P_combined, pad], axis=0)
            elif P_combined.shape[0] > 128:
                P_combined = P_combined[:128]
            norms = np.linalg.norm(P_combined, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            P_combined = P_combined / norms
            new_fc1_weight = P_combined.astype(np.float32)
            new_fc1_bias = (-mu @ P_combined.T).astype(np.float32)
        elif best_method == "lda_pca_linear":
            H_full, _ = lda_pca_features(X, X, n_lda=9, n_pca=119, use_relu=False)
            mu = X.mean(axis=0)
            Xc = X - mu
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            P_pca = Vt[:119]
            Sw = np.zeros((D, D))
            Sb = np.zeros((D, D))
            overall_mean = Xc.mean(axis=0)
            for c in range(C):
                idx = (y == c)
                if idx.sum() == 0:
                    continue
                Xc_c = Xc[idx]
                mu_c = Xc_c.mean(axis=0)
                Sw += (Xc_c - mu_c).T @ (Xc_c - mu_c)
                n_c = idx.sum()
                diff = mu_c - overall_mean
                Sb += n_c * np.outer(diff, diff)
            Sw_reg = Sw + 1e-3 * np.eye(D)
            try:
                L = np.linalg.cholesky(Sw_reg)
                L_inv = np.linalg.inv(L)
                M = L_inv @ Sb @ L_inv.T
                eigvals, eigvecs = np.linalg.eigh(M)
                idx = np.argsort(eigvals)[::-1]
                eigvecs = eigvecs[:, idx]
                V_lda = L_inv.T @ eigvecs[:, :9]
                V_lda = V_lda.T
            except np.linalg.LinAlgError:
                V_lda = np.zeros((0, D))
            if V_lda.shape[0] > 0:
                P_combined = np.concatenate([V_lda, P_pca], axis=0)
            else:
                P_combined = P_pca
            if P_combined.shape[0] < 128:
                pad = np.zeros((128 - P_combined.shape[0], D))
                P_combined = np.concatenate([P_combined, pad], axis=0)
            elif P_combined.shape[0] > 128:
                P_combined = P_combined[:128]
            norms = np.linalg.norm(P_combined, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            P_combined = P_combined / norms
            new_fc1_weight = P_combined.astype(np.float32)
            new_fc1_bias = (-mu @ P_combined.T).astype(np.float32)
        else:
            # Should not happen
            new_fc1_weight = w["fc1.weight"]
            new_fc1_bias = w["fc1.bias"]

        # Now compute features on full data using the new fc1
        H_full = np.maximum(X @ new_fc1_weight.T.astype(np.float64) + new_fc1_bias.astype(np.float64), 0.0)
        # For linear methods, we should not apply ReLU. But we already computed H_full above for linear?
        # Actually, for linear methods, the features are just the linear projections without ReLU.
        # In the code above, for pca_linear and lda_pca_linear, we set use_relu=False, so H_full is just the linear projections.
        # But then we computed H_full again with ReLU. That's a bug. We need to handle linear separately.
        # Let's fix: if best_method ends with "linear", we should not apply ReLU.
        if best_method.endswith("linear"):
            H_full = X @ new_fc1_weight.T.astype(np.float64) + new_fc1_bias.astype(np.float64)
        else:
            H_full = np.maximum(X @ new_fc1_weight.T.astype(np.float64) + new_fc1_bias.astype(np.float64), 0.0)
        # Fit fc2
        W = fit_ridge(H_full, y, best_lambda)
        new_fc2_weight = W[:-1, :].T.astype(np.float32)
        new_fc2_bias = W[-1, :].astype(np.float32)

    # Update weights
    w["fc1.weight"] = new_fc1_weight.astype(np.float32)
    w["fc1.bias"] = new_fc1_bias.astype(np.float32)
    w["fc2.weight"] = new_fc2_weight.astype(np.float32)
    w["fc2.bias"] = new_fc2_bias.astype(np.float32)

    # Ensure finite and within bounds
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)

    diagnostics = {
        "best_method": best_method,
        "best_lambda": best_lambda,
        "cv_accuracy": float(best_score),
        "fc1_weight_mean": float(np.mean(w["fc1.weight"])),
        "fc1_weight_std": float(np.std(w["fc1.weight"])),
        "fc1_bias_mean": float(np.mean(w["fc1.bias"])),
        "fc1_bias_std": float(np.std(w["fc1.bias"])),
        "fc2_weight_mean": float(np.mean(w["fc2.weight"])),
        "fc2_weight_std": float(np.std(w["fc2.weight"])),
        "fc2_bias_mean": float(np.mean(w["fc2.bias"])),
        "fc2_bias_std": float(np.std(w["fc2.bias"])),
    }
    return {"weights": w, "diagnostics": diagnostics}
