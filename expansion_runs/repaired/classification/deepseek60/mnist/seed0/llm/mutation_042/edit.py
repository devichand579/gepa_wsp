import numpy as np

def edit(w, f):
    # Extract data
    X_raw = f["x"].reshape(f["x"].shape[0], -1).astype(np.float32)  # (2000, 784)
    y = f["y"].astype(np.int64)  # (2000,)
    H_cur = f["features"].astype(np.float32)  # (2000, 128)
    Z_cur = f["logits"].astype(np.float32)  # (2000, 10)
    P_cur = f["probs"].astype(np.float32)  # (2000, 10)

    # Current fc2 parameters
    W2_cur = w["fc2.weight"].astype(np.float32)  # (10, 128)
    b2_cur = w["fc2.bias"].astype(np.float32)    # (10,)

    # Helper: fit ridge linear classifier on features H with targets T, prior W_prior, b_prior
    def fit_ridge(H, T, W_prior, b_prior, lam):
        N, D = H.shape
        X = np.hstack([H, np.ones((N, 1), dtype=np.float32)])  # (N, D+1)
        W_aug_prior = np.hstack([W_prior, b_prior[:, None]])   # (10, D+1)
        # Solve (X^T X + lam I) W^T = X^T T + lam W_prior^T
        A = X.T @ X + lam * np.eye(D + 1, dtype=np.float32)
        B = X.T @ T + lam * W_aug_prior.T
        W_aug = np.linalg.solve(A, B).T  # (10, D+1)
        return W_aug[:, :-1], W_aug[:, -1]

    # Helper: evaluate accuracy on given H, y, W, b
    def eval_acc(H, y, W, b):
        logits = H @ W.T + b
        pred = np.argmax(logits, axis=1)
        return np.mean(pred == y)

    # Helper: cross-validated accuracy for a given feature matrix H and target construction
    def cv_score(H, y, W_prior, b_prior, lam, margin, n_folds=5):
        N = H.shape[0]
        idx = np.arange(N)
        rng = np.random.RandomState(42)
        rng.shuffle(idx)
        folds = np.array_split(idx, n_folds)
        accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            tr_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            H_tr, y_tr = H[tr_idx], y[tr_idx]
            H_val, y_val = H[val_idx], y[val_idx]
            # Build targets for training: current logits + margin for correct, -margin for others
            # We need current logits for training samples. We can compute them using current W2, b2.
            Z_tr = H_tr @ W_prior.T + b_prior
            T_tr = Z_tr.copy()
            for k in range(len(y_tr)):
                c = y_tr[k]
                T_tr[k, c] += margin
                T_tr[k, :] -= margin
                T_tr[k, c] += margin  # correct class gets +2*margin net? Actually we want +margin for correct, -margin for others.
            # Wait, above is wrong. Let's do it properly:
            T_tr = Z_tr.copy()
            for k in range(len(y_tr)):
                c = y_tr[k]
                T_tr[k, c] += margin
                T_tr[k, np.arange(10) != c] -= margin
            W_new, b_new = fit_ridge(H_tr, T_tr, W_prior, b_prior, lam)
            accs.append(eval_acc(H_val, y_val, W_new, b_new))
        return np.mean(accs)

    # 1. Evaluate current fc2 on full pool
    acc_cur = eval_acc(H_cur, y, W2_cur, b2_cur)

    # 2. Try ridge correction on current features
    best_acc = acc_cur
    best_params = None
    best_W2 = W2_cur
    best_b2 = b2_cur

    # Grid search over lambda and margin
    lambdas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    margins = [0.0, 0.5, 1.0, 2.0, 3.0]
    for lam in lambdas:
        for margin in margins:
            if margin == 0.0:
                # Just fit to current logits (no margin change)
                # Use CV to estimate
                score = cv_score(H_cur, y, W2_cur, b2_cur, lam, margin)
                if score > best_acc:
                    best_acc = score
                    best_params = (lam, margin)
                    # Fit on full pool
                    Z_full = H_cur @ W2_cur.T + b2_cur
                    T_full = Z_full.copy()
                    W_new, b_new = fit_ridge(H_cur, T_full, W2_cur, b2_cur, lam)
                    best_W2, best_b2 = W_new, b_new
            else:
                score = cv_score(H_cur, y, W2_cur, b2_cur, lam, margin)
                if score > best_acc:
                    best_acc = score
                    best_params = (lam, margin)
                    Z_full = H_cur @ W2_cur.T + b2_cur
                    T_full = Z_full.copy()
                    for k in range(len(y)):
                        c = y[k]
                        T_full[k, c] += margin
                        T_full[k, np.arange(10) != c] -= margin
                    W_new, b_new = fit_ridge(H_cur, T_full, W2_cur, b2_cur, lam)
                    best_W2, best_b2 = W_new, b_new

    # 3. Try PCA-based fc1 replacement
    # Center raw pixels
    X_mean = X_raw.mean(axis=0)
    X_centered = X_raw - X_mean
    # Compute PCA via SVD (economy)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # Vt is (min(N,784), 784). Take top 128 components.
    n_comp = min(128, Vt.shape[0])
    W1_pca = Vt[:n_comp, :].astype(np.float32)  # (128, 784)
    b1_pca = (-X_mean @ W1_pca.T).astype(np.float32)  # (128,)
    # Compute new features
    H_pca = np.maximum(X_raw @ W1_pca.T + b1_pca, 0.0).astype(np.float32)
    # Fit fc2 on H_pca using ridge with current fc2 as prior? But current fc2 is for 128 features, so shape matches.
    # We can use the same W2_cur, b2_cur as prior, but they were trained on different features. Better to use zero prior or random.
    # Let's use zero prior for PCA features, or we can use the current fc2 as prior but it might not be meaningful.
    # We'll just fit from scratch with small ridge.
    W2_pca = np.zeros((10, 128), dtype=np.float32)
    b2_pca = np.zeros(10, dtype=np.float32)
    # Evaluate CV for PCA features
    best_acc_pca = 0.0
    best_params_pca = None
    best_W2_pca = W2_pca
    best_b2_pca = b2_pca
    for lam in [0.001, 0.01, 0.1, 1.0, 10.0]:
        # For PCA, we can just fit to one-hot targets with margin? Or just one-hot.
        # Let's use one-hot targets with a small margin.
        # We'll do CV.
        N = H_pca.shape[0]
        idx = np.arange(N)
        rng = np.random.RandomState(42)
        rng.shuffle(idx)
        folds = np.array_split(idx, 5)
        accs = []
        for i in range(5):
            val_idx = folds[i]
            tr_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            H_tr, y_tr = H_pca[tr_idx], y[tr_idx]
            H_val, y_val = H_pca[val_idx], y[val_idx]
            # One-hot targets
            T_tr = np.zeros((len(y_tr), 10), dtype=np.float32)
            T_tr[np.arange(len(y_tr)), y_tr] = 1.0
            W_new, b_new = fit_ridge(H_tr, T_tr, W2_pca, b2_pca, lam)
            accs.append(eval_acc(H_val, y_val, W_new, b_new))
        score = np.mean(accs)
        if score > best_acc_pca:
            best_acc_pca = score
            best_params_pca = lam
            # Fit on full pool
            T_full = np.zeros((N, 10), dtype=np.float32)
            T_full[np.arange(N), y] = 1.0
            W_new, b_new = fit_ridge(H_pca, T_full, W2_pca, b2_pca, lam)
            best_W2_pca, best_b2_pca = W_new, b_new

    # 4. Choose best between current-feature correction and PCA-based
    if best_acc_pca > best_acc:
        # Use PCA-based fc1 and fc2
        w["fc1.weight"] = W1_pca
        w["fc1.bias"] = b1_pca
        w["fc2.weight"] = best_W2_pca
        w["fc2.bias"] = best_b2_pca
        method = "pca_fc1_ridge_fc2"
        final_acc = best_acc_pca
    else:
        # Use corrected fc2 on current features
        w["fc2.weight"] = best_W2
        w["fc2.bias"] = best_b2
        method = "ridge_correction_fc2"
        final_acc = best_acc

    # Ensure float32 and finite
    for k in w:
        w[k] = np.asarray(w[k], dtype=np.float32)
        w[k] = np.nan_to_num(w[k], nan=0.0, posinf=1000.0, neginf=-1000.0)
        w[k] = np.clip(w[k], -1000.0, 1000.0)

    diagnostics = {
        "method": method,
        "cv_accuracy_current_features": float(best_acc),
        "cv_accuracy_pca_features": float(best_acc_pca),
        "chosen_cv_accuracy": float(final_acc),
        "best_params_current": best_params,
        "best_params_pca": best_params_pca,
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
        "fc1_weight_norm": float(np.linalg.norm(w["fc1.weight"])),
        "fc1_bias_norm": float(np.linalg.norm(w["fc1.bias"])),
    }

    return {"weights": w, "diagnostics": diagnostics}
