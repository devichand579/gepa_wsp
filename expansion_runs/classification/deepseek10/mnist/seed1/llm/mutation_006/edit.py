import numpy as np

def edit(w, f):
    # ------------------------------------------------------------------
    # Load permitted feedback
    # ------------------------------------------------------------------
    x = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)   # (N, 784)
    y = f["y"].astype(np.int64)                                  # (N,)
    N = x.shape[0]
    C = 10

    # Current final hidden features (after fc1 + ReLU)
    H_cur = f["features"].astype(np.float64)                     # (N, 128)

    # ------------------------------------------------------------------
    # Closed-form linear classifiers
    # ------------------------------------------------------------------
    def fit_ridge(H, Y, lam):
        H_mean = H.mean(axis=0)
        Y_mean = Y.mean(axis=0)
        Hc = H - H_mean
        Yc = Y - Y_mean
        D = H.shape[1]
        A = Hc.T @ Hc + lam * np.eye(D)
        B = Hc.T @ Yc
        W = np.linalg.solve(A, B)          # (D, C)
        bias = Y_mean - H_mean @ W         # (C,)
        return W, bias

    def fit_lda(H, y, lam):
        N, D = H.shape
        mu = np.zeros((C, D))
        for c in range(C):
            mu[c] = H[y == c].mean(axis=0)
        S = np.zeros((D, D))
        for c in range(C):
            Hc = H[y == c] - mu[c]
            S += Hc.T @ Hc
        S /= (N - C)
        S_reg = S + lam * np.eye(D)
        W = np.linalg.solve(S_reg, mu.T)   # (D, C)
        bias = np.zeros(C)
        for c in range(C):
            bias[c] = -0.5 * mu[c] @ W[:, c] + np.log(np.sum(y == c) / N)
        return W, bias

    def cv_score(H, y, method, param):
        n_folds = 5
        rng = np.random.RandomState(42)
        indices = np.arange(N)
        fold_indices = [[] for _ in range(n_folds)]
        for c in range(C):
            idx_c = indices[y == c]
            rng.shuffle(idx_c)
            for i, idx in enumerate(idx_c):
                fold_indices[i % n_folds].append(idx)
        fold_indices = [np.array(fi) for fi in fold_indices]

        accs = []
        for i in range(n_folds):
            val_idx = fold_indices[i]
            train_idx = np.concatenate([fold_indices[j] for j in range(n_folds) if j != i])
            H_train, y_train = H[train_idx], y[train_idx]
            H_val, y_val = H[val_idx], y[val_idx]
            Y_train = np.zeros((len(y_train), C))
            Y_train[np.arange(len(y_train)), y_train] = 1.0

            if method == "ridge":
                W, bias = fit_ridge(H_train, Y_train, param)
            else:
                W, bias = fit_lda(H_train, y_train, param)

            scores = H_val @ W + bias
            pred = np.argmax(scores, axis=1)
            accs.append(np.mean(pred == y_val))
        return float(np.mean(accs))

    # ------------------------------------------------------------------
    # Candidate feature sets
    # ------------------------------------------------------------------
    candidates = []

    # 1) Current features, ridge
    for lam in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        acc = cv_score(H_cur, y, "ridge", lam)
        candidates.append(("cur_ridge", lam, H_cur, acc))

    # 2) Current features, LDA
    for lam in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        acc = cv_score(H_cur, y, "lda", lam)
        candidates.append(("cur_lda", lam, H_cur, acc))

    # 3) Global ReLU-threshold shifts on current fc1
    pre = x @ w["fc1.weight"].T.astype(np.float64) + w["fc1.bias"].astype(np.float64)
    for c_shift in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
        H_shift = np.maximum(0.0, pre + c_shift)
        for lam in [0.01, 0.1, 1.0, 10.0, 100.0]:
            acc = cv_score(H_shift, y, "ridge", lam)
            candidates.append((f"shift_{c_shift}", lam, H_shift, acc))

    # 4) PCA features (centered), with and without whitening
    x_mean = x.mean(axis=0)
    xc = x - x_mean
    # SVD of centered data
    U, S, Vt = np.linalg.svd(xc, full_matrices=False)
    components = Vt[:128]                      # (128, 784)
    # Centered PCA features: (x - x_mean) @ components.T
    H_pca_cent = xc @ components.T
    H_pca_cent = np.maximum(0.0, H_pca_cent)
    for lam in [0.01, 0.1, 1.0, 10.0, 100.0]:
        acc = cv_score(H_pca_cent, y, "ridge", lam)
        candidates.append(("pca_cent", lam, H_pca_cent, acc))

    # Whitened PCA features
    eps = 1e-6
    H_pca_white = xc @ (components.T / np.sqrt(S[:128] + eps))
    H_pca_white = np.maximum(0.0, H_pca_white)
    for lam in [0.01, 0.1, 1.0, 10.0, 100.0]:
        acc = cv_score(H_pca_white, y, "ridge", lam)
        candidates.append(("pca_white", lam, H_pca_white, acc))

    # ------------------------------------------------------------------
    # Pick best candidate by CV accuracy
    # ------------------------------------------------------------------
    best = max(candidates, key=lambda t: t[3])
    best_name, best_param, best_H, best_acc = best

    # Refit on full pool using the best method/param
    if "ridge" in best_name or best_name.startswith("shift") or best_name.startswith("pca"):
        method = "ridge"
    else:
        method = "lda"

    Y_full = np.zeros((N, C))
    Y_full[np.arange(N), y] = 1.0

    if method == "ridge":
        W, bias = fit_ridge(best_H, Y_full, best_param)
    else:
        W, bias = fit_lda(best_H, y, best_param)

    # ------------------------------------------------------------------
    # Write back weights
    # ------------------------------------------------------------------
    w_out = {k: v.copy() for k, v in w.items()}

    # If best candidate uses a different feature extractor, update fc1 accordingly.
    if best_name.startswith("shift"):
        c_shift = float(best_name.split("_")[1])
        w_out["fc1.bias"] = (w["fc1.bias"].astype(np.float64) + c_shift).astype(np.float32)
    elif best_name.startswith("pca_cent"):
        w_out["fc1.weight"] = components.astype(np.float32)
        w_out["fc1.bias"] = (-x_mean @ components.T).astype(np.float32)
    elif best_name.startswith("pca_white"):
        W_pca = (components.T / np.sqrt(S[:128] + eps)).T  # (128, 784)
        w_out["fc1.weight"] = W_pca.astype(np.float32)
        w_out["fc1.bias"] = (-x_mean @ W_pca.T).astype(np.float32)
    # else: keep fc1 unchanged

    # fc2 is always replaced by the closed-form solution
    w_out["fc2.weight"] = W.T.astype(np.float32)   # (10, 128)
    w_out["fc2.bias"] = bias.astype(np.float32)    # (10,)

    # Sanity: finite, bounded
    for k in w_out:
        w_out[k] = np.nan_to_num(w_out[k], nan=0.0, posinf=1000.0, neginf=-1000.0)
        w_out[k] = np.clip(w_out[k], -1000.0, 1000.0).astype(np.float32)

    diagnostics = {
        "best_candidate": best_name,
        "best_param": float(best_param),
        "cv_accuracy": float(best_acc),
        "fc2_weight_norm": float(np.linalg.norm(w_out["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w_out["fc2.bias"])),
        "fc1_bias_norm": float(np.linalg.norm(w_out["fc1.bias"])),
    }

    return {"weights": w_out, "diagnostics": diagnostics}
