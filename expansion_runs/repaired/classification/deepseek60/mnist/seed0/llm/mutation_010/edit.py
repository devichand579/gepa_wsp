import numpy as np

def edit(w, f):
    # ---------- data ----------
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float32)
    y = f["y"].astype(np.int64)
    N, D = X.shape
    C = 10

    mu = X.mean(axis=0)
    Mu = np.zeros((C, D), dtype=np.float32)
    counts = np.zeros(C, dtype=np.int32)
    for c in range(C):
        idx = (y == c)
        counts[c] = int(np.sum(idx))
        if counts[c] > 0:
            Mu[c] = X[idx].mean(axis=0)

    Xc = X - mu

    # ---------- global PCA ----------
    # SVD on centered data; keep enough components
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pca_dirs = Vt.T.astype(np.float32)  # D x min(N,D)

    # ---------- within-class scatter ----------
    S_w = np.zeros((D, D), dtype=np.float64)
    for c in range(C):
        idx = (y == c)
        if counts[c] > 0:
            Xc_c = X[idx] - Mu[c]
            S_w += Xc_c.T @ Xc_c
    S_w = S_w.astype(np.float32)

    # ---------- between-class scatter ----------
    M = Mu - mu
    S_b = M.T @ (counts[:, None] * M)
    S_b = S_b.astype(np.float32)

    # ---------- LDA directions ----------
    eps = 1e-3 * np.trace(S_w) / D
    S_w_reg = S_w + eps * np.eye(D, dtype=np.float32)
    try:
        L = np.linalg.cholesky(S_w_reg)
        Linv = np.linalg.inv(L)
        A = Linv @ S_b @ Linv.T
        A = (A + A.T) * 0.5
        eigvals, eigvecs = np.linalg.eigh(A)
        order = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order]
        # take up to 9 non-zero LDA directions
        lda_dirs = []
        for i in range(min(9, eigvecs.shape[1])):
            v = Linv.T @ eigvecs[:, i]
            nv = np.linalg.norm(v)
            if nv > 1e-8:
                lda_dirs.append(v / nv)
        lda_dirs = np.array(lda_dirs, dtype=np.float32) if lda_dirs else np.zeros((0, D), dtype=np.float32)
    except Exception:
        lda_dirs = np.zeros((0, D), dtype=np.float32)

    # ---------- within-class PCA ----------
    R = X - Mu[y]
    U_r, S_r, Vt_r = np.linalg.svd(R, full_matrices=False)
    within_pca_dirs = Vt_r.T.astype(np.float32)  # D x min(N,D)

    # ---------- candidate fc1 matrices ----------
    candidates = []

    # 1. current fc1
    W1_cur = w["fc1.weight"].astype(np.float32).copy()
    b1_cur = w["fc1.bias"].astype(np.float32).copy()
    candidates.append(("current", W1_cur, b1_cur))

    # helper to build normalized directions
    def norm_rows(A):
        A = A.astype(np.float32).copy()
        n = np.linalg.norm(A, axis=1, keepdims=True)
        n[n < 1e-8] = 1.0
        return A / n

    # 2. class means + global PCA
    cm = norm_rows(Mu - mu)
    W1 = np.vstack([cm, pca_dirs[:, :118].T])
    b1 = -W1 @ mu
    candidates.append(("cm+pca", W1, b1))

    # 3. class means + within-class PCA
    W1 = np.vstack([cm, within_pca_dirs[:, :118].T])
    b1 = -W1 @ mu
    candidates.append(("cm+wpca", W1, b1))

    # 4. LDA + within-class PCA
    if lda_dirs.shape[0] > 0:
        W1 = np.vstack([lda_dirs, within_pca_dirs[:, :128 - lda_dirs.shape[0]].T])
        b1 = -W1 @ mu
        candidates.append(("lda+wpca", W1, b1))

    # 5. LDA + global PCA
    if lda_dirs.shape[0] > 0:
        W1 = np.vstack([lda_dirs, pca_dirs[:, :128 - lda_dirs.shape[0]].T])
        b1 = -W1 @ mu
        candidates.append(("lda+pca", W1, b1))

    # 6. class means + LDA + within-class PCA
    if lda_dirs.shape[0] > 0:
        W1 = np.vstack([cm, lda_dirs, within_pca_dirs[:, :128 - cm.shape[0] - lda_dirs.shape[0]].T])
        b1 = -W1 @ mu
        candidates.append(("cm+lda+wpca", W1, b1))

    # 7. class means + LDA + global PCA
    if lda_dirs.shape[0] > 0:
        W1 = np.vstack([cm, lda_dirs, pca_dirs[:, :128 - cm.shape[0] - lda_dirs.shape[0]].T])
        b1 = -W1 @ mu
        candidates.append(("cm+lda+pca", W1, b1))

    # 8. global PCA only
    W1 = pca_dirs[:, :128].T
    b1 = -W1 @ mu
    candidates.append(("pca", W1, b1))

    # 9. within-class PCA only
    W1 = within_pca_dirs[:, :128].T
    b1 = -W1 @ mu
    candidates.append(("wpca", W1, b1))

    # 10. random projections (a few seeds)
    for seed in range(3):
        rng = np.random.RandomState(seed)
        W1 = rng.randn(128, D).astype(np.float32)
        W1 = norm_rows(W1)
        b1 = -W1 @ mu
        candidates.append((f"rand{seed}", W1, b1))

    # ---------- ridge CV for fc2 ----------
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    n_folds = 5
    # deterministic folds
    fold_size = N // n_folds
    indices = np.arange(N)
    # use a fixed permutation for reproducibility
    rng = np.random.RandomState(123)
    rng.shuffle(indices)
    folds = []
    for k in range(n_folds):
        start = k * fold_size
        end = (k + 1) * fold_size if k < n_folds - 1 else N
        val_idx = indices[start:end]
        train_idx = np.setdiff1d(indices, val_idx, assume_unique=False)
        folds.append((train_idx, val_idx))

    best_candidate = None
    best_cv_acc = -1.0
    best_lambda = None
    best_W1 = None
    best_b1 = None

    for name, W1, b1 in candidates:
        # compute features for all data
        H = np.maximum(0.0, X @ W1.T + b1).astype(np.float32)
        # one-hot targets
        Y = np.zeros((N, C), dtype=np.float32)
        Y[np.arange(N), y] = 1.0

        # CV over lambdas
        cv_scores = []
        for lam in lambdas:
            accs = []
            for train_idx, val_idx in folds:
                H_tr = H[train_idx]
                Y_tr = Y[train_idx]
                H_val = H[val_idx]
                Y_val = Y[val_idx]

                # augment with bias column
                H_tr_aug = np.hstack([H_tr, np.ones((H_tr.shape[0], 1), dtype=np.float32)])
                H_val_aug = np.hstack([H_val, np.ones((H_val.shape[0], 1), dtype=np.float32)])

                # ridge normal equations
                A = H_tr_aug.T @ H_tr_aug + lam * np.eye(H_tr_aug.shape[1], dtype=np.float32)
                B = H_tr_aug.T @ Y_tr
                sol = np.linalg.solve(A, B)
                pred = H_val_aug @ sol
                pred_label = np.argmax(pred, axis=1)
                acc = np.mean(pred_label == y[val_idx])
                accs.append(acc)
            cv_scores.append(np.mean(accs))

        best_lam_idx = int(np.argmax(cv_scores))
        best_lam = lambdas[best_lam_idx]
        best_acc = cv_scores[best_lam_idx]

        if best_acc > best_cv_acc:
            best_cv_acc = best_acc
            best_lambda = best_lam
            best_candidate = name
            best_W1 = W1
            best_b1 = b1

    # ---------- final fit on full data ----------
    W1 = best_W1
    b1 = best_b1
    H = np.maximum(0.0, X @ W1.T + b1).astype(np.float32)
    Y = np.zeros((N, C), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    H_aug = np.hstack([H, np.ones((N, 1), dtype=np.float32)])
    A = H_aug.T @ H_aug + best_lambda * np.eye(H_aug.shape[1], dtype=np.float32)
    B = H_aug.T @ Y
    sol = np.linalg.solve(A, B)
    W2 = sol[:128].T.astype(np.float32)
    b2 = sol[128].astype(np.float32)

    # ---------- write back ----------
    w["fc1.weight"] = W1.astype(np.float32)
    w["fc1.bias"] = b1.astype(np.float32)
    w["fc2.weight"] = W2.astype(np.float32)
    w["fc2.bias"] = b2.astype(np.float32)

    # clip to safe range
    for k in ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)

    diagnostics = {
        "chosen_candidate": best_candidate,
        "chosen_lambda": float(best_lambda),
        "cv_accuracy": float(best_cv_acc),
        "fc1_weight_norm": float(np.linalg.norm(w["fc1.weight"])),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
    }
    return {"weights": w, "diagnostics": diagnostics}
