import numpy as np

def edit(w, f):
    # Extract full training pool
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float32)  # (N, 784)
    y = f["y"].astype(np.int64)                                 # (N,)
    N = X.shape[0]
    C = 10

    # Global and class means
    mean_X = X.mean(axis=0)
    means = np.zeros((C, 784), dtype=np.float32)
    counts = np.zeros(C, dtype=np.float32)
    for c in range(C):
        mask = (y == c)
        counts[c] = mask.sum()
        means[c] = X[mask].mean(axis=0)

    candidates = []

    # 1. Class-mean directions
    for c in range(C):
        candidates.append(means[c] - mean_X)

    # 2. LDA directions (Sw^-1 * class-mean differences)
    Sw = np.zeros((784, 784), dtype=np.float32)
    for c in range(C):
        Xc = X[y == c]
        Xc_centered = Xc - means[c]
        Sw += Xc_centered.T @ Xc_centered
    Sw /= (N - C)
    Sw_reg = Sw + 1e-3 * np.eye(784, dtype=np.float32)
    diff = (means - mean_X).T  # (784, 10)
    lda_dirs = np.linalg.solve(Sw_reg, diff).T  # (10, 784)
    for c in range(C):
        candidates.append(lda_dirs[c])

    # 3. PCA directions (top 128)
    X_centered = X - mean_X
    cov = X_centered.T @ X_centered / (N - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    pca_dirs = eigvecs[:, -128:].T  # (128, 784)
    for i in range(128):
        candidates.append(pca_dirs[i])

    # 4. Ridge-regression directions
    Y_onehot = np.zeros((N, C), dtype=np.float32)
    Y_onehot[np.arange(N), y] = 1.0
    XTY = X_centered.T @ Y_onehot + np.outer(mean_X, counts)
    XTX = (N - 1) * cov + N * np.outer(mean_X, mean_X)
    for lam in [1e-4, 1e-3, 1e-2, 1e-1]:
        W_ridge = np.linalg.solve(XTX + lam * np.eye(784, dtype=np.float32), XTY)
        for c in range(C):
            candidates.append(W_ridge[:, c])

    # 5. Random directions
    rng = np.random.RandomState(0)
    for _ in range(200):
        candidates.append(rng.randn(784).astype(np.float32))

    # 6. Current fc1 rows
    W1_cur = w["fc1.weight"].copy()
    for i in range(128):
        candidates.append(W1_cur[i])

    C_mat = np.array(candidates, dtype=np.float32)  # (M, 784)
    M = C_mat.shape[0]

    # Project and center
    S = X @ C_mat.T  # (N, M)
    S_centered = S - S.mean(axis=0, keepdims=True)
    H_cand = np.maximum(0, S_centered)  # (N, M)

    # ANOVA F-value per candidate
    H_mean = H_cand.mean(axis=0)
    H_class_means = np.zeros((C, M), dtype=np.float32)
    for c in range(C):
        H_class_means[c] = H_cand[y == c].mean(axis=0)
    SSB = np.sum(counts[:, None] * (H_class_means - H_mean) ** 2, axis=0)
    H_total_var = H_cand.var(axis=0) * N
    SSW = H_total_var - SSB
    F = (SSB / (C - 1)) / (SSW / (N - C) + 1e-8)
    top_idx = np.argsort(F)[-128:]

    # New fc1
    W1_new = C_mat[top_idx]  # (128, 784)
    b1_new = -S[:, top_idx].mean(axis=0)  # (128,)

    # New hidden features
    H_new = np.maximum(0, X @ W1_new.T + b1_new)  # (N, 128)

    # Standardize hidden features for stable LDA/ridge
    H_mean = H_new.mean(axis=0)
    H_std = H_new.std(axis=0) + 1e-8
    H_std_feat = (H_new - H_mean) / H_std

    # Fit fc2 via LDA with shrinkage
    H_means = np.zeros((C, 128), dtype=np.float32)
    for c in range(C):
        H_means[c] = H_std_feat[y == c].mean(axis=0)
    H_global_mean = H_std_feat.mean(axis=0)

    Sw_H = np.zeros((128, 128), dtype=np.float32)
    for c in range(C):
        Hc = H_std_feat[y == c]
        Hc_centered = Hc - H_means[c]
        Sw_H += Hc_centered.T @ Hc_centered
    Sw_H /= (N - C)
    shrink = 1e-2 * np.trace(Sw_H) / 128
    Sw_H_reg = Sw_H + shrink * np.eye(128, dtype=np.float32)
    diff_H = (H_means - H_global_mean).T
    W2_lda = np.linalg.solve(Sw_H_reg, diff_H).T  # (10, 128)
    b2_lda = -W2_lda @ H_global_mean

    # Also try ridge on standardized features
    H_aug = np.hstack([H_std_feat, np.ones((N, 1), dtype=np.float32)])
    best_acc = -1.0
    best_W2 = None
    best_b2 = None
    best_name = None

    def train_acc(W2, b2):
        logits = H_std_feat @ W2.T + b2
        pred = np.argmax(logits, axis=1)
        return np.mean(pred == y)

    acc_lda = train_acc(W2_lda, b2_lda)
    if acc_lda > best_acc:
        best_acc = acc_lda
        best_W2 = W2_lda
        best_b2 = b2_lda
        best_name = "lda"

    for lam in [1e-4, 1e-3, 1e-2, 1e-1]:
        A = H_aug.T @ H_aug + lam * np.eye(129, dtype=np.float32)
        B = H_aug.T @ Y_onehot
        W_aug = np.linalg.solve(A, B)
        W2_r = W_aug[:128].T
        b2_r = W_aug[128]
        acc = train_acc(W2_r, b2_r)
        if acc > best_acc:
            best_acc = acc
            best_W2 = W2_r
            best_b2 = b2_r
            best_name = f"ridge_{lam}"

    # Convert fc2 back to original feature scale
    # H_std = (H_new - H_mean) / H_std
    # logits = H_std @ W2.T + b2 = (H_new - H_mean)/H_std @ W2.T + b2
    #        = H_new @ (W2 / H_std).T + (b2 - (W2 / H_std) @ H_mean)
    W2_final = best_W2 / H_std[None, :]
    b2_final = best_b2 - W2_final @ H_mean

    # Write back
    w["fc1.weight"] = W1_new.astype(np.float32)
    w["fc1.bias"] = b1_new.astype(np.float32)
    w["fc2.weight"] = W2_final.astype(np.float32)
    w["fc2.bias"] = b2_final.astype(np.float32)

    # Clip and sanitize
    for k in w:
        if isinstance(w[k], np.ndarray):
            w[k] = np.clip(w[k], -1000, 1000).astype(np.float32)
            w[k] = np.nan_to_num(w[k], nan=0.0, posinf=1000.0, neginf=-1000.0)

    diagnostics = {
        "chosen": best_name,
        "train_accuracy": float(best_acc),
        "num_candidates": int(M),
        "selected_indices": top_idx.tolist()[:10],
        "fc1_weight_mean": float(w["fc1.weight"].mean()),
        "fc1_weight_std": float(w["fc1.weight"].std()),
        "fc2_weight_mean": float(w["fc2.weight"].mean()),
        "fc2_weight_std": float(w["fc2.weight"].std()),
        "fc2_bias_mean": float(w["fc2.bias"].mean()),
        "fc2_bias_std": float(w["fc2.bias"].std()),
    }

    return {"weights": w, "diagnostics": diagnostics}
