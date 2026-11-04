import numpy as np

def edit(w, f):
    # --- load full feedback pool ---
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)  # (2000, 784)
    y = f["y"].astype(np.int64)
    N, D = X.shape
    C = 10

    # --- center data ---
    mean_all = X.mean(axis=0)
    Xc = X - mean_all

    # --- total scatter ---
    St = Xc.T @ Xc  # (784, 784)

    # --- between-class scatter ---
    B = np.zeros((C, D), dtype=np.float64)
    for c in range(C):
        idx = (y == c)
        n_c = np.sum(idx)
        if n_c > 0:
            mean_c = X[idx].mean(axis=0)
            B[c] = np.sqrt(n_c) * (mean_c - mean_all)
    Sb = B.T @ B

    # --- within-class scatter with small ridge ---
    Sw = St - Sb
    eps = 1e-3
    Sw_reg = Sw + eps * np.eye(D)

    # --- LDA directions (top 9) ---
    Sw_inv = np.linalg.inv(Sw_reg)
    M = B @ Sw_inv @ B.T  # (10, 10)
    eigvals, eigvecs = np.linalg.eigh(M)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    U = eigvecs[:, :9]  # (10, 9)
    V_lda = Sw_inv @ B.T @ U  # (784, 9)
    V_lda = V_lda / np.linalg.norm(V_lda, axis=0, keepdims=True)
    V_lda = V_lda.T  # (9, 784)

    # --- PCA directions (top 128) ---
    _, _, Vt_pca = np.linalg.svd(Xc, full_matrices=False)
    V_pca = Vt_pca[:128, :]  # (128, 784)

    # --- combine and orthonormalize ---
    V_combined = np.vstack([V_lda, V_pca])  # (137, 784)
    Q, _ = np.linalg.qr(V_combined.T)       # Q: (784, 137)
    W1 = Q[:, :128].T                       # (128, 784)

    # --- set fc1 ---
    w["fc1.weight"] = W1.astype(np.float32)
    b1 = -(mean_all @ W1.T)                 # (128,)
    w["fc1.bias"] = b1.astype(np.float32)

    # --- hidden features after ReLU ---
    H = np.maximum(0.0, Xc @ W1.T)          # (2000, 128)

    # --- standardize hidden features for better conditioning ---
    H_mean = H.mean(axis=0)
    H_std = H.std(axis=0) + 1e-8
    H_norm = (H - H_mean) / H_std

    H_aug = np.hstack([H_norm, np.ones((N, 1))])  # (2000, 129)
    Y = np.eye(C)[y]                              # (2000, 10)

    # --- closed-form ridge regression ---
    lam = 1e-3
    A = H_aug.T @ H_aug + lam * np.eye(129)
    W2_aug = np.linalg.solve(A, H_aug.T @ Y)      # (129, 10)

    # --- map back to original hidden scale ---
    W2 = W2_aug[:128, :] / H_std[:, None]         # (128, 10)
    b2 = W2_aug[128, :] - (H_mean / H_std) @ W2_aug[:128, :]

    w["fc2.weight"] = W2.T.astype(np.float32)     # (10, 128)
    w["fc2.bias"] = b2.astype(np.float32)         # (10,)

    # --- diagnostics ---
    logits = H_aug @ W2_aug
    pred = np.argmax(logits, axis=1)
    acc = float(np.mean(pred == y))
    diagnostics = {
        "train_accuracy": acc,
        "fc1_weight_std": float(np.std(w["fc1.weight"])),
        "fc2_weight_std": float(np.std(w["fc2.weight"])),
        "lambda": lam,
        "lda_directions": 9,
        "pca_directions": 119,
    }
    return {"weights": w, "diagnostics": diagnostics}
