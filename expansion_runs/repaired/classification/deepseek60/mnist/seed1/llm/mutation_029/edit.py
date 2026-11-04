import numpy as np

def edit(w, f):
    # Extract training data
    X = f["x"].reshape(-1, 784).astype(np.float32)  # (2000, 784)
    y = f["y"].astype(np.int64)                     # (2000,)
    F_old = f["features"].astype(np.float32)        # (2000, 128)
    fc2_w = w["fc2.weight"].astype(np.float32)      # (10, 128)

    # Identify least important hidden units by fc2 weight norm
    importance = np.linalg.norm(fc2_w, axis=0)
    idx_sorted = np.argsort(importance)             # ascending
    K = 38                                          # replace 38 units (19 directions * 2)
    replace_idx = idx_sorted[:K]
    keep_idx = idx_sorted[K:]

    # --- Compute raw-pixel directions ---
    n, d = X.shape
    classes = np.arange(10)
    counts = np.array([(y == c).sum() for c in classes], dtype=np.float32)
    means = np.zeros((10, d), dtype=np.float32)
    for c in classes:
        means[c] = X[y == c].mean(axis=0)
    mean_all = X.mean(axis=0)

    # Within-class scatter with shrinkage
    Sw = np.zeros((d, d), dtype=np.float32)
    for c in classes:
        Xc = X[y == c] - means[c]
        Sw += Xc.T @ Xc
    alpha_raw = 0.1
    Sw_shrunk = (1 - alpha_raw) * Sw + alpha_raw * (np.trace(Sw) / d) * np.eye(d, dtype=np.float32)
    Sw_inv = np.linalg.inv(Sw_shrunk)

    # Ridge class-score directions
    X_centered = X - mean_all
    Y_onehot = np.eye(10, dtype=np.float32)[y]
    Y_centered = Y_onehot - Y_onehot.mean(axis=0)
    lam = 1.0
    ridge_dirs = np.zeros((10, d), dtype=np.float32)
    for c in classes:
        A = X_centered.T @ X_centered + lam * np.eye(d, dtype=np.float32)
        b = X_centered.T @ Y_centered[:, c]
        ridge_dirs[c] = np.linalg.solve(A, b)

    # LDA directions via 10x10 eigenproblem
    M = means - mean_all                     # (10, 784)
    N = np.diag(counts)                      # (10, 10)
    B = M @ Sw_inv @ M.T                     # (10, 10)
    C = B @ N                                # (10, 10)
    eigvals, eigvecs = np.linalg.eig(C)
    order = np.argsort(eigvals.real)[::-1]
    lda_dirs = []
    for i in order[:9]:
        u = eigvecs[:, i].real.astype(np.float32)
        v = Sw_inv @ M.T @ u
        v = v / (np.linalg.norm(v) + 1e-12)
        lda_dirs.append(v)
    lda_dirs = np.array(lda_dirs, dtype=np.float32)  # (9, 784)

    # Combine directions: 10 ridge + 9 LDA = 19
    all_dirs = np.vstack([ridge_dirs, lda_dirs])     # (19, 784)

    # Build new fc1 weights and biases
    W1_new = w["fc1.weight"].astype(np.float32).copy()
    b1_new = w["fc1.bias"].astype(np.float32).copy()

    # For each direction, create +w and -w units
    new_units = []
    for w_dir in all_dirs:
        # + direction: weight = w_dir, bias = -w_dir @ mean_all
        new_units.append((w_dir, -w_dir @ mean_all))
        # - direction: weight = -w_dir, bias = w_dir @ mean_all
        new_units.append((-w_dir, w_dir @ mean_all))

    # Assign to replace_idx
    for i, (w_dir, b_dir) in enumerate(new_units):
        idx = replace_idx[i]
        W1_new[idx] = w_dir
        b1_new[idx] = b_dir

    # Compute new hidden features for all training examples
    F_new = np.maximum(X @ W1_new.T + b1_new, 0.0)  # (2000, 128)

    # --- Fit fc2 via regularized LDA on new features ---
    n_feat = F_new.shape[1]
    means_F = np.zeros((10, n_feat), dtype=np.float32)
    for c in classes:
        means_F[c] = F_new[y == c].mean(axis=0)
    mean_all_F = F_new.mean(axis=0)

    Sw_F = np.zeros((n_feat, n_feat), dtype=np.float32)
    for c in classes:
        Xc = F_new[y == c] - means_F[c]
        Sw_F += Xc.T @ Xc
    alpha_F = 0.1
    Sw_F_shrunk = (1 - alpha_F) * Sw_F + alpha_F * (np.trace(Sw_F) / n_feat) * np.eye(n_feat, dtype=np.float32)

    W2 = np.zeros((10, n_feat), dtype=np.float32)
    b2 = np.zeros(10, dtype=np.float32)
    for c in classes:
        w_c = np.linalg.solve(Sw_F_shrunk, means_F[c])
        W2[c] = w_c
        b2[c] = -0.5 * (means_F[c] @ w_c)

    # Write back to w
    w["fc1.weight"] = W1_new.astype(np.float32)
    w["fc1.bias"] = b1_new.astype(np.float32)
    w["fc2.weight"] = W2.astype(np.float32)
    w["fc2.bias"] = b2.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "replaced_units": int(K),
        "ridge_dirs": 10,
        "lda_dirs": 9,
        "fc1_weight_mean": float(w["fc1.weight"].mean()),
        "fc1_weight_std": float(w["fc1.weight"].std()),
        "fc2_weight_mean": float(w["fc2.weight"].mean()),
        "fc2_weight_std": float(w["fc2.weight"].std()),
        "fc2_bias_mean": float(w["fc2.bias"].mean()),
        "fc2_bias_std": float(w["fc2.bias"].std()),
    }
    return {"weights": w, "diagnostics": diagnostics}
