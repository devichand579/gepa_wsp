import numpy as np

def edit(w, f):
    # Extract data
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float32)  # [2000, 784]
    y = f["y"].astype(np.int64)                                 # [2000]
    N, D = X.shape
    C = 10

    # Current hidden features (already computed)
    H_cur = f["features"].astype(np.float32)                    # [2000, 128]

    # Helper: LDA classifier on given features
    def fit_lda(H, y):
        N, D = H.shape
        classes = np.unique(y)
        means = np.zeros((C, D), dtype=np.float32)
        priors = np.zeros(C, dtype=np.float32)
        for c in range(C):
            idx = (y == c)
            means[c] = H[idx].mean(axis=0)
            priors[c] = idx.mean()
        # Pooled within-class covariance
        Sw = np.zeros((D, D), dtype=np.float64)
        for c in range(C):
            idx = (y == c)
            diff = H[idx] - means[c]
            Sw += diff.T @ diff
        Sw /= (N - C)
        # Regularize slightly for numerical stability
        Sw += np.eye(D, dtype=np.float64) * 1e-4
        Sw_inv = np.linalg.inv(Sw)
        # LDA weights and biases
        W = (Sw_inv @ means.T).T.astype(np.float32)             # [C, D]
        b = (-0.5 * np.sum(means * (Sw_inv @ means.T).T, axis=1) + np.log(priors)).astype(np.float32)
        return W, b

    # Helper: evaluate accuracy of a linear classifier on features
    def eval_lda(H, y, W, b):
        logits = H @ W.T + b
        pred = np.argmax(logits, axis=1)
        return (pred == y).mean()

    # 1) Baseline: refit fc2 with LDA on current features
    W2_lda, b2_lda = fit_lda(H_cur, y)
    acc_cur = eval_lda(H_cur, y, W2_lda, b2_lda)

    # 2) Try replacing some fc1 units with discriminative directions
    # Generate candidate directions from raw pixels
    # a) LDA directions on raw pixels
    means_raw = np.zeros((C, D), dtype=np.float32)
    for c in range(C):
        means_raw[c] = X[y == c].mean(axis=0)
    global_mean = X.mean(axis=0)
    # Between-class scatter
    Sb = np.zeros((D, D), dtype=np.float64)
    for c in range(C):
        diff = (means_raw[c] - global_mean).astype(np.float64)
        Sb += (y == c).sum() * np.outer(diff, diff)
    # Within-class scatter
    Sw_raw = np.zeros((D, D), dtype=np.float64)
    for c in range(C):
        idx = (y == c)
        diff = X[idx] - means_raw[c]
        Sw_raw += diff.T @ diff
    Sw_raw /= (N - C)
    Sw_raw += np.eye(D, dtype=np.float64) * 1e-2
    # LDA directions: eigenvectors of Sw^{-1} Sb
    M = np.linalg.solve(Sw_raw, Sb)
    eigvals, eigvecs = np.linalg.eigh(M)
    # Take top 9 directions (non-zero eigenvalues)
    lda_dirs = eigvecs[:, -9:].T.astype(np.float32)  # [9, 784]

    # b) Class mean directions
    mean_dirs = (means_raw - global_mean).astype(np.float32)  # [10, 784]

    # c) Ridge regression directions for a few lambdas
    ridge_dirs = []
    XtX = X.T @ X
    for lam in [0.1, 1.0, 10.0]:
        A = XtX + lam * np.eye(D, dtype=np.float32)
        for c in range(C):
            yc = (y == c).astype(np.float32)
            wc = np.linalg.solve(A, X.T @ yc)
            ridge_dirs.append(wc)
    ridge_dirs = np.array(ridge_dirs, dtype=np.float32)  # [30, 784]

    # d) PCA directions (top 128)
    # Compute covariance and eigenvectors
    cov = X.T @ X / N
    eigvals_pca, eigvecs_pca = np.linalg.eigh(cov)
    pca_dirs = eigvecs_pca[:, -128:].T.astype(np.float32)  # [128, 784]

    # e) Random projections
    rng = np.random.RandomState(42)
    rand_dirs = rng.randn(200, D).astype(np.float32)

    # Combine all candidate directions
    all_dirs = np.vstack([
        w["fc1.weight"].astype(np.float32),  # current 128
        lda_dirs,
        mean_dirs,
        ridge_dirs,
        pca_dirs,
        rand_dirs
    ])  # [~500, 784]

    # For each direction, create two candidates: +d and -d
    # We'll evaluate each as a hidden unit: h = relu(X @ d)
    # Compute F-statistic for each candidate
    # To save memory, process in batches
    best_units = []
    best_F = []
    batch_size = 50
    for start in range(0, all_dirs.shape[0], batch_size):
        end = min(start + batch_size, all_dirs.shape[0])
        dirs_batch = all_dirs[start:end]
        # Compute h for +d and -d
        for sign in [1, -1]:
            d = sign * dirs_batch  # [B, 784]
            H_cand = np.maximum(X @ d.T, 0)  # [N, B]
            # Compute F-statistic for each column
            for j in range(H_cand.shape[1]):
                h = H_cand[:, j]
                # Between-class variance
                means_h = np.array([h[y == c].mean() for c in range(C)])
                overall_mean = h.mean()
                ss_between = np.sum((means_h - overall_mean)**2) * (N / C)  # assuming balanced
                ss_within = np.sum((h - means_h[y])**2)
                if ss_within < 1e-8:
                    F = 0.0
                else:
                    F = (ss_between / (C - 1)) / (ss_within / (N - C))
                best_units.append(d[j])
                best_F.append(F)
    best_F = np.array(best_F)
    best_units = np.array(best_units, dtype=np.float32)

    # Select top 128 units by F-statistic
    top_idx = np.argsort(best_F)[-128:]
    W1_new = best_units[top_idx]  # [128, 784]
    b1_new = np.zeros(128, dtype=np.float32)

    # Compute new hidden features
    H_new = np.maximum(X @ W1_new.T + b1_new, 0)  # [2000, 128]

    # Fit LDA on new features
    W2_new, b2_new = fit_lda(H_new, y)
    acc_new = eval_lda(H_new, y, W2_new, b2_new)

    # Choose best configuration
    if acc_new > acc_cur:
        w["fc1.weight"] = W1_new
        w["fc1.bias"] = b1_new
        w["fc2.weight"] = W2_new
        w["fc2.bias"] = b2_new
        chosen = "new_features_lda"
        final_acc = acc_new
    else:
        w["fc2.weight"] = W2_lda
        w["fc2.bias"] = b2_lda
        chosen = "current_features_lda"
        final_acc = acc_cur

    # Diagnostics
    diagnostics = {
        "chosen": chosen,
        "train_accuracy": float(final_acc),
        "num_candidates": int(len(best_F)),
        "fc1_weight_mean": float(w["fc1.weight"].mean()),
        "fc1_weight_std": float(w["fc1.weight"].std()),
        "fc2_weight_mean": float(w["fc2.weight"].mean()),
        "fc2_weight_std": float(w["fc2.weight"].std()),
        "fc2_bias_mean": float(w["fc2.bias"].mean()),
        "fc2_bias_std": float(w["fc2.bias"].std()),
    }

    return {"weights": w, "diagnostics": diagnostics}
