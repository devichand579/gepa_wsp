import numpy as np

def edit(w, f):
    # Extract current parameters
    fc1_w = w["fc1.weight"].copy()
    fc1_b = w["fc1.bias"].copy()
    fc2_w = w["fc2.weight"].copy()
    fc2_b = w["fc2.bias"].copy()

    # Feedback data
    X = f["features"].astype(np.float64)          # (2000, 128)
    y = f["y"].astype(np.int64)                   # (2000,)
    N, D = X.shape
    K = 10

    # One-hot targets
    Y = np.zeros((N, K), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Compute class means and global mean
    class_counts = np.bincount(y, minlength=K).astype(np.float64)
    class_means = np.zeros((K, D), dtype=np.float64)
    for k in range(K):
        if class_counts[k] > 0:
            class_means[k] = X[y == k].mean(axis=0)
    global_mean = X.mean(axis=0)

    # Pooled within-class covariance (regularized)
    S_w = np.zeros((D, D), dtype=np.float64)
    for k in range(K):
        Xk = X[y == k]
        if len(Xk) > 1:
            diff = Xk - class_means[k]
            S_w += diff.T @ diff
    S_w /= max(N - K, 1)
    # Ridge regularization for numerical stability
    reg = 1e-3 * np.trace(S_w) / D
    S_w += reg * np.eye(D)

    # LDA weights: W = S_w^{-1} (class_means - global_mean)^T
    # Solve S_w * W^T = (class_means - global_mean)^T
    M = class_means - global_mean                     # (K, D)
    W_lda = np.linalg.solve(S_w, M.T).T               # (K, D)
    # LDA bias: b_k = -0.5 * M_k^T S_w^{-1} M_k + log(pi_k)
    # Note: M_k here is class_means[k] - global_mean
    b_lda = np.zeros(K, dtype=np.float64)
    for k in range(K):
        b_lda[k] = -0.5 * M[k] @ W_lda[k] + np.log(class_counts[k] / N)

    # Compute new logits on feedback pool
    logits_new = X @ W_lda.T + b_lda

    # Current logits scale
    logits_cur = f["logits"].astype(np.float64)
    std_cur = logits_cur.std()
    std_new = logits_new.std()
    if std_new > 1e-12:
        scale = std_cur / std_new
    else:
        scale = 1.0

    # Apply scale to weights and bias
    W_final = (W_lda * scale).astype(np.float32)
    b_final = (b_lda * scale).astype(np.float32)

    # Ensure finite and bounded
    W_final = np.clip(W_final, -1000.0, 1000.0)
    b_final = np.clip(b_final, -1000.0, 1000.0)

    # Update weights
    w["fc2.weight"] = W_final
    w["fc2.bias"] = b_final

    # Diagnostics
    diag = {
        "lda_scale": float(scale),
        "fc2_weight_norm": float(np.linalg.norm(W_final)),
        "fc2_bias_norm": float(np.linalg.norm(b_final)),
        "fc2_weight_min": float(W_final.min()),
        "fc2_weight_max": float(W_final.max()),
        "fc2_bias_min": float(b_final.min()),
        "fc2_bias_max": float(b_final.max()),
        "logits_std_new": float(std_new),
        "logits_std_cur": float(std_cur),
    }

    return {"weights": w, "diagnostics": diag}
