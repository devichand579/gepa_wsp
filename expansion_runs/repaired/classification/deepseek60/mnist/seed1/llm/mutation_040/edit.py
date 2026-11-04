import numpy as np

def edit(w, f):
    # Load full training pool
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)  # (2000, 784)
    y = f["y"].astype(np.int64)
    n, d = X.shape
    num_classes = 10

    # Class statistics
    counts = np.zeros(num_classes)
    means = np.zeros((num_classes, d))
    for k in range(num_classes):
        idx = (y == k)
        counts[k] = idx.sum()
        if counts[k] > 0:
            means[k] = X[idx].mean(axis=0)
    global_mean = X.mean(axis=0)

    # Within-class scatter
    X_centered = X - means[y]
    S_w = X_centered.T @ X_centered  # (784, 784)
    trace_Sw = np.trace(S_w)
    lambda_w = 1e-3 * trace_Sw / d
    S_w_reg = S_w + lambda_w * np.eye(d)

    # LDA directions: w_k = S_w^{-1} (means[k] - global_mean)
    diff = means - global_mean  # (10, 784)
    W_lda = np.linalg.solve(S_w_reg, diff.T).T  # (10, 784)

    # Normalize LDA directions to unit norm for stable thresholds
    for k in range(num_classes):
        norm = np.linalg.norm(W_lda[k])
        if norm > 1e-12:
            W_lda[k] /= norm

    # Build fc1: 10 LDA directions × 12 thresholds = 120 units
    W1_list = []
    b1_list = []
    num_thresholds = 12
    for k in range(num_classes):
        w_dir = W_lda[k]
        z = X @ w_dir
        thresholds = np.quantile(z, np.linspace(0.05, 0.95, num_thresholds))
        for t in thresholds:
            W1_list.append(w_dir)
            b1_list.append(-t)

    # Add 8 PCA directions (top 8 principal components)
    X_centered_global = X - global_mean
    U, S, Vt = np.linalg.svd(X_centered_global, full_matrices=False)
    W_pca = Vt[:8]  # (8, 784)
    for i in range(8):
        W1_list.append(W_pca[i])
        b1_list.append(0.0)  # threshold at 0 (centered)

    W1 = np.array(W1_list)  # (128, 784)
    b1 = np.array(b1_list)  # (128,)

    # Compute hidden features
    H = np.maximum(0, X @ W1.T + b1)  # (2000, 128)

    # Fit fc2 via LDA on hidden features
    means_h = np.zeros((num_classes, 128))
    for k in range(num_classes):
        idx = (y == k)
        if counts[k] > 0:
            means_h[k] = H[idx].mean(axis=0)
    H_centered = H - means_h[y]
    S_h = H_centered.T @ H_centered
    trace_Sh = np.trace(S_h)
    lambda_h = 1e-3 * trace_Sh / 128
    S_h_reg = S_h + lambda_h * np.eye(128)

    # LDA weights: W2_k = S_h^{-1} means_h[k]
    W2 = np.linalg.solve(S_h_reg, means_h.T).T  # (10, 128)
    # Bias: b2_k = -0.5 * means_h[k] @ W2_k + log(prior_k)
    b2 = -0.5 * np.sum(means_h * W2, axis=1) + np.log(counts / n)

    # Clip to ensure finite and |value| <= 1000
    W1 = np.clip(W1, -1000.0, 1000.0)
    b1 = np.clip(b1, -1000.0, 1000.0)
    W2 = np.clip(W2, -1000.0, 1000.0)
    b2 = np.clip(b2, -1000.0, 1000.0)

    # Update weights
    w["fc1.weight"] = W1.astype(np.float32)
    w["fc1.bias"] = b1.astype(np.float32)
    w["fc2.weight"] = W2.astype(np.float32)
    w["fc2.bias"] = b2.astype(np.float32)

    # Diagnostics
    logits = H @ W2.T + b2
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)
    acc = (preds == y).mean()
    ce = -np.log(probs[np.arange(n), y] + 1e-12).mean()

    diagnostics = {
        "train_accuracy": float(acc),
        "train_cross_entropy": float(ce),
        "fc1_weight_mean": float(w["fc1.weight"].mean()),
        "fc1_weight_std": float(w["fc1.weight"].std()),
        "fc2_weight_mean": float(w["fc2.weight"].mean()),
        "fc2_weight_std": float(w["fc2.weight"].std()),
        "fc2_bias_mean": float(w["fc2.bias"].mean()),
        "fc2_bias_std": float(w["fc2.bias"].std()),
    }
    return {"weights": w, "diagnostics": diagnostics}
