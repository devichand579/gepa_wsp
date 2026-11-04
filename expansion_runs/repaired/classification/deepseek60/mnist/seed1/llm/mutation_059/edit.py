import numpy as np

def edit(w, f):
    # Extract fixed features and labels from the full feedback pool
    X = f["features"].astype(np.float64)   # (2000, 128)
    y = f["y"].astype(np.int64)            # (2000,)
    N, D = X.shape
    K = 10

    # Class means and counts
    means = np.zeros((K, D))
    counts = np.zeros(K)
    for k in range(K):
        idx = (y == k)
        counts[k] = np.sum(idx)
        if counts[k] > 0:
            means[k] = np.mean(X[idx], axis=0)

    # Within-class scatter matrix Sw
    Sw = np.zeros((D, D))
    for k in range(K):
        idx = (y == k)
        if counts[k] > 0:
            Xk = X[idx] - means[k]
            Sw += Xk.T @ Xk

    # Small ridge regularisation for numerical stability
    avg_var = np.trace(Sw) / max(1, N - K)
    reg = 1e-6 * avg_var
    Sw_reg = Sw + reg * np.eye(D)

    # LDA weights and biases
    Sw_inv = np.linalg.inv(Sw_reg)
    W = (Sw_inv @ means.T).T               # (K, D)
    bias = np.zeros(K)
    for k in range(K):
        bias[k] = -0.5 * means[k] @ W[k] + np.log(counts[k] / N)

    # Ensure finite float32 within bounds
    W = np.clip(W, -1000.0, 1000.0).astype(np.float32)
    bias = np.clip(bias, -1000.0, 1000.0).astype(np.float32)

    # Update only fc2
    w["fc2.weight"] = W
    w["fc2.bias"] = bias

    # Diagnostics
    logits = X @ W.T.astype(np.float64) + bias.astype(np.float64)
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == y)

    diag = {
        "method": "LDA on existing features",
        "reg": float(reg),
        "counts": counts.tolist(),
        "train_accuracy": float(acc),
        "fc2_weight_mean": float(np.mean(W)),
        "fc2_weight_std": float(np.std(W)),
        "fc2_bias_mean": float(np.mean(bias)),
        "fc2_bias_std": float(np.std(bias)),
    }
    return {"weights": w, "diagnostics": diag}
