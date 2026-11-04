import numpy as np

def edit(w, f):
    # Extract features and labels from the full feedback pool
    X = f["features"].astype(np.float64)   # (2000, 1568)
    y = f["y"].astype(np.int64)            # (2000,)
    n, d = X.shape
    classes = np.unique(y)
    C = len(classes)

    # Global mean and variance (diagonal covariance)
    mu = X.mean(axis=0)                    # (d,)
    var = X.var(axis=0) + 1e-6             # (d,) add small epsilon
    inv_var = 1.0 / var

    # Class means and priors
    mu_c = np.zeros((C, d), dtype=np.float64)
    priors = np.zeros(C, dtype=np.float64)
    for i, c in enumerate(classes):
        idx = (y == c)
        mu_c[i] = X[idx].mean(axis=0)
        priors[i] = idx.mean()

    # Diagonal LDA weights and biases
    # logits_c = sum_j (mu_cj / var_j) * h_j - 0.5 * sum_j (mu_cj^2 / var_j) + log(pi_c)
    W = mu_c * inv_var[None, :]            # (C, d)
    b = -0.5 * np.sum(mu_c * mu_c * inv_var[None, :], axis=1) + np.log(priors + 1e-12)

    # Map back to the original class order (0..9) in case classes are not sorted
    W_full = np.zeros((10, d), dtype=np.float64)
    b_full = np.zeros(10, dtype=np.float64)
    for i, c in enumerate(classes):
        W_full[c] = W[i]
        b_full[c] = b[i]

    # Write into the head only; keep conv layers unchanged
    w["head.weight"] = W_full.astype(np.float32)
    w["head.bias"] = b_full.astype(np.float32)

    # Optional: clip to safe range (absolute value <= 1000)
    w["head.weight"] = np.clip(w["head.weight"], -1000.0, 1000.0)
    w["head.bias"] = np.clip(w["head.bias"], -1000.0, 1000.0)

    # Diagnostics
    logits = X @ W_full.T + b_full[None, :]
    pred = np.argmax(logits, axis=1)
    train_acc = np.mean(pred == y)
    diagnostics = {
        "train_accuracy": float(train_acc),
        "head_weight_norm": float(np.linalg.norm(w["head.weight"])),
        "head_bias_norm": float(np.linalg.norm(w["head.bias"])),
        "method": "diagonal_lda"
    }
    return {"weights": w, "diagnostics": diagnostics}
