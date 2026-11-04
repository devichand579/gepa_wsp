import numpy as np

def edit(w, f):
    # Extract full feedback features and labels
    F = f["features"]          # (2000, 1568)
    y = f["y"]                 # (2000,)
    N, p = F.shape
    K = 10

    # Compute class means
    mu = np.zeros((K, p), dtype=np.float64)
    counts = np.zeros(K, dtype=np.int64)
    for k in range(K):
        mask = (y == k)
        counts[k] = mask.sum()
        if counts[k] > 0:
            mu[k] = F[mask].mean(axis=0)

    # Center features by their class mean
    F_c = F - mu[y]

    # Within-class scatter matrix
    S = F_c.T @ F_c          # (p, p)

    # Shrinkage: S_shrunk = (1-alpha)*S + alpha * trace(S)/p * I
    alpha = 0.001
    trace_S = np.trace(S)
    S_shrunk = (1.0 - alpha) * S + alpha * (trace_S / p) * np.eye(p, dtype=np.float64)

    # Solve for LDA weights: S_shrunk @ W.T = mu.T
    # W is (K, p)
    W = np.linalg.solve(S_shrunk, mu.T).T   # (K, p)

    # Bias: b_k = -0.5 * mu_k @ W_k + log(pi_k)
    b = np.zeros(K, dtype=np.float64)
    for k in range(K):
        b[k] = -0.5 * mu[k] @ W[k] + np.log(counts[k] / N)

    # Compute logits on the full pool to calibrate scale
    Z = F @ W.T + b                         # (N, K)
    std_Z = Z.std()
    if std_Z > 1e-8:
        scale = 1.0 / std_Z
        W *= scale
        b *= scale

    # Convert to float32 and assign to head
    w["head.weight"] = W.astype(np.float32)
    w["head.bias"]   = b.astype(np.float32)

    # Diagnostics
    diag = {
        "lda_alpha": alpha,
        "lda_scale": float(scale) if std_Z > 1e-8 else 1.0,
        "head_weight_std": float(w["head.weight"].std()),
        "head_bias_mean": float(w["head.bias"].mean()),
        "full_pool_acc": float((np.argmax(F @ W.T + b, axis=1) == y).mean())
    }

    return {"weights": w, "diagnostics": diag}
