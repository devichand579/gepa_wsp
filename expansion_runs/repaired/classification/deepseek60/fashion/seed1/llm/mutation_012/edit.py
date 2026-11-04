import numpy as np

def edit(w, f):
    # Read current weights
    conv1_w = w["conv1.weight"]
    conv1_b = w["conv1.bias"]
    conv2_w = w["conv2.weight"]
    conv2_b = w["conv2.bias"]
    head_w = w["head.weight"]
    head_b = w["head.bias"]

    # Feedback features and labels
    X = f["features"].astype(np.float64)  # [N, 1568]
    y = f["y"].astype(np.int64)           # [N]
    N, D = X.shape
    C = 10

    # Class means and counts
    counts = np.zeros(C, dtype=np.float64)
    means = np.zeros((C, D), dtype=np.float64)
    for c in range(C):
        idx = (y == c)
        counts[c] = idx.sum()
        if counts[c] > 0:
            means[c] = X[idx].mean(axis=0)

    # Global mean
    mu = X.mean(axis=0)

    # Within-class covariance (pooled)
    # S = sum_c sum_{i in c} (x_i - mu_c)(x_i - mu_c)^T / (N - C)
    # Compute efficiently using centered data
    Xc = X - means[y]
    S = (Xc.T @ Xc) / max(N - C, 1)

    # Shrinkage toward diagonal
    # Use a small shrinkage to stabilize high-dimensional covariance
    # Estimate shrinkage intensity from data
    var_diag = np.diag(S)
    mean_var = var_diag.mean()
    # Simple shrinkage: lambda = 0.1 or based on N/D
    # Here we use a modest fixed shrinkage; this is a closed-form regularized LDA.
    shrink = 0.15
    S_shrunk = (1.0 - shrink) * S + shrink * np.diag(var_diag)

    # Add tiny ridge for numerical stability
    S_shrunk += 1e-6 * np.eye(D)

    # Solve for LDA weights: W = S^{-1} (mu_c - mu)
    # We can solve S_shrunk @ W^T = (means - mu).T
    # Use Cholesky or solve
    # Since D=1568, solving is okay.
    # Compute inverse or solve linear system.
    # For efficiency, use np.linalg.solve with multiple RHS.
    RHS = (means - mu).T  # [D, C]
    W_lda = np.linalg.solve(S_shrunk, RHS).T  # [C, D]

    # Bias: b_c = -0.5 * mu_c^T S^{-1} mu_c + log(prior_c)
    # But we can also fit bias to match class log-priors.
    # Compute S^{-1} means
    # Actually W_lda = S^{-1} (mu_c - mu)
    # The discriminant is: x^T S^{-1} mu_c - 0.5 mu_c^T S^{-1} mu_c + log pi_c
    # We can compute b_c = -0.5 * mu_c^T S^{-1} mu_c + log(counts/N)
    # Note: our W_lda already includes -mu, so we can absorb mu into bias.
    # Let's compute the full LDA logits: x^T S^{-1} mu_c - 0.5 mu_c^T S^{-1} mu_c + log pi_c
    # We have W_lda = S^{-1} (mu_c - mu). So x^T W_lda = x^T S^{-1} mu_c - x^T S^{-1} mu.
    # The term -x^T S^{-1} mu is common to all classes, so it can be dropped.
    # So we can use W = S^{-1} mu_c, and b_c = -0.5 mu_c^T S^{-1} mu_c + log pi_c.
    # Let's compute S^{-1} mu_c directly.
    # We already have S_shrunk. Solve S_shrunk @ M = means.T -> M = S^{-1} means.T
    M = np.linalg.solve(S_shrunk, means.T)  # [D, C]
    W = M.T  # [C, D]
    b = -0.5 * np.sum(means.T * M, axis=0) + np.log(counts / N + 1e-12)

    # Now we have LDA logits: X @ W.T + b
    # We can optionally calibrate the scale to improve CE.
    # Compute current logits on feedback
    logits = X @ W.T + b  # [N, C]
    # Compute optimal temperature-like scale by matching mean margin to log(9)
    # This is a heuristic but preserves argmax.
    # Compute margins between top1 and top2
    sorted_logits = np.sort(logits, axis=1)
    margins = sorted_logits[:, -1] - sorted_logits[:, -2]
    mean_margin = margins.mean()
    target_margin = np.log(9.0)  # ~2.197
    if mean_margin > 1e-8:
        alpha = target_margin / mean_margin
    else:
        alpha = 1.0
    # Apply scaling
    W = W * alpha
    b = b * alpha

    # Clip to safe range
    W = np.clip(W, -1000.0, 1000.0).astype(np.float32)
    b = np.clip(b, -1000.0, 1000.0).astype(np.float32)

    # Update weights
    w["head.weight"] = W
    w["head.bias"] = b

    # Diagnostics
    # Compute accuracy on feedback
    pred = np.argmax(X @ W.T + b, axis=1)
    acc = (pred == y).mean()
    # Cross-entropy
    logits = X @ W.T + b
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    ce = -np.log(probs[np.arange(N), y] + 1e-12).mean()

    diagnostics = {
        "lda_shrink": float(shrink),
        "alpha": float(alpha),
        "feedback_accuracy": float(acc),
        "feedback_cross_entropy": float(ce),
        "head_weight_mean": float(W.mean()),
        "head_weight_std": float(W.std()),
        "head_bias_mean": float(b.mean()),
        "head_bias_std": float(b.std()),
    }

    return {"weights": w, "diagnostics": diagnostics}
