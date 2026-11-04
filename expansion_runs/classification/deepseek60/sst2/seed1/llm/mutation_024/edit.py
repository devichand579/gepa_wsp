import numpy as np

def edit(w, f):
    # Extract cached features and labels from the full feedback pool
    X = f["features"].astype(np.float64)  # [4000, 32]
    y = f["y"].astype(np.int64)           # [4000]
    n, d = X.shape

    # Compute class means and pooled covariance with shrinkage
    X0 = X[y == 0]
    X1 = X[y == 1]
    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)
    n0, n1 = X0.shape[0], X1.shape[0]

    # Pooled within-class covariance
    C0 = np.cov(X0, rowvar=False, bias=True)
    C1 = np.cov(X1, rowvar=False, bias=True)
    C = (n0 * C0 + n1 * C1) / (n0 + n1)

    # Shrinkage toward diagonal (Ledoit-Wolf style simple shrinkage)
    # Use a small shrinkage to stabilize inversion; 0.15 worked well in prior attempts.
    shrinkage = 0.15
    C_shrunk = (1.0 - shrinkage) * C + shrinkage * np.diag(np.diag(C))
    # Add tiny ridge for numerical stability
    C_shrunk += np.eye(d) * 1e-6

    # LDA direction: w = C^{-1} (mu1 - mu0)
    diff = mu1 - mu0
    w_dir = np.linalg.solve(C_shrunk, diff)

    # Normalize direction to unit norm (does not affect decision boundary)
    w_dir = w_dir / np.linalg.norm(w_dir)

    # Decision threshold: project means and set midpoint
    proj0 = w_dir @ mu0
    proj1 = w_dir @ mu1
    threshold = 0.5 * (proj0 + proj1)

    # We need logits such that logit1 - logit0 = w_dir @ x - threshold
    # Set head.weight[0] = -w_dir/2, head.weight[1] = +w_dir/2
    # Then logit1 - logit0 = w_dir @ x + (b1 - b0)
    # We want w_dir @ x - threshold, so b1 - b0 = -threshold.
    # Set b0 = 0, b1 = -threshold.
    head_w = np.zeros((2, d), dtype=np.float32)
    head_w[0] = -0.5 * w_dir.astype(np.float32)
    head_w[1] =  0.5 * w_dir.astype(np.float32)

    head_b = np.zeros(2, dtype=np.float32)
    head_b[0] = 0.0
    head_b[1] = -threshold

    # Write back to weights
    w["head.weight"] = head_w
    w["head.bias"] = head_b

    # Diagnostics
    # Compute training accuracy and cross-entropy with the new head
    logits = X @ head_w.T + head_b  # [4000, 2]
    # Softmax
    logits_max = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == y)
    # Cross-entropy
    eps = 1e-12
    ce = -np.mean(np.log(probs[np.arange(n), y] + eps))

    diagnostics = {
        "train_accuracy": float(acc),
        "train_ce": float(ce),
        "shrinkage": shrinkage,
        "w_dir_norm": float(np.linalg.norm(w_dir)),
        "threshold": float(threshold),
        "n_samples": int(n),
        "n_features": int(d),
    }

    return {"weights": w, "diagnostics": diagnostics}
