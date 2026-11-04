import numpy as np

def edit(w, f):
    # Extract cached features and labels for the entire pool
    X = f["features"].astype(np.float64)   # [4000, 32]
    y = f["y"].astype(np.int64)            # [4000]
    N, D = X.shape

    # Center the features
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu

    # Compute class means and pooled covariance
    idx0 = (y == 0)
    idx1 = (y == 1)
    mu0 = Xc[idx0].mean(axis=0)
    mu1 = Xc[idx1].mean(axis=0)

    # Pooled covariance with shrinkage
    cov0 = np.cov(Xc[idx0].T, bias=True)
    cov1 = np.cov(Xc[idx1].T, bias=True)
    n0 = idx0.sum()
    n1 = idx1.sum()
    cov = (n0 * cov0 + n1 * cov1) / (n0 + n1)

    # Shrinkage towards diagonal
    shrinkage = 0.15
    cov_shrunk = (1.0 - shrinkage) * cov + shrinkage * np.diag(np.diag(cov))

    # Add small ridge for numerical stability
    cov_shrunk += 1e-6 * np.eye(D)

    # LDA direction
    inv_cov = np.linalg.inv(cov_shrunk)
    w_dir = inv_cov @ (mu1 - mu0)

    # Normalize direction to avoid huge logits
    w_dir = w_dir / (np.linalg.norm(w_dir) + 1e-12)

    # Compute scores on the training data
    scores = Xc @ w_dir
    # Optimal threshold: midpoint between class means of scores
    thr = 0.5 * (scores[idx0].mean() + scores[idx1].mean())

    # Set head weights: row0 = -w_dir, row1 = +w_dir, bias to center at thr
    # We want logit1 - logit0 = 2 * (Xc @ w_dir - thr)
    # So set head.weight[0] = -w_dir, head.weight[1] = +w_dir
    # head.bias[0] = 0, head.bias[1] = -2 * thr
    # But we must account for the fact that X = features, and we centered by mu.
    # The network computes logits = head @ features + bias.
    # We want logits = [0, 2*(features @ w_dir - thr)].
    # So head.weight[0] = 0, head.weight[1] = 2*w_dir
    # head.bias[0] = 0, head.bias[1] = -2*thr - 2*(mu @ w_dir)
    # Actually, features @ w_dir = (Xc + mu) @ w_dir = Xc @ w_dir + mu @ w_dir.
    # So we want logit1 = 2*(Xc @ w_dir - thr) = 2*(features @ w_dir - mu @ w_dir - thr).
    # Thus head.weight[1] = 2*w_dir, head.bias[1] = -2*(mu @ w_dir + thr).
    # For row0, we can set it to zero to keep logit0 = 0.
    head_w = np.zeros((2, D), dtype=np.float32)
    head_w[1] = (2.0 * w_dir).astype(np.float32)
    head_b = np.zeros(2, dtype=np.float32)
    head_b[1] = (-2.0 * (mu @ w_dir + thr)).astype(np.float32)

    # Update weights
    w["head.weight"] = head_w
    w["head.bias"] = head_b

    # Diagnostics
    diag = {
        "lda_shrinkage": shrinkage,
        "w_dir_norm": float(np.linalg.norm(w_dir)),
        "thr": float(thr),
        "head_bias1": float(head_b[1]),
        "train_accuracy": float(((scores > thr) == (y == 1)).mean()),
        "feature_std_min": float(X.std(axis=0).min()),
        "feature_std_max": float(X.std(axis=0).max()),
    }

    return {"weights": w, "diagnostics": diag}
