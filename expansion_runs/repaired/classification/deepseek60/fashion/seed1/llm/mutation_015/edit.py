import numpy as np

def edit(w, f):
    # Extract full feedback features and labels
    X = f["features"].astype(np.float64)  # (2000, 1568)
    y = f["y"].astype(np.int64)           # (2000,)
    N, D = X.shape
    C = 10

    # One-hot encode labels
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Augment features with a bias column
    X_aug = np.hstack([X, np.ones((N, 1))])  # (N, D+1)

    # Closed-form ridge regression: W_aug = (X_aug^T X_aug + lambda I)^-1 X_aug^T Y
    lam = 0.01
    A = X_aug.T @ X_aug + lam * np.eye(D + 1)
    B = X_aug.T @ Y
    W_aug = np.linalg.solve(A, B)  # (D+1, C)

    W = W_aug[:D, :].T  # (C, D)
    b = W_aug[D, :]     # (C,)

    # Compute logits on the full feedback pool
    logits = X @ W.T + b  # (N, C)

    # Compute mean margin between top-1 and top-2 logits
    sorted_logits = np.sort(logits, axis=1)
    top1 = sorted_logits[:, -1]
    top2 = sorted_logits[:, -2]
    mean_margin = np.mean(top1 - top2)

    # Scale so that the mean margin becomes log(9) ~ 2.197
    target_margin = np.log(9.0)
    alpha = target_margin / (mean_margin + 1e-12)
    alpha = np.clip(alpha, 0.1, 100.0)  # avoid extreme scaling

    W = W * alpha
    b = b * alpha

    # Safety clip to keep absolute values <= 1000
    W = np.clip(W, -1000.0, 1000.0)
    b = np.clip(b, -1000.0, 1000.0)

    # Update only the head parameters
    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

    # Diagnostics
    diag = {
        "lambda": float(lam),
        "alpha": float(alpha),
        "mean_margin": float(mean_margin),
        "target_margin": float(target_margin),
        "head_weight_std": float(np.std(W)),
        "head_bias_mean": float(np.mean(b)),
        "head_weight_min": float(np.min(W)),
        "head_weight_max": float(np.max(W)),
    }
    return {"weights": w, "diagnostics": diag}
