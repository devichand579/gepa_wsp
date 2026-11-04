import numpy as np

def edit(w, f):
    # Extract full feedback features and labels
    X = f["features"].astype(np.float64)  # [2000, 1568]
    y = f["y"].astype(np.int64)           # [2000]
    N, D = X.shape

    # One-hot targets
    Y = np.zeros((N, 10), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Augment features with a bias column
    X_aug = np.concatenate([X, np.ones((N, 1))], axis=1)  # [N, D+1]

    # Ridge regression: solve (X^T X + λI) W = X^T Y
    lam = 1.0
    A = X_aug.T @ X_aug + lam * np.eye(D + 1)
    b = X_aug.T @ Y
    W_aug = np.linalg.solve(A, b)  # [D+1, 10]

    # Extract head weights and bias
    head_weight = W_aug[:-1].T.astype(np.float32)  # [10, D]
    head_bias = W_aug[-1].astype(np.float32)       # [10]

    # Safety clip to stay within allowed bounds
    head_weight = np.clip(head_weight, -1000, 1000)
    head_bias = np.clip(head_bias, -1000, 1000)

    w["head.weight"] = head_weight
    w["head.bias"] = head_bias

    # Diagnostics: evaluate new head on the feedback pool
    logits_new = X @ head_weight.T + head_bias
    preds = np.argmax(logits_new, axis=1)
    acc = np.mean(preds == y)

    logits_max = logits_new.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits_new - logits_max)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    ce = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))

    diagnostics = {
        "train_accuracy": float(acc),
        "train_cross_entropy": float(ce),
        "head_weight_mean": float(np.mean(head_weight)),
        "head_weight_std": float(np.std(head_weight)),
        "head_weight_min": float(np.min(head_weight)),
        "head_weight_max": float(np.max(head_weight)),
        "head_bias": head_bias.tolist()
    }

    return {"weights": w, "diagnostics": diagnostics}
