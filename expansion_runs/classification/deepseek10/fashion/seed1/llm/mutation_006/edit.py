import numpy as np

def edit(w, f):
    # Extract feedback features and labels
    X = f["features"].astype(np.float64)  # (2000, 1568)
    y = f["y"].astype(np.int64)           # (2000,)
    N, D = X.shape
    num_classes = 10

    # One-hot targets
    Y = np.zeros((N, num_classes), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Add bias column to features
    X_aug = np.hstack([X, np.ones((N, 1), dtype=np.float64)])  # (2000, 1569)

    # Closed-form least-squares solution (uses SVD, handles rank deficiency)
    W_aug, _, _, _ = np.linalg.lstsq(X_aug, Y, rcond=None)  # (1569, 10)

    # Extract head weight and bias
    head_weight = W_aug[:D, :].T  # (10, 1568)
    head_bias = W_aug[D, :]       # (10,)

    # Compute logits on feedback to find optimal temperature
    logits = X @ head_weight.T + head_bias  # (2000, 10)

    # Grid search for temperature that minimizes cross-entropy
    best_T = 1.0
    best_ce = np.inf
    for T in np.logspace(-1, 3, 200):
        scaled_logits = logits * T
        # Stable softmax
        max_logits = np.max(scaled_logits, axis=1, keepdims=True)
        exp_logits = np.exp(scaled_logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        ce = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))
        if ce < best_ce:
            best_ce = ce
            best_T = T

    # Apply temperature scaling
    head_weight *= best_T
    head_bias *= best_T

    # Update weights in the dictionary (float32, finite, bounded)
    w["head.weight"] = np.clip(head_weight, -1000.0, 1000.0).astype(np.float32)
    w["head.bias"] = np.clip(head_bias, -1000.0, 1000.0).astype(np.float32)

    # Diagnostics
    final_logits = X @ w["head.weight"].T + w["head.bias"]
    final_preds = np.argmax(final_logits, axis=1)
    train_acc = np.mean(final_preds == y)
    max_logits = np.max(final_logits, axis=1, keepdims=True)
    exp_logits = np.exp(final_logits - max_logits)
    final_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    final_ce = -np.mean(np.log(final_probs[np.arange(N), y] + 1e-12))

    diagnostics = {
        "best_temperature": float(best_T),
        "train_accuracy": float(train_acc),
        "train_cross_entropy": float(final_ce),
        "head_weight_mean": float(np.mean(w["head.weight"])),
        "head_weight_std": float(np.std(w["head.weight"])),
        "head_bias": w["head.bias"].tolist()
    }

    return {"weights": w, "diagnostics": diagnostics}
