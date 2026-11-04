import numpy as np

def edit(w, f):
    # Extract features and labels from the full feedback pool
    F = f["features"].astype(np.float64)  # (2000, 128)
    y = f["y"].astype(np.int64)           # (2000,)
    N, D = F.shape
    C = 10

    # Standardize features (avoid division by zero)
    mu = F.mean(axis=0, keepdims=True)
    sigma = F.std(axis=0, keepdims=True)
    sigma[sigma < 1e-6] = 1.0
    F_std = (F - mu) / sigma

    # Add bias column
    F_aug = np.hstack([F_std, np.ones((N, 1))])  # (N, D+1)

    # One-hot targets
    Y = np.zeros((N, C))
    Y[np.arange(N), y] = 1.0

    # Ridge regression (closed-form)
    lam = 0.1
    A = F_aug.T @ F_aug + lam * np.eye(D + 1)
    B = F_aug.T @ Y
    W_aug = np.linalg.solve(A, B)  # (D+1, C)

    # Extract fc2 weights and bias
    W = W_aug[:D, :].T  # (C, D)
    b = W_aug[D, :]     # (C,)

    # Compute new logits on training data
    logits_new = F_aug @ W_aug  # (N, C)

    # Scale to match current logit standard deviation
    current_logits = f["logits"]
    current_std = current_logits.std()
    new_std = logits_new.std()
    scale = current_std / (new_std + 1e-12)

    W = W * scale
    b = b * scale

    # Update weights
    w["fc2.weight"] = W.astype(np.float32)
    w["fc2.bias"] = b.astype(np.float32)

    # Diagnostics
    logits_scaled = logits_new * scale
    logits_scaled -= logits_scaled.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits_scaled)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    ce = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))
    acc = np.mean(np.argmax(logits_scaled, axis=1) == y)

    diag = {
        "scale": float(scale),
        "new_std_before_scale": float(new_std),
        "current_std": float(current_std),
        "lambda": lam,
        "fc2_weight_norm": float(np.linalg.norm(W)),
        "fc2_bias_norm": float(np.linalg.norm(b)),
        "fc2_weight_min": float(W.min()),
        "fc2_weight_max": float(W.max()),
        "fc2_bias_min": float(b.min()),
        "fc2_bias_max": float(b.max()),
        "train_accuracy": float(acc),
        "train_ce": float(ce),
    }
    return {"weights": w, "diagnostics": diag}
