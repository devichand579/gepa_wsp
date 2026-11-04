import numpy as np

def edit(w, f):
    # Extract current hidden features and labels from the full feedback pool
    F = f["features"].astype(np.float64)   # (2000, 128)
    y = f["y"].astype(np.int64)            # (2000,)
    N, D = F.shape
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Augment features with a bias column
    F_aug = np.concatenate([F, np.ones((N, 1), dtype=np.float64)], axis=1)  # (2000, 129)

    # Ridge regression: solve (F_aug^T F_aug + lambda * I) W = F_aug^T Y
    # Do not regularise the bias term.
    lambda_ = 1e-3
    reg = np.eye(D + 1, dtype=np.float64) * lambda_
    reg[-1, -1] = 0.0

    A = F_aug.T @ F_aug + reg
    B = F_aug.T @ Y
    W_aug = np.linalg.solve(A, B)          # (129, 10)

    W_new = W_aug[:-1, :].T                # (10, 128)
    b_new = W_aug[-1, :]                   # (10,)

    # Update only fc2
    w["fc2.weight"] = np.clip(W_new.astype(np.float32), -1000.0, 1000.0)
    w["fc2.bias"]   = np.clip(b_new.astype(np.float32), -1000.0, 1000.0)

    # Diagnostics: training accuracy and cross-entropy on the feedback pool
    logits = F @ w["fc2.weight"].T + w["fc2.bias"]
    pred = np.argmax(logits, axis=1)
    acc = float(np.mean(pred == y))

    logits_max = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    ce = float(-np.mean(np.log(probs[np.arange(N), y] + 1e-12)))

    diagnostics = {
        "train_acc": acc,
        "train_ce": ce,
        "lambda": lambda_,
        "fc2_weight_mean": float(np.mean(w["fc2.weight"])),
        "fc2_weight_std": float(np.std(w["fc2.weight"])),
        "fc2_bias_mean": float(np.mean(w["fc2.bias"])),
        "fc2_bias_std": float(np.std(w["fc2.bias"])),
    }

    return {"weights": w, "diagnostics": diagnostics}
