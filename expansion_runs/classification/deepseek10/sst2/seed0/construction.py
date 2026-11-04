import numpy as np

def edit(w, f):
    # Extract final hidden features and labels from the full training pool
    X = f["features"].astype(np.float64)          # (4000, 32)
    y = f["y"].astype(np.int64)                   # (4000,)
    N, D = X.shape

    # One-hot targets
    Y = np.zeros((N, 2), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Augment with bias column
    X_aug = np.hstack([X, np.ones((N, 1))])       # (4000, 33)

    # Ridge regression closed-form solution: (X^T X + λI) W = X^T Y
    lam = 1e-3
    A = X_aug.T @ X_aug + lam * np.eye(D + 1)
    B = X_aug.T @ Y
    W_aug = np.linalg.solve(A, B)                 # (33, 2)

    head_weight = W_aug[:D, :].T                  # (2, 32)
    head_bias   = W_aug[D, :]                     # (2,)

    # Write back into the parameter dictionary
    w["head.weight"] = head_weight.astype(np.float32)
    w["head.bias"]   = head_bias.astype(np.float32)

    # Diagnostics: training accuracy after refit
    logits = X @ head_weight.T + head_bias
    preds = np.argmax(logits, axis=1)
    acc = float(np.mean(preds == y))

    diagnostics = {
        "train_accuracy": acc,
        "head_weight_norm": float(np.linalg.norm(head_weight)),
        "head_bias_norm": float(np.linalg.norm(head_bias)),
        "lambda": lam,
    }

    return {"weights": w, "diagnostics": diagnostics}
