import numpy as np

def edit(w, f):
    # Read full feedback pool
    X = f["features"].astype(np.float64)          # [2000, 1568]
    y = f["y"].astype(np.int64)                   # [2000]
    N, D = X.shape
    C = 10

    # Current head accuracy for reference
    W_cur = w["head.weight"].astype(np.float64)   # [10, 1568]
    b_cur = w["head.bias"].astype(np.float64)     # [10]
    logits_cur = X @ W_cur.T + b_cur
    acc_cur = np.mean(np.argmax(logits_cur, axis=1) == y)

    # Build one-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Augment features with a bias column
    X_aug = np.concatenate([X, np.ones((N, 1), dtype=np.float64)], axis=1)  # [2000, 1569]

    # Ridge regression: solve (X_aug^T X_aug + lambda I) W = X_aug^T Y
    # Use a modest lambda; 0.1 worked well in prior feedback.
    lam = 0.1
    A = X_aug.T @ X_aug
    A[np.diag_indices_from(A)] += lam
    B = X_aug.T @ Y
    W_aug = np.linalg.solve(A, B)                 # [1569, 10]

    W_new = W_aug[:D, :].T                        # [10, 1568]
    b_new = W_aug[D, :]                           # [10]

    # Evaluate new head on the full pool
    logits_new = X @ W_new.T + b_new
    acc_new = np.mean(np.argmax(logits_new, axis=1) == y)

    # If the new head is not better on the training pool, keep the current one.
    if acc_new >= acc_cur:
        w["head.weight"] = W_new.astype(np.float32)
        w["head.bias"] = b_new.astype(np.float32)
        used = "ridge"
        acc_used = acc_new
    else:
        used = "current"
        acc_used = acc_cur

    # Ensure finite float32 and bounded values
    for k in ["conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias", "head.weight", "head.bias"]:
        w[k] = np.clip(w[k].astype(np.float32), -1000.0, 1000.0)

    diagnostics = {
        "acc_current": float(acc_cur),
        "acc_new_ridge": float(acc_new),
        "acc_used": float(acc_used),
        "used": used,
        "lambda": lam,
        "head_weight_std": float(np.std(w["head.weight"])),
        "head_bias_mean": float(np.mean(w["head.bias"])),
    }
    return {"weights": w, "diagnostics": diagnostics}
