import numpy as np

def edit(w, f):
    # Extract full-pool features and labels
    X = f["features"].astype(np.float64)          # (2000, 1568)
    y = f["y"].astype(np.int64)                   # (2000,)
    N, D = X.shape
    num_classes = 10

    # One-hot targets
    Y = np.zeros((N, num_classes), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Augment with bias column
    X_aug = np.hstack([X, np.ones((N, 1))])       # (2000, 1569)

    # Ridge regression: minimize ||X_aug W - Y||^2 + λ ||W||^2
    # Do not regularize the bias term (last row/col)
    lam = 0.1
    reg = np.eye(D + 1) * lam
    reg[-1, -1] = 0.0

    A = X_aug.T @ X_aug + reg
    b = X_aug.T @ Y
    W = np.linalg.solve(A, b)                     # (1569, 10)

    # Split into head weight and bias
    head_weight = W[:-1].T.astype(np.float32)     # (10, 1568)
    head_bias = W[-1].astype(np.float32)          # (10,)

    # Update weights
    w["head.weight"] = head_weight
    w["head.bias"] = head_bias

    # Diagnostics: evaluate on full pool and paired batch
    logits_full = X_aug @ W
    pred_full = np.argmax(logits_full, axis=1)
    acc_full = float(np.mean(pred_full == y))

    # Paired batch evaluation (using cached head input)
    X_batch = f["layer.head.input0"].astype(np.float64)  # (128, 1568)
    y_batch = f["batch.y"].astype(np.int64)              # (128,)
    X_batch_aug = np.hstack([X_batch, np.ones((X_batch.shape[0], 1))])
    logits_batch = X_batch_aug @ W
    pred_batch = np.argmax(logits_batch, axis=1)
    acc_batch = float(np.mean(pred_batch == y_batch))

    diagnostics = {
        "lambda": lam,
        "full_pool_acc": acc_full,
        "paired_batch_acc": acc_batch,
        "head_weight_std": float(np.std(head_weight)),
        "head_bias_mean": float(np.mean(head_bias))
    }

    return {"weights": w, "diagnostics": diagnostics}
