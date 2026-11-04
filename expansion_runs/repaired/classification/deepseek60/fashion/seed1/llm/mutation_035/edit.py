import numpy as np

def edit(w, f):
    # Extract full feedback pool
    X = f["features"].astype(np.float64)          # [2000, 1568]
    y = f["y"].astype(np.int64)                   # [2000]
    N, D = X.shape
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Center features and targets
    X_mean = X.mean(axis=0, keepdims=True)
    Y_mean = Y.mean(axis=0, keepdims=True)
    Xc = X - X_mean
    Yc = Y - Y_mean

    # Ridge regression with a small regularization.
    # We use lambda = 0.1 as a conservative choice that avoids overfitting
    # while still correcting the head's mistakes.
    lam = 0.1
    # Solve (Xc^T Xc + lam I) W^T = Xc^T Yc
    # W is [C, D]
    A = Xc.T @ Xc + lam * np.eye(D, dtype=np.float64)
    B = Xc.T @ Yc
    W = np.linalg.solve(A, B).T                   # [C, D]
    b = (Y_mean - X_mean @ W.T).ravel()           # [C]

    # Scale weights and bias so that max absolute weight is 1000.
    max_abs_w = np.max(np.abs(W))
    if max_abs_w > 0:
        scale = 1000.0 / max_abs_w
        W = W * scale
        b = b * scale

    # Ensure finite and within bounds
    W = np.nan_to_num(W, nan=0.0, posinf=1000.0, neginf=-1000.0)
    b = np.nan_to_num(b, nan=0.0, posinf=1000.0, neginf=-1000.0)
    W = np.clip(W, -1000.0, 1000.0)
    b = np.clip(b, -1000.0, 1000.0)

    # Write back to the head
    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

    # Diagnostics
    logits = X @ W.T + b
    preds = np.argmax(logits, axis=1)
    acc = float(np.mean(preds == y))
    # Paired batch accuracy
    batch_ids = f["batch.ids"]
    Xb = X[batch_ids]
    yb = y[batch_ids]
    logits_b = Xb @ W.T + b
    preds_b = np.argmax(logits_b, axis=1)
    acc_b = float(np.mean(preds_b == yb))

    diagnostics = {
        "full_pool_accuracy": acc,
        "paired_batch_accuracy": acc_b,
        "lambda": lam,
        "max_abs_weight": float(np.max(np.abs(W))),
        "max_abs_bias": float(np.max(np.abs(b))),
    }
    return {"weights": w, "diagnostics": diagnostics}
