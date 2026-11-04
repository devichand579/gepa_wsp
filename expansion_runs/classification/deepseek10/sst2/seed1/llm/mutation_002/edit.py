import numpy as np

def edit(w, f):
    # Extract pooled features (input to the head) and labels
    F = f["features"].astype(np.float64)          # [4000, 32]
    y = f["y"].astype(np.int64)                   # [4000]
    N, D = F.shape

    # One‑hot targets
    Y = np.zeros((N, 2), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Center features and targets
    F_mean = F.mean(axis=0, keepdims=True)        # [1, D]
    Y_mean = Y.mean(axis=0, keepdims=True)        # [1, 2]
    Fc = F - F_mean
    Yc = Y - Y_mean

    # Ridge regression: solve (Fc^T Fc + λ I) W = Fc^T Yc
    lam = 1e-2
    A = Fc.T @ Fc + lam * np.eye(D)
    B = Fc.T @ Yc
    W = np.linalg.solve(A, B)                     # [D, 2]
    b = (Y_mean - F_mean @ W).ravel()             # [2]

    # Head weight is [out, in] = [2, 32]
    w["head.weight"] = W.T.astype(np.float32)
    w["head.bias"]   = b.astype(np.float32)

    # Diagnostics: training accuracy on the feedback pool
    logits = F @ W + b
    preds = np.argmax(logits, axis=1)
    acc = float((preds == y).mean())

    return {
        "weights": w,
        "diagnostics": {
            "head_fit_train_accuracy": acc,
            "ridge_lambda": lam,
            "head_weight_norm": float(np.linalg.norm(W)),
            "head_bias": b.tolist(),
        },
    }
