def edit(w, f):
    """
    Edit the network head using a closed‑form ridge regression on the
    entire feedback pool.  Only head.weight and head.bias are changed;
    all other parameters stay untouched so downstream activations stay valid.
    """
    import numpy as np

    # ------- data from feedback arrays -------
    X = f["features"].astype(np.float32)          # shape (N, 32)
    y = f["y"].astype(np.int32)                   # shape (N,), values 0 or 1
    N, D = X.shape
    C = 2                                          # binary output

    # one‑hot encode labels into a (N, C) matrix
    Y = np.zeros((N, C), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    # ridge regularisation (small constant)
    lam = 1e-3

    # ------- closed‑form ridge solution -------
    # Solve (X^T X + λ I) W = X^T Y  for W of shape (D, C)
    XTX = X.T @ X + lam * np.eye(D, dtype=np.float32)
    W = np.linalg.solve(XTX, X.T @ Y)            # shape (32, 2)

    # bias = mean of residuals (Y - X @ W)
    b = np.mean(Y - X @ W, axis=0)               # shape (2,)

    # ------- update head parameters -------
    w["head.weight"] = W.astype(np.float32)       # [2, 32]
    w["head.bias"]   = b.astype(np.float32)       # [2]

    # ------- diagnostics (optional but useful) -------
    preds = X @ W + b
    pred_labels = np.argmax(preds, axis=1)
    acc = float(np.mean(pred_labels == y))

    diagnostics = {
        "ridge_lambda": float(lam),
        "feedback_accuracy": acc,
        "head_weight_mean": float(np.mean(W)),
        "head_bias_mean": float(np.mean(b)),
    }

    return {"weights": w, "diagnostics": diagnostics}
