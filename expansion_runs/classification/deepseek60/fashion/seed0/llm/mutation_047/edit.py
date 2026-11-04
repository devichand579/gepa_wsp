import numpy as np

def edit(w, f):
    # Extract features and labels
    X = f["features"].astype(np.float64)  # (2000, 1568)
    y = f["y"].astype(np.int64)           # (2000,)
    n, d = X.shape
    n_classes = 10

    # One-hot encode labels
    Y = np.zeros((n, n_classes), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    # Center features and labels
    X_mean = X.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    Xc = X - X_mean
    Yc = Y - Y_mean

    # SVD of centered features
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    S2 = S ** 2

    # GCV to select lambda
    lambdas = np.logspace(-6, 6, 50)
    best_gcv = np.inf
    best_lambda = 1.0
    for lam in lambdas:
        d = S2 / (S2 + lam)
        trace_H = d.sum()
        # y_hat = U @ diag(d) @ U.T @ Yc
        y_hat = U @ (d[:, None] * (U.T @ Yc))
        residuals = Yc - y_hat
        rss = np.sum(residuals ** 2)
        gcv = (rss / n) / ((1.0 - trace_H / n) ** 2)
        if gcv < best_gcv:
            best_gcv = gcv
            best_lambda = lam

    # Compute final weights and bias with best lambda
    lam = best_lambda
    d = S2 / (S2 + lam)
    # W = V diag(S/(S^2+lam)) U^T Yc
    W = Vt.T @ ((S / (S2 + lam))[:, None] * (U.T @ Yc))  # (1568, 10)
    bias = Y_mean - X_mean @ W

    # Update head weights (PyTorch linear: [out, in] = [10, 1568])
    w["head.weight"] = W.T.astype(np.float32)
    w["head.bias"] = bias.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lambda),
        "best_gcv": float(best_gcv),
        "head_weight_norm": float(np.linalg.norm(w["head.weight"])),
        "head_bias_norm": float(np.linalg.norm(w["head.bias"])),
        "head_weight_max_abs": float(np.max(np.abs(w["head.weight"]))),
        "head_bias_max_abs": float(np.max(np.abs(w["head.bias"]))),
    }
    return {"weights": w, "diagnostics": diagnostics}
