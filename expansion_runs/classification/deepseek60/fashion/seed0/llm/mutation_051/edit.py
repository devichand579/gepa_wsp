import numpy as np

def edit(w, f):
    # Extract features and labels from the full feedback pool
    X = f["features"].astype(np.float64)          # [2000, 1568]
    y = f["y"].astype(np.int64)                   # [2000]
    n, d = X.shape
    num_classes = 10

    # One-hot encode labels
    Y = np.zeros((n, num_classes), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    # Center features and labels (helps numerical stability and GCV)
    X_mean = X.mean(axis=0, keepdims=True)
    Y_mean = Y.mean(axis=0, keepdims=True)
    Xc = X - X_mean
    Yc = Y - Y_mean

    # SVD of centered feature matrix
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # S: [min(n,d)] singular values
    S2 = S ** 2

    # GCV for ridge regression on centered data
    # We search over a log-spaced grid of lambda values.
    # GCV(lambda) = (1/n) * ||Yc - Xc W||^2 / (1 - trace(H)/n)^2
    # where H = Xc (Xc^T Xc + lambda I)^-1 Xc^T.
    # Using SVD, trace(H) = sum( S2 / (S2 + lambda) )
    # and residual sum of squares = ||Yc||^2 - sum( (U^T Yc)^2 * S2 / (S2 + lambda) )
    # We compute this efficiently for each lambda.
    UTY = U.T @ Yc
    Yc_norm2 = np.sum(Yc ** 2)

    lambdas = np.logspace(-2, 6, 50)
    best_lambda = None
    best_gcv = np.inf
    for lam in lambdas:
        shrink = S2 / (S2 + lam)
        trace_H = np.sum(shrink)
        rss = Yc_norm2 - np.sum((UTY ** 2) * shrink)
        gcv = (rss / n) / ((1.0 - trace_H / n) ** 2)
        if gcv < best_gcv:
            best_gcv = gcv
            best_lambda = lam

    # Solve for final weights using best_lambda
    lam = best_lambda
    shrink = S2 / (S2 + lam)
    # W = V diag(shrink) U^T Yc
    W = Vt.T @ (shrink[:, None] * UTY)   # [d, num_classes]
    # Bias: since we centered, the intercept is Y_mean - X_mean @ W
    b = Y_mean.flatten() - X_mean.flatten() @ W

    # Update head weights and bias
    # head.weight shape: [10, 1568] (out, in)
    w["head.weight"] = W.T.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

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
