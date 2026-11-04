import numpy as np

def edit(w, f):
    # Extract features and labels from the full feedback pool
    X = f["features"].astype(np.float64)          # (2000, 1568)
    y = f["y"].astype(np.int64)                   # (2000,)
    n, d = X.shape
    K = 10

    # One-hot targets
    Y = np.zeros((n, K), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    # Center features and targets to fit an affine head (bias absorbed by centering)
    X_mean = X.mean(axis=0, keepdims=True)
    Y_mean = Y.mean(axis=0, keepdims=True)
    Xc = X - X_mean
    Yc = Y - Y_mean

    # Ridge regression with GCV to choose lambda.
    # Use SVD of Xc: Xc = U S V^T
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    s2 = S * S
    # GCV for ridge: minimize ||Yc - Xc W||^2 / (n - df)^2
    # For each lambda, df = sum(s2 / (s2 + lambda))
    # We can compute the GCV score efficiently.
    # Candidate lambdas: log-spaced from 1e-3 to 1e6
    lambdas = np.logspace(-3, 6, 100)
    best_lambda = 1.0
    best_gcv = np.inf
    # Precompute U^T Yc
    UTY = U.T @ Yc
    for lam in lambdas:
        df = np.sum(s2 / (s2 + lam))
        # Fitted values: U diag(s2/(s2+lam)) U^T Yc
        # Residual sum of squares = ||Yc||^2 - sum( (s2/(s2+lam)) * (UTY^2) )? Actually:
        # Yc_hat = U diag(s2/(s2+lam)) U^T Yc
        # RSS = ||Yc||^2 - sum( (2*s2/(s2+lam) - (s2/(s2+lam))^2) * (UTY^2) )
        # Simpler: compute RSS directly using SVD.
        # Since d=1568, n=2000, we can compute for each lambda.
        # To avoid recomputing, we can compute the hat matrix trace.
        # Actually, GCV = (RSS/n) / (1 - df/n)^2
        # RSS = sum over columns of Yc of ||Yc_j - Xc W_j||^2
        # We can compute W = V diag(1/(s2+lam)) S U^T Yc
        # But that's per lambda. 100 lambdas * 2000x1568 is okay.
        # Let's do it directly for simplicity.
        # W = Vt.T @ diag(1/(s2+lam)) @ S @ UTY
        # But we can compute Yc_hat = U @ diag(s2/(s2+lam)) @ UTY
        # Then RSS = ||Yc - Yc_hat||^2
        # This is O(n*K) per lambda.
        scale = s2 / (s2 + lam)
        Yc_hat = U @ (scale[:, None] * UTY)
        rss = np.sum((Yc - Yc_hat) ** 2)
        gcv = (rss / n) / ((1.0 - df / n) ** 2)
        if gcv < best_gcv:
            best_gcv = gcv
            best_lambda = lam

    # Solve for W and bias with best_lambda
    # W = V diag(1/(s2+lam)) S U^T Yc
    # Actually Xc = U S Vt, so Xc^T Xc = V S^2 V^T
    # W = V diag(1/(s2+lam)) S U^T Yc
    # We can compute using the SVD components.
    scale = s2 / (s2 + best_lambda)
    # W = Vt.T @ diag(1/(s2+lam)) @ S @ UTY
    # But S is singular values, UTY is U^T Yc.
    # W = Vt.T @ ( (1.0/(s2+best_lambda))[:,None] * (S[:,None] * UTY) )
    W = Vt.T @ ((1.0 / (s2 + best_lambda))[:, None] * (S[:, None] * UTY))
    # W shape (d, K)
    # Bias: b = Y_mean - X_mean @ W
    b = Y_mean - X_mean @ W

    # Set head weights and bias
    # head.weight shape (10, 1568) = W.T
    # head.bias shape (10,) = b.ravel()
    w["head.weight"] = W.T.astype(np.float32)
    w["head.bias"] = b.ravel().astype(np.float32)

    # Diagnostics
    diag = {
        "best_lambda": float(best_lambda),
        "best_gcv": float(best_gcv),
        "head_weight_norm": float(np.linalg.norm(w["head.weight"])),
        "head_bias_norm": float(np.linalg.norm(w["head.bias"])),
        "head_weight_max_abs": float(np.max(np.abs(w["head.weight"]))),
        "head_bias_max_abs": float(np.max(np.abs(w["head.bias"]))),
    }
    return {"weights": w, "diagnostics": diag}
