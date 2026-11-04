import numpy as np

def edit(w, f):
    # Extract full feedback features and labels
    H = f["features"].astype(np.float64)   # (2000, 128)
    y = f["y"].astype(np.int64)            # (2000,)
    n, d = H.shape
    c = 10

    # One-hot encode labels
    Y = np.zeros((n, c), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    # Center features and targets for bias handling
    H_mean = H.mean(axis=0)
    Hc = H - H_mean
    Y_mean = Y.mean(axis=0)
    Yc = Y - Y_mean

    # Precompute Gram matrices
    HtH = Hc.T @ Hc          # (128, 128)
    HtY = Hc.T @ Yc          # (128, 10)

    # GCV over a log-spaced grid of ridge penalties
    lambdas = np.logspace(-2, 5, 60)
    best_gcv = np.inf
    best_lam = 1.0
    I = np.eye(d)

    for lam in lambdas:
        A = HtH + lam * I
        # Solve for W (128, 10)
        W = np.linalg.solve(A, HtY)
        # Residual sum of squares
        Y_pred = Hc @ W
        rss = np.sum((Yc - Y_pred) ** 2)
        # Effective degrees of freedom: trace(H (HtH + lam I)^{-1} H^T)
        # = trace((HtH + lam I)^{-1} HtH)
        A_inv = np.linalg.inv(A)
        edf = np.trace(A_inv @ HtH)
        gcv = (rss / n) / (1.0 - edf / n) ** 2
        if gcv < best_gcv:
            best_gcv = gcv
            best_lam = lam

    # Final fit with the best lambda
    lam = best_lam
    A = HtH + lam * I
    W = np.linalg.solve(A, HtY)   # (128, 10)
    W_final = W.T                 # (10, 128)
    b_final = Y_mean - W_final @ H_mean

    # Clip to safe finite float32 range
    W_final = np.clip(W_final, -1000.0, 1000.0).astype(np.float32)
    b_final = np.clip(b_final, -1000.0, 1000.0).astype(np.float32)

    # Update the classifier weights
    w["fc2.weight"] = W_final
    w["fc2.bias"] = b_final

    diagnostics = {
        "lambda": float(lam),
        "fc2_weight_mean": float(np.mean(W_final)),
        "fc2_weight_std": float(np.std(W_final)),
        "fc2_bias_mean": float(np.mean(b_final)),
        "fc2_bias_std": float(np.std(b_final)),
        "gcv": float(best_gcv),
    }
    return {"weights": w, "diagnostics": diagnostics}
