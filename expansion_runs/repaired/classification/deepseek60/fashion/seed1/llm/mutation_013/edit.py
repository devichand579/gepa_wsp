import numpy as np

def edit(w, f):
    # Read current weights
    conv1_w = w["conv1.weight"].copy()
    conv1_b = w["conv1.bias"].copy()
    conv2_w = w["conv2.weight"].copy()
    conv2_b = w["conv2.bias"].copy()
    head_w = w["head.weight"].copy()
    head_b = w["head.bias"].copy()

    # Feedback data
    X = f["features"].astype(np.float64)          # [2000, 1568]
    y = f["y"].astype(np.int64)                   # [2000]
    N, D = X.shape
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Closed-form ridge regression on the full feedback pool.
    # Use a small regularization to avoid numerical issues.
    # The feature dimension is 1568 and N=2000, so lambda=0.1 is safe.
    lam = 0.1
    XtX = X.T @ X
    XtY = X.T @ Y
    # Solve (X^T X + lam I) W^T = X^T Y
    # W is [C, D], so we solve for W^T [D, C]
    A = XtX + lam * np.eye(D, dtype=np.float64)
    Wt = np.linalg.solve(A, XtY)                  # [D, C]
    W_new = Wt.T                                  # [C, D]
    # Bias: use the mean of the targets minus the mean of the features times W
    b_new = (Y.mean(axis=0) - X.mean(axis=0) @ W_new.T)  # [C]

    # Now calibrate the new head by a scalar alpha.
    # Compute logits on the full feedback pool.
    logits = X @ W_new.T + b_new                  # [N, C]
    # Compute accuracy of the new head on the full pool.
    pred = np.argmax(logits, axis=1)
    acc = np.mean(pred == y)

    # Find alpha such that the average top-1 probability equals acc.
    # This is a monotonic function of alpha. We can solve it by a simple
    # bisection on a small interval. This is not an iterative training loop;
    # it is a closed-form calibration step on the already-fitted head.
    def mean_top_prob(alpha):
        z = alpha * logits
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p = p / p.sum(axis=1, keepdims=True)
        return np.mean(p.max(axis=1))

    # Bisection for alpha in [0.1, 20.0]
    lo, hi = 0.1, 20.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if mean_top_prob(mid) < acc:
            lo = mid
        else:
            hi = mid
    alpha = 0.5 * (lo + hi)

    # Apply calibration
    W_new *= alpha
    b_new *= alpha

    # Write back
    w["head.weight"] = W_new.astype(np.float32)
    w["head.bias"] = b_new.astype(np.float32)

    # Diagnostics
    diag = {
        "ridge_lambda": lam,
        "alpha": float(alpha),
        "feedback_accuracy": float(acc),
        "mean_top_prob_before": float(mean_top_prob(1.0)),
        "mean_top_prob_after": float(mean_top_prob(alpha)),
        "head_weight_mean": float(W_new.mean()),
        "head_weight_std": float(W_new.std()),
        "head_bias_mean": float(b_new.mean()),
        "head_bias_std": float(b_new.std()),
    }

    return {"weights": w, "diagnostics": diag}
