import numpy as np

def edit(w, f):
    """
    Hypothesis: The current network is severely overfit: the head weights are huge
    (up to ±1000) and the training cross-entropy is ~6e-6. The convolutional feature
    extractor is likely already good (100% training accuracy), but the linear head
    has memorized the training set. Replacing only the head with a regularized linear
    classifier fitted on the full feedback feature matrix should reduce overfitting
    and improve held-out accuracy. I will use ridge regression on standardized
    features, with the ridge penalty chosen by generalized cross-validation (GCV)
    computed in closed form via SVD. This avoids iterative training loops.

    Predicted effect: Feedback accuracy may drop from 100% to ~95-99%, but the
    cross-entropy should become reasonable and held-out accuracy should increase
    due to much smaller, better-conditioned head weights.
    """
    # Extract features and labels from feedback pool
    F = f["features"].astype(np.float64)  # (2000, 1568)
    y = f["y"].astype(np.int64)           # (2000,)
    n, d = F.shape
    num_classes = 10

    # Center and standardize features
    mean_F = F.mean(axis=0)
    std_F = F.std(axis=0)
    std_F[std_F < 1e-6] = 1.0
    F_std = (F - mean_F) / std_F

    # One-hot targets, centered
    T = np.zeros((n, num_classes), dtype=np.float64)
    T[np.arange(n), y] = 1.0
    mean_T = T.mean(axis=0)
    T_centered = T - mean_T

    # SVD of standardized features
    U, S, Vt = np.linalg.svd(F_std, full_matrices=False)
    S2 = S**2
    UT_T = U.T @ T_centered                     # (min(n,d), 10)
    UT_T_sq_sum = np.sum(UT_T**2, axis=1)       # (min(n,d),)
    T_centered_sq = np.sum(T_centered**2)

    # GCV over a range of lambdas
    lambdas = np.logspace(-3, 6, 100)
    best_lam = None
    best_gcv = np.inf
    for lam in lambdas:
        d_factor = S2 / (S2 + lam)
        trace_H = np.sum(d_factor)
        rss = T_centered_sq - np.sum((2 * d_factor - d_factor**2) * UT_T_sq_sum)
        gcv = (rss / n) / ((1 - trace_H / n)**2)
        if gcv < best_gcv:
            best_gcv = gcv
            best_lam = lam

    # Compute ridge solution on standardized features
    d_factor = S2 / (S2 + best_lam)
    W_std = Vt.T @ (d_factor[:, None] * UT_T)   # (d, 10)

    # Convert back to original feature space
    W_orig = W_std / std_F[:, None]             # (d, 10)
    head_weight = W_orig.T                      # (10, d)
    head_bias = mean_T - mean_F @ W_orig        # (10,)

    # Update the head weights and biases
    w["head.weight"] = head_weight.astype(np.float32)
    w["head.bias"] = head_bias.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lam),
        "best_gcv": float(best_gcv),
        "head_weight_norm": float(np.linalg.norm(head_weight)),
        "head_bias_norm": float(np.linalg.norm(head_bias)),
        "head_weight_max_abs": float(np.max(np.abs(head_weight))),
        "head_bias_max_abs": float(np.max(np.abs(head_bias))),
    }

    return {"weights": w, "diagnostics": diagnostics}
