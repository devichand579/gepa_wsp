import numpy as np
import math

def edit(w, f):
    # Extract full-pool features and labels
    X = f["features"].astype(np.float64)   # (2000, 1568)
    y = f["y"].astype(np.int64)            # (2000,)
    n, p = X.shape
    C = 10

    # Class means
    mu = np.zeros((C, p), dtype=np.float64)
    for k in range(C):
        mu[k] = X[y == k].mean(axis=0)

    # Pooled within-class covariance
    S = np.zeros((p, p), dtype=np.float64)
    for k in range(C):
        Xc = X[y == k] - mu[k]
        S += Xc.T @ Xc
    S /= (n - C)

    # Ledoit-Wolf shrinkage towards mu * I
    mu_trace = np.trace(S) / p
    F = mu_trace * np.eye(p)
    # d2 = || S - F ||_F^2 / p
    d2 = np.sum((S - F) ** 2) / p
    # b2 = (1/n^2) * sum_i || x_i x_i^T - S ||_F^2
    # Using the simplified identity: sum_i || x_i x_i^T - S ||_F^2 = sum_i (x_i^T x_i)^2 - n * trace(S^2)
    # We compute on the centered within-class data? Actually Ledoit-Wolf is applied to the pooled covariance.
    # We can use the full centered data? The pooled covariance is computed from within-class centered data.
    # To compute b2 correctly, we need the individual centered vectors. We can reconstruct them by centering each class.
    # But we can also approximate by using the full data centered by global mean. For simplicity and robustness,
    # we use a small fixed shrinkage intensity. A typical value that works well is 0.1.
    # However, we can compute the optimal shrinkage using the formula on the within-class centered data.
    # Let's build the within-class centered data matrix Xc_all (n, p).
    Xc_all = np.zeros_like(X)
    for k in range(C):
        idx = (y == k)
        Xc_all[idx] = X[idx] - mu[k]
    # Now compute b2 using Xc_all
    # b2 = (1/n^2) * sum_i || x_i x_i^T - S ||_F^2
    # We can compute sum_i (x_i^T x_i)^2 efficiently:
    sq_norms = np.sum(Xc_all ** 2, axis=1)  # (n,)
    sum_sq_norms_sq = np.sum(sq_norms ** 2)
    trace_S2 = np.sum(S * S)
    b2 = (sum_sq_norms_sq - n * trace_S2) / (n ** 2)
    # Optimal shrinkage intensity
    if d2 > 1e-12:
        lam = min(1.0, max(0.0, b2 / d2))
    else:
        lam = 0.1
    # Regularized covariance
    S_reg = (1.0 - lam) * S + lam * F
    # Add a tiny ridge for numerical stability
    S_reg += 1e-6 * np.eye(p)

    # Invert covariance
    inv_S = np.linalg.inv(S_reg)

    # LDA weights and biases
    W = mu @ inv_S  # (C, p)
    # Bias: -0.5 * mu_k^T Sigma^-1 mu_k + log(prior)
    prior = np.array([np.sum(y == k) for k in range(C)], dtype=np.float64) / n
    b = -0.5 * np.sum(mu * (mu @ inv_S), axis=1) + np.log(prior + 1e-12)

    # Convert to float32 and assign
    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "shrinkage_lambda": float(lam),
        "trace_S": float(np.trace(S)),
        "trace_S_reg": float(np.trace(S_reg)),
        "W_norm": float(np.linalg.norm(W)),
        "b_norm": float(np.linalg.norm(b)),
        "prior": prior.tolist(),
    }
    return {"weights": w, "diagnostics": diagnostics}
