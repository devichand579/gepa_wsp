import numpy as np

def edit(w, f):
    # Read full feedback features and labels
    X = np.asarray(f["features"], dtype=np.float64)   # [2000, 1568]
    y = np.asarray(f["y"], dtype=np.int64)            # [2000]
    n, p = X.shape
    C = int(y.max()) + 1

    # Center features globally
    mu = X.mean(axis=0)
    Xc = X - mu

    # Class means on centered features
    Mu = np.zeros((C, p), dtype=np.float64)
    counts = np.zeros(C, dtype=np.float64)
    for c in range(C):
        idx = (y == c)
        counts[c] = idx.sum()
        if counts[c] > 0:
            Mu[c] = Xc[idx].mean(axis=0)

    # Sample covariance of centered features
    S = (Xc.T @ Xc) / n
    m = np.trace(S) / p

    # Ledoit-Wolf optimal shrinkage intensity toward m * I
    # b2 = (1/n^2) * ( sum_k ||x_k||^4 - n ||S||_F^2 )
    # d2 = ||S - m I||_F^2 = ||S||_F^2 - m^2 p
    S_fro2 = np.sum(S * S)
    x_norm4_sum = np.sum(np.sum(Xc * Xc, axis=1) ** 2)
    b2 = (x_norm4_sum - n * S_fro2) / (n * n)
    d2 = S_fro2 - m * m * p
    if d2 > 0:
        alpha = min(1.0, max(0.0, b2 / d2))
    else:
        alpha = 1.0

    # Shrunk covariance
    Sigma = (1.0 - alpha) * S + alpha * m * np.eye(p, dtype=np.float64)

    # Solve Sigma W = Mu^T for class weights
    # Add tiny jitter for numerical safety
    Sigma = Sigma + 1e-8 * np.eye(p, dtype=np.float64)
    W = np.linalg.solve(Sigma, Mu.T)   # [p, C]

    # Bias for LDA: -0.5 * mu_c^T Sigma^{-1} mu_c + log prior
    # Here Mu rows are centered class means.
    quad = np.sum(Mu.T * W, axis=0)    # [C]
    priors = counts / n
    bias = -0.5 * quad + np.log(np.maximum(priors, 1e-12))

    # Convert to float32 and write head
    head_weight = W.T.astype(np.float32)   # [10, 1568]
    head_bias = bias.astype(np.float32)    # [10]

    # Clip to allowed range
    head_weight = np.clip(head_weight, -1000.0, 1000.0)
    head_bias = np.clip(head_bias, -1000.0, 1000.0)

    w["head.weight"] = head_weight
    w["head.bias"] = head_bias

    diagnostics = {
        "alpha_shrinkage": float(alpha),
        "trace_cov": float(m),
        "head_weight_norm": float(np.linalg.norm(head_weight)),
        "head_bias_norm": float(np.linalg.norm(head_bias)),
        "head_weight_max_abs": float(np.max(np.abs(head_weight))),
        "head_bias_max_abs": float(np.max(np.abs(head_bias))),
        "class_counts": counts.astype(int).tolist(),
    }
    return {"weights": w, "diagnostics": diagnostics}
