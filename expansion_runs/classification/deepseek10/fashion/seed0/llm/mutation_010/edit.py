import numpy as np
import math

def edit(w, f):
    # Extract features and labels from the full feedback pool
    X = f["features"].astype(np.float64)  # [2000, 1568]
    y = f["y"].astype(np.int64)           # [2000]
    n, d = X.shape
    classes = np.unique(y)
    C = len(classes)  # should be 10

    # Compute class means and pooled within-class covariance for LDA
    mu = np.zeros((C, d), dtype=np.float64)
    counts = np.zeros(C, dtype=np.float64)
    for i, c in enumerate(classes):
        mask = (y == c)
        counts[i] = np.sum(mask)
        if counts[i] > 0:
            mu[i] = np.mean(X[mask], axis=0)

    # Compute within-class scatter Sw = sum_c sum_{i in c} (x_i - mu_c)(x_i - mu_c)^T
    Sw = np.zeros((d, d), dtype=np.float64)
    for i, c in enumerate(classes):
        mask = (y == c)
        if counts[i] > 0:
            Xc = X[mask] - mu[i]
            Sw += Xc.T @ Xc

    # Add small ridge for numerical stability
    ridge = 1e-3
    Sw_reg = Sw + ridge * np.eye(d, dtype=np.float64)

    # Solve for LDA weights: W = Sw^{-1} mu_c
    # We can solve Sw_reg @ W.T = mu.T  => W.T = solve(Sw_reg, mu.T)
    W = np.linalg.solve(Sw_reg, mu.T).T  # shape [C, d]
    # Bias: b_c = -0.5 * mu_c^T Sw^{-1} mu_c + log(prior_c)
    # Compute Sw^{-1} mu_c for each c: this is W.T columns? Actually W = Sw^{-1} mu_c, so mu_c^T Sw^{-1} mu_c = mu_c^T W_c
    b = np.zeros(C, dtype=np.float64)
    for i, c in enumerate(classes):
        b[i] = -0.5 * np.dot(mu[i], W[i]) + np.log(counts[i] / n)

    # Now we have LDA head: weight = W, bias = b
    # Scale the head to make logits confident. Choose scale T such that max abs weight <= 1000.
    max_w = np.max(np.abs(W))
    max_b = np.max(np.abs(b))
    # We can scale by up to 1000 / max(max_w, max_b) but we want large confidence.
    # Let's scale so that max abs weight is around 1000 (safe).
    if max_w > 0:
        T = 1000.0 / max_w
    else:
        T = 1000.0
    # Also ensure bias doesn't exceed 1000
    if max_b * T > 1000.0:
        T = 1000.0 / max_b

    # Apply scaling
    W_scaled = W * T
    b_scaled = b * T

    # Update the head weights and bias in w
    # w["head.weight"] shape [10, 1568], w["head.bias"] shape [10]
    w["head.weight"] = W_scaled.astype(np.float32)
    w["head.bias"] = b_scaled.astype(np.float32)

    # Optionally, we could also scale conv layers, but not needed.
    # Ensure all weights are finite and within bounds
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)

    # Diagnostics
    diagnostics = {
        "lda_ridge": ridge,
        "scale_T": float(T),
        "head_weight_max_abs": float(np.max(np.abs(W_scaled))),
        "head_bias_max_abs": float(np.max(np.abs(b_scaled))),
        "head_weight_mean_abs": float(np.mean(np.abs(W_scaled))),
        "head_bias_mean_abs": float(np.mean(np.abs(b_scaled))),
    }

    return {"weights": w, "diagnostics": diagnostics}
