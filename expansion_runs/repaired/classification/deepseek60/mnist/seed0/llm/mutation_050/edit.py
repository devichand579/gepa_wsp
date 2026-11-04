import numpy as np

def edit(w, f):
    """
    Hypothesis: The current network's fc1 features are already good, but the final
    linear classifier fc2 may be suboptimal. Instead of only re-fitting fc2 on the
    existing features (which previous attempts showed does not improve), we can
    replace the entire network with a closed-form linear classifier on the raw
    pixels, embedded exactly into the existing architecture.

    We compute a regularized LDA classifier on the full feedback pool (x, y).
    The resulting 10x784 weight matrix W and bias b are placed into the first 10
    rows of fc1.weight and fc1.bias (with a large positive constant C added to
    the bias to make ReLU act as identity). The remaining 118 hidden units are
    zeroed. fc2.weight is set to the 10x128 matrix [I_10 | 0] and fc2.bias is
    set to -C. This makes the network compute exactly logits = W @ x + b.

    Predicted effect: A reliable increase in held-out accuracy because LDA on
    raw pixels can capture global digit shapes better than the current fc1+fc2
    combination, especially with proper regularization. Expected accuracy
    around 92-95% (similar or slightly better than current 95.35%).
    """
    # Extract full feedback pool
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)  # [2000, 784]
    y = f["y"].astype(np.int64)                                 # [2000]
    N, D = X.shape
    C = 10

    # Compute class means and global mean
    mu_c = np.zeros((C, D), dtype=np.float64)
    counts = np.zeros(C, dtype=np.float64)
    for c in range(C):
        mask = (y == c)
        counts[c] = mask.sum()
        if counts[c] > 0:
            mu_c[c] = X[mask].mean(axis=0)
    mu = X.mean(axis=0)

    # Compute within-class covariance (pooled)
    S = np.zeros((D, D), dtype=np.float64)
    for c in range(C):
        mask = (y == c)
        if counts[c] > 1:
            Xc = X[mask] - mu_c[c]
            S += Xc.T @ Xc
    S /= (N - C)

    # Regularize covariance (ridge). Choose lambda based on trace.
    # A small ridge stabilizes the inverse.
    trace_S = np.trace(S)
    lam = 1e-3 * trace_S / D  # small relative ridge
    S_reg = S + lam * np.eye(D, dtype=np.float64)

    # Solve for LDA weights: W = (mu_c - mu) @ inv(S_reg)
    # We can solve S_reg @ W.T = (mu_c - mu).T
    diff = mu_c - mu  # [C, D]
    # Solve S_reg @ W.T = diff.T  => W.T = solve(S_reg, diff.T)
    W_T = np.linalg.solve(S_reg, diff.T)  # [D, C]
    W = W_T.T  # [C, D]

    # Bias: b_c = -0.5 * mu_c^T inv(S_reg) mu_c + log(prior)
    # We can compute using W: b_c = -0.5 * (mu_c @ W_c) + log(counts[c]/N)
    # Actually W_c = inv(S_reg) @ (mu_c - mu). So mu_c @ W_c = mu_c^T inv(S_reg) (mu_c - mu)
    # The standard LDA bias is -0.5 * mu_c^T inv(S_reg) mu_c + log(prior_c).
    # We can compute it directly.
    inv_S_mu = np.linalg.solve(S_reg, mu_c.T).T  # [C, D]
    b = -0.5 * np.sum(mu_c * inv_S_mu, axis=1) + np.log(counts / N)

    # Now we have a linear classifier: logits = W @ x + b
    # Embed into the network.
    # We need to choose C_large such that W @ x + b + C_large > 0 for all x in [0,1]^784.
    # The minimum of W_c @ x over x in [0,1]^D is sum_{j: W_cj < 0} W_cj.
    # So min_val_c = sum_{j: W_cj < 0} W_cj + b_c.
    min_vals = np.zeros(C, dtype=np.float64)
    for c in range(C):
        w_c = W[c]
        min_wx = np.sum(w_c[w_c < 0])  # sum of negative weights
        min_vals[c] = min_wx + b[c]
    C_large = -np.min(min_vals) + 1.0  # ensure positive

    # Prepare new weights
    fc1_weight = np.zeros_like(w["fc1.weight"], dtype=np.float32)
    fc1_bias = np.zeros_like(w["fc1.bias"], dtype=np.float32)
    fc2_weight = np.zeros_like(w["fc2.weight"], dtype=np.float32)
    fc2_bias = np.zeros_like(w["fc2.bias"], dtype=np.float32)

    # Set first 10 rows of fc1 to W, bias to b + C_large
    fc1_weight[:10, :] = W.astype(np.float32)
    fc1_bias[:10] = (b + C_large).astype(np.float32)

    # fc2: first 10 columns identity, rest zero
    fc2_weight[:, :10] = np.eye(10, dtype=np.float32)
    fc2_bias[:] = -C_large

    # Update w
    w["fc1.weight"] = fc1_weight
    w["fc1.bias"] = fc1_bias
    w["fc2.weight"] = fc2_weight
    w["fc2.bias"] = fc2_bias

    # Diagnostics
    diagnostics = {
        "method": "lda_raw_pixels",
        "lambda": float(lam),
        "C_large": float(C_large),
        "min_vals": min_vals.tolist(),
        "W_norm": float(np.linalg.norm(W)),
        "b_norm": float(np.linalg.norm(b)),
        "fc1_weight_norm": float(np.linalg.norm(fc1_weight)),
        "fc2_weight_norm": float(np.linalg.norm(fc2_weight)),
    }

    return {"weights": w, "diagnostics": diagnostics}
