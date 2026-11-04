import numpy as np

def edit(w, f):
    # Extract feedback data
    x = f["x"]          # (4000, 64) int64
    y = f["y"]          # (4000,) int64
    
    N = x.shape[0]
    V = 2000            # vocabulary size
    D = 32              # embedding dimension
    
    # Build normalized bag-of-words matrix X (N, V)
    X = np.zeros((N, V), dtype=np.float32)
    rows = np.repeat(np.arange(N), x.shape[1])
    cols = x.flatten()
    mask = (cols != 0)
    np.add.at(X, (rows[mask], cols[mask]), 1.0)
    lengths = mask.reshape(N, x.shape[1]).sum(axis=1)
    lengths = np.maximum(lengths, 1)  # avoid division by zero
    X = X / lengths[:, None]
    
    # Ridge regression on normalized counts
    # Solve (X^T X + lambda I) w = X^T y  with intercept
    y_float = y.astype(np.float32) * 2 - 1  # -1, 1
    X_aug = np.hstack([X, np.ones((N, 1), dtype=np.float32)])
    lambda_reg = 1.0
    A = X_aug.T @ X_aug + lambda_reg * np.eye(V + 1, dtype=np.float32)
    b_vec = X_aug.T @ y_float
    w_aug = np.linalg.solve(A, b_vec)
    w_token = w_aug[:V]          # (2000,)
    bias = w_aug[V]              # scalar
    
    # Design embedding: embed[token] = A_const * u + w_token * v
    # where u, v are zero-mean, unit-variance, orthogonal vectors in R^D
    rng = np.random.RandomState(42)
    u = rng.randn(D).astype(np.float32)
    u -= u.mean()
    u /= u.std()
    v = rng.randn(D).astype(np.float32)
    v -= v.mean()
    v -= (v @ u) / (u @ u) * u   # orthogonalize against u
    v /= v.std()
    
    A_const = 100.0
    
    # Build embedding matrix
    embed = np.zeros((V, D), dtype=np.float32)
    for t in range(V):
        embed[t] = A_const * u + w_token[t] * v
    # Padding token (0) gets no token-specific weight
    embed[0] = A_const * u
    
    # Head: we want logit_diff = (1/L) sum w_token * count + bias
    # After LayerNorm, pooled ≈ u + (1/L) sum w_token * (v / A_const)
    # So head_diff = (A_const / ||v||^2) * v  (since ||v||^2 = D)
    head_diff = (A_const / D) * v   # (32,)
    head_weight = np.zeros((2, D), dtype=np.float32)
    head_weight[0] = -0.5 * head_diff
    head_weight[1] =  0.5 * head_diff
    head_bias = np.array([-0.5 * bias, 0.5 * bias], dtype=np.float32)
    
    # Write into w
    w["embed.weight"] = embed.astype(np.float32)
    w["head.weight"] = head_weight.astype(np.float32)
    w["head.bias"] = head_bias.astype(np.float32)
    
    # Ensure all other parameters are zero (they already are, but just in case)
    for key in w:
        if key not in ["embed.weight", "head.weight", "head.bias"]:
            if w[key].ndim == 1:
                w[key][:] = 0.0
            else:
                w[key][:] = 0.0
    # Norm weights should be 1, biases 0
    for key in ["block0.norm1.weight", "block0.norm2.weight",
                "block1.norm1.weight", "block1.norm2.weight",
                "norm.weight"]:
        if key in w:
            w[key][:] = 1.0
    for key in ["block0.norm1.bias", "block0.norm2.bias",
                "block1.norm1.bias", "block1.norm2.bias",
                "norm.bias"]:
        if key in w:
            w[key][:] = 0.0
    
    # Diagnostics
    diag = {
        "w_token_mean": float(w_token.mean()),
        "w_token_std": float(w_token.std()),
        "w_token_max_abs": float(np.abs(w_token).max()),
        "bias": float(bias),
        "A_const": A_const,
        "embed_norm_mean": float(np.linalg.norm(embed, axis=1).mean()),
        "head_diff_norm": float(np.linalg.norm(head_diff)),
    }
    return {"weights": w, "diagnostics": diag}
