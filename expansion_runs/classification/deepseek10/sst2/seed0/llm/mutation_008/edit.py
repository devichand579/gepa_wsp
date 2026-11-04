import numpy as np

def edit(w, f):
    # Read full feedback pool
    x = f["x"]          # [4000, 64] int64
    y = f["y"]          # [4000] int64
    N, L = x.shape
    V = w["embed.weight"].shape[0]  # 2000

    # Build token-count matrix (term frequencies) with length normalization
    # We use masked mean pooling over non-pad tokens, so features are counts / length.
    # To avoid division by zero, compute lengths first.
    lengths = np.maximum((x != 0).sum(axis=1), 1).astype(np.float32)  # [N]
    X = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        # Count non-pad tokens
        tokens = x[i][x[i] != 0]
        if len(tokens) > 0:
            np.add.at(X[i], tokens, 1.0)
        X[i] /= lengths[i]

    # Add bias column
    Xb = np.concatenate([X, np.ones((N, 1), dtype=np.float32)], axis=1)  # [N, V+1]

    # Closed-form ridge regression: solve (X^T X + lambda I) beta = X^T y
    # Use a small lambda for numerical stability and mild regularization.
    lam = 1.0
    A = Xb.T @ Xb + lam * np.eye(V + 1, dtype=np.float32)
    b = Xb.T @ y.astype(np.float32)
    beta = np.linalg.solve(A, b)  # [V+1]

    w_token = beta[:V]      # token weights
    bias = beta[V]          # scalar bias

    # Scale token weights so that they fit comfortably in the first coordinate
    # while keeping the embedding vectors mean-0 and variance-1.
    max_abs = np.max(np.abs(w_token)) + 1e-8
    # We want first coordinate = w_token / scale, and we need |w_token/scale| <= sqrt(32)
    # to be able to construct a mean-0, var-1 vector. Use a safe scale.
    scale = max(max_abs / (np.sqrt(32) * 0.9), 1e-8)

    # Construct embedding vectors: shape [V, 32]
    # For each token, set first coordinate = w_token / scale.
    # The remaining 31 coordinates are chosen to make mean=0 and variance=1.
    # Let c = w_token / scale. We need sum(v) = 0 and sum(v^2) = 32.
    # Set v[0] = c. For i>=1, set v[i] = a + d_i, where sum(d_i)=0.
    # A simple symmetric construction: v[i] = -c/31 for all i>=1.
    # Then sum = c + 31*(-c/31) = 0.
    # Variance: sum(v^2) = c^2 + 31*(c/31)^2 = c^2 + c^2/31 = c^2 * 32/31.
    # To make sum(v^2) = 32, we need c^2 * 32/31 = 32 => c^2 = 31 => |c| = sqrt(31).
    # This fixes |c|, which is too restrictive. Instead, we can use a different construction:
    # Let v[0] = c. We need sum_{i=1}^{31} v[i] = -c and sum_{i=1}^{31} v[i]^2 = 32 - c^2.
    # We can set v[i] = -c/31 + d_i, where sum d_i = 0 and sum d_i^2 = 32 - c^2 - c^2/31.
    # This is possible as long as 32 - c^2 - c^2/31 >= 0 => c^2 <= 32 * 31/32 = 31.
    # So |c| <= sqrt(31) ≈ 5.568. Our scale ensures |c| <= sqrt(32)*0.9 ≈ 5.09, so safe.
    # We can simply set v[i] = -c/31 for i>=1, and then add a small zero-mean perturbation
    # to adjust the norm if needed. But to keep it simple and exact, we can use a two-step:
    # First, set v[0] = c, v[i] = -c/31 for i>=1. This gives mean 0.
    # Its squared norm is c^2 * 32/31. We want norm^2 = 32.
    # So we can scale the entire vector by sqrt(32 / (c^2 * 32/31)) = sqrt(31 / c^2).
    # But scaling would change the first coordinate. Instead, we can adjust the other coordinates
    # by adding a vector orthogonal to [1,1,...] and to the first coordinate.
    # Simpler: use a fixed orthogonal basis for the remaining 31 dimensions.
    # We can construct v as follows:
    # v[0] = c
    # For i=1..31, v[i] = -c/31 + alpha * u[i], where u is a fixed unit vector orthogonal to ones.
    # Choose u = [1, -1, 0, ..., 0] normalized? Actually we need sum u = 0 and sum u^2 = 1.
    # Let u = [1, -1, 0, ..., 0] / sqrt(2). Then sum u = 0, sum u^2 = 1.
    # Then sum v = c + 31*(-c/31) + alpha * sum u = 0.
    # sum v^2 = c^2 + 31*(c^2/31^2) + alpha^2 * sum u^2 = c^2 + c^2/31 + alpha^2.
    # We want this = 32. So alpha^2 = 32 - c^2 * 32/31.
    # This is valid as long as c^2 <= 31. Our scale ensures c^2 <= 32*0.81 = 25.9, so safe.
    # We'll use this construction.

    embed = w["embed.weight"].copy()  # [V, 32]
    # Zero out all embeddings first (including padding token 0)
    embed[:] = 0.0

    # Fixed orthogonal vector u for the remaining 31 dims
    u = np.zeros(32, dtype=np.float32)
    u[1] = 1.0 / np.sqrt(2.0)
    u[2] = -1.0 / np.sqrt(2.0)

    for token in range(V):
        c = w_token[token] / scale
        # Clamp c to safe range
        c = np.clip(c, -np.sqrt(31.0) * 0.99, np.sqrt(31.0) * 0.99)
        v = np.zeros(32, dtype=np.float32)
        v[0] = c
        v[1:] = -c / 31.0
        # Add alpha * u to make norm exactly sqrt(32)
        alpha_sq = 32.0 - c * c * (32.0 / 31.0)
        if alpha_sq > 0:
            alpha = np.sqrt(alpha_sq)
            v += alpha * u
        else:
            # Fallback: just normalize to norm sqrt(32) (should not happen)
            v = v / np.linalg.norm(v) * np.sqrt(32.0)
        embed[token] = v

    # Ensure padding token embedding is zero (doesn't matter for pooling, but safe)
    embed[0] = 0.0

    # Set head to read first coordinate.
    # pooled[0] = mean(v[0]) = mean(w_token / scale) = (1/scale) * mean(w_token)
    # We want logit = mean(w_token) + bias.
    # So head.weight[1,0] = scale, head.weight[0,0] = -scale
    # head.bias[1] = bias, head.bias[0] = -bias
    head_w = np.zeros_like(w["head.weight"])
    head_w[1, 0] = scale
    head_w[0, 0] = -scale
    head_b = np.array([-bias, bias], dtype=np.float32)

    # Update weights
    w["embed.weight"] = embed.astype(np.float32)
    w["head.weight"] = head_w.astype(np.float32)
    w["head.bias"] = head_b.astype(np.float32)

    # Zero out positional embedding and block projections/feed-forward outputs to keep blocks identity
    w["pos.weight"][:] = 0.0
    for blk in ["block0", "block1"]:
        w[f"{blk}.proj.weight"][:] = 0.0
        w[f"{blk}.proj.bias"][:] = 0.0
        w[f"{blk}.ff2.weight"][:] = 0.0
        w[f"{blk}.ff2.bias"][:] = 0.0
        # Keep norm1/norm2 as identity (already 1/0), but ensure
        w[f"{blk}.norm1.weight"][:] = 1.0
        w[f"{blk}.norm1.bias"][:] = 0.0
        w[f"{blk}.norm2.weight"][:] = 1.0
        w[f"{blk}.norm2.bias"][:] = 0.0

    # Final norm identity
    w["norm.weight"][:] = 1.0
    w["norm.bias"][:] = 0.0

    # Diagnostics
    diagnostics = {
        "scale": float(scale),
        "bias": float(bias),
        "max_abs_token_weight": float(max_abs),
        "embed_norm_mean": float(np.mean(np.linalg.norm(embed, axis=1))),
        "head_weight_norm": float(np.linalg.norm(head_w)),
        "head_bias": head_b.tolist(),
    }

    return {"weights": w, "diagnostics": diagnostics}
