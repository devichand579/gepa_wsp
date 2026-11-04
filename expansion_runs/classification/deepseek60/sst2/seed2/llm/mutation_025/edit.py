import numpy as np

def edit(w, f):
    # Extract training data
    X = f["x"]          # [4000, 64] int64 token ids
    y = f["y"]          # [4000] int64 labels
    N, L = X.shape
    V = w["embed.weight"].shape[0]  # 2000

    # Build bag-of-words matrix (excluding padding id 0)
    BOW = np.zeros((N, V), dtype=np.float64)
    for i in range(N):
        for t in range(L):
            tok = int(X[i, t])
            if tok != 0:
                BOW[i, tok] += 1.0

    # Convert labels to ±1
    y_pm = np.where(y == 1, 1.0, -1.0)

    # Ridge regression with small L2 penalty (closed form)
    # Solve (BOW^T BOW + lambda I) w = BOW^T y_pm
    # Use a modest lambda for stability. We can pick lambda via simple CV or just a fixed small value.
    # Since BOW is sparse and high-dimensional, lambda=1.0 is reasonable.
    lam = 1.0
    A = BOW.T @ BOW + lam * np.eye(V, dtype=np.float64)
    b = BOW.T @ y_pm
    w_lin = np.linalg.solve(A, b)  # [V]

    # Normalize w_lin to have max absolute value 1 (to fit into LayerNorm constraints)
    max_abs = np.max(np.abs(w_lin))
    if max_abs > 0:
        w_lin = w_lin / max_abs

    # We will encode w_lin into embed.weight.
    # Choose a fixed unit vector u in R^32 with zero mean and unit variance.
    # For simplicity, use a random vector and then orthogonalize against the all-ones vector.
    rng = np.random.RandomState(42)
    u = rng.randn(32)
    u = u - u.mean()
    u = u / np.linalg.norm(u)

    # For each token v, we want embed[v] such that LayerNorm(embed[v]) = w_lin[v] * u + orthogonal_noise.
    # Since LayerNorm normalizes to zero mean and unit variance, we can set:
    # embed[v] = w_lin[v] * u + sqrt(1 - w_lin[v]^2) * v_v
    # where v_v is any unit vector orthogonal to u and to the all-ones vector.
    # To keep it simple, we can just set embed[v] = w_lin[v] * u + sqrt(1 - w_lin[v]^2) * e_{v mod 31}
    # but we need to ensure zero mean and unit variance. Actually, if u has zero mean and unit variance,
    # and v_v has zero mean and unit variance and is orthogonal to u, then the combination has zero mean and unit variance.
    # We can construct a set of 31 orthonormal vectors orthogonal to u and to the all-ones vector.
    # Let's build an orthonormal basis for the subspace orthogonal to u and 1.
    # We have 32 dims. 1 is one direction. u is another (orthogonal to 1). So we have 30 dims left.
    # We can use a random matrix and orthogonalize.
    M = rng.randn(32, 30)
    # Orthogonalize against u and 1
    ones = np.ones(32)
    for j in range(30):
        M[:, j] -= (M[:, j] @ u) * u
        M[:, j] -= (M[:, j] @ ones) / 32.0 * ones
    # QR decomposition to get orthonormal basis
    Q, _ = np.linalg.qr(M)
    # Q is 32x30, columns orthonormal, orthogonal to u and 1.
    # We can use these 30 vectors as v_v for tokens. Since we have 2000 tokens, we cycle through them.
    # Actually, we only need the dot product with u to be w_lin[v]. The orthogonal part doesn't affect the head if head_diff = u.
    # So we can just set embed[v] = w_lin[v] * u + sqrt(1 - w_lin[v]^2) * Q[:, v % 30]
    # But we must ensure that for tokens with w_lin[v] = 0, the vector is still unit variance. sqrt(1-0)=1, so it's fine.
    # However, we need to handle the case where w_lin[v] might be > 1 in absolute value. We already normalized max_abs to 1.
    # So |w_lin[v]| <= 1.

    embed = np.zeros((V, 32), dtype=np.float32)
    for v in range(V):
        wv = w_lin[v]
        # Ensure wv in [-1, 1]
        wv = np.clip(wv, -1.0, 1.0)
        orth = Q[:, v % 30]
        vec = wv * u + np.sqrt(max(0.0, 1.0 - wv*wv)) * orth
        # Ensure zero mean and unit variance exactly (numerical)
        vec = vec - vec.mean()
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        embed[v] = vec.astype(np.float32)

    # Set embed.weight
    w["embed.weight"] = embed

    # Zero out pos
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)

    # Zero out transformer blocks (set all weights and biases to zero)
    for block in ["block0", "block1"]:
        for name in ["norm1.weight", "norm1.bias", "qkv.weight", "qkv.bias",
                     "proj.weight", "proj.bias", "norm2.weight", "norm2.bias",
                     "ff1.weight", "ff1.bias", "ff2.weight", "ff2.bias"]:
            key = f"{block}.{name}"
            if key in w:
                w[key] = np.zeros_like(w[key], dtype=np.float32)

    # Set final norm to identity (weight=1, bias=0)
    w["norm.weight"] = np.ones_like(w["norm.weight"], dtype=np.float32)
    w["norm.bias"] = np.zeros_like(w["norm.bias"], dtype=np.float32)

    # Set head: we want logit = pooled @ u + bias.
    # pooled = mean_t embed[token_t] = mean_t (w_lin[token_t] * u + orth) = (mean_t w_lin[token_t]) * u + mean_t orth.
    # The mean of orth is not necessarily zero, but it's small and random. We can just set head to u.
    # To be precise, we can compute the actual pooled features on the training data and fit the head.
    # Let's compute the pooled features for the training set using the new embed.
    # We need to simulate the forward pass up to the pooled feature.
    # Since blocks are zero, h = embed(ids) + pos = embed(ids). norm(h) = LayerNorm(embed(ids)) = embed(ids) (since we made them zero-mean unit-variance).
    # pooled = mean over non-padding tokens of embed[token].
    # We can compute this exactly.
    pooled = np.zeros((N, 32), dtype=np.float64)
    for i in range(N):
        count = 0
        for t in range(L):
            tok = int(X[i, t])
            if tok != 0:
                pooled[i] += embed[tok]
                count += 1
        if count > 0:
            pooled[i] /= count

    # Now fit a linear classifier on pooled features to predict y_pm.
    # We can use ridge regression on pooled features.
    # pooled is [N, 32]. Add bias? The head has bias, so we can include a column of ones.
    P = np.hstack([pooled, np.ones((N, 1))])
    # Ridge regression: solve (P^T P + lambda I) w = P^T y_pm
    lam_head = 1e-3
    A_head = P.T @ P + lam_head * np.eye(33)
    b_head = P.T @ y_pm
    w_head = np.linalg.solve(A_head, b_head)

    # w_head is [33]. The first 32 are head.weight[1] - head.weight[0] (since we want logit for class 1 minus class 0).
    # Actually, the network computes logits = head.weight @ pooled + head.bias.
    # We want logit_1 - logit_0 = (head.weight[1] - head.weight[0]) @ pooled + (head.bias[1] - head.bias[0]).
    # So we can set head.weight[1] = w_head[:32], head.weight[0] = -w_head[:32], and head.bias[1] = w_head[32], head.bias[0] = -w_head[32].
    # This makes the difference exactly w_head.
    head_weight = np.zeros((2, 32), dtype=np.float32)
    head_bias = np.zeros(2, dtype=np.float32)
    head_weight[1] = w_head[:32]
    head_weight[0] = -w_head[:32]
    head_bias[1] = w_head[32]
    head_bias[0] = -w_head[32]

    w["head.weight"] = head_weight
    w["head.bias"] = head_bias

    # Diagnostics
    diag = {
        "w_lin_max_abs": float(np.max(np.abs(w_lin))),
        "w_lin_mean": float(np.mean(w_lin)),
        "w_lin_std": float(np.std(w_lin)),
        "head_weight_norm": float(np.linalg.norm(head_weight[1])),
        "head_bias_1": float(head_bias[1]),
        "embed_row_mean_maxabs": float(np.max(np.abs(embed.mean(axis=1)))),
        "embed_row_std_maxdev": float(np.max(np.abs(embed.std(axis=1) - 1.0))),
    }

    return {"weights": w, "diagnostics": diag}
