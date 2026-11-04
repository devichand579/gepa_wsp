import numpy as np

def edit(w, f):
    # Extract training data
    X = f["x"]  # [4000, 64] int64
    y = f["y"]  # [4000] int64
    N, L = X.shape
    V = w["embed.weight"].shape[0]  # 2000
    D = w["embed.weight"].shape[1]  # 32

    # Compute token counts per class
    # Flatten X and y
    X_flat = X.reshape(-1)
    y_flat = np.repeat(y, L)
    # Mask for non-pad tokens (token != 0)
    mask = X_flat != 0
    X_nonpad = X_flat[mask]
    y_nonpad = y_flat[mask]

    # Count occurrences of each token in positive and negative
    c_pos = np.bincount(X_nonpad[y_nonpad == 1], minlength=V).astype(np.float64)
    c_neg = np.bincount(X_nonpad[y_nonpad == 0], minlength=V).astype(np.float64)
    N_pos = np.sum(y == 1)
    N_neg = np.sum(y == 0)

    # LDA-like weight: difference in frequency
    # Add smoothing to avoid division by zero
    eps = 1e-6
    w_t = (c_pos + eps) / (N_pos + eps * V) - (c_neg + eps) / (N_neg + eps * V)
    # Standardize w_t to zero mean, unit variance (over non-pad tokens)
    # Only consider tokens that appear at least once
    valid = (c_pos + c_neg) > 0
    if np.sum(valid) > 1:
        w_t_valid = w_t[valid]
        w_t_mean = np.mean(w_t_valid)
        w_t_std = np.std(w_t_valid) + 1e-8
        w_t = (w_t - w_t_mean) / w_t_std
    w_t[0] = 0.0  # pad token

    # Frequency feature: log(1 + total count)
    total_count = c_pos + c_neg
    f_t = np.log1p(total_count)
    # Standardize frequency feature over valid tokens
    if np.sum(valid) > 1:
        f_t_valid = f_t[valid]
        f_mean = np.mean(f_t_valid)
        f_std = np.std(f_t_valid) + 1e-8
        f_t = (f_t - f_mean) / f_std
    f_t[0] = 0.0

    # Constant feature: 1 for non-pad, 0 for pad
    const_t = np.ones(V, dtype=np.float64)
    const_t[0] = 0.0

    # Build new embedding matrix: columns 0,1,2 are w_t, const_t, f_t; rest zero
    new_embed = np.zeros((V, D), dtype=np.float32)
    new_embed[:, 0] = w_t.astype(np.float32)
    new_embed[:, 1] = const_t.astype(np.float32)
    new_embed[:, 2] = f_t.astype(np.float32)

    # Update weights: embed, pos, blocks, norm
    w["embed.weight"] = new_embed
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)

    # Zero out all transformer blocks (they are already zero, but ensure)
    for name in list(w.keys()):
        if name.startswith("block0.") or name.startswith("block1."):
            w[name] = np.zeros_like(w[name], dtype=np.float32)

    # Set final norm to identity (weight=1, bias=0)
    w["norm.weight"] = np.ones(D, dtype=np.float32)
    w["norm.bias"] = np.zeros(D, dtype=np.float32)

    # Now compute pooled features for all training examples using the new embed
    # Forward pass: h = embed(ids) (pos=0, blocks identity)
    # For each example, get token embeddings, apply LayerNorm, then mean over non-pad tokens.
    # We can do this efficiently with NumPy.
    # embed_matrix: [V, D]
    embed_matrix = new_embed  # float32

    # Gather embeddings for all tokens in X: shape [N, L, D]
    # X is [N, L]
    emb = embed_matrix[X]  # [N, L, D]

    # Apply LayerNorm over last dimension (D)
    # mean over D: [N, L, 1]
    mean = np.mean(emb, axis=2, keepdims=True)
    var = np.var(emb, axis=2, keepdims=True)
    # epsilon = 1e-5
    eps_norm = 1e-5
    normed = (emb - mean) / np.sqrt(var + eps_norm)  # [N, L, D]

    # Masked mean over tokens: only non-pad (ids != 0)
    mask_tokens = (X != 0).astype(np.float32)  # [N, L]
    # Sum over L, divide by count
    sum_normed = np.sum(normed * mask_tokens[:, :, None], axis=1)  # [N, D]
    counts = np.sum(mask_tokens, axis=1, keepdims=True)  # [N, 1]
    counts = np.maximum(counts, 1.0)  # avoid division by zero
    pooled = sum_normed / counts  # [N, D]

    # Now fit ridge regression head: predict y (0/1) from pooled features
    # Add bias term by augmenting features with a column of ones
    pooled_aug = np.hstack([pooled, np.ones((N, 1), dtype=np.float32)])  # [N, D+1]
    # Ridge regression: solve (X^T X + lambda I) beta = X^T y
    # Use lambda = 1.0 for regularization
    lambda_reg = 1.0
    D_aug = pooled_aug.shape[1]
    A = pooled_aug.T @ pooled_aug + lambda_reg * np.eye(D_aug, dtype=np.float32)
    b = pooled_aug.T @ y.astype(np.float32)
    beta = np.linalg.solve(A, b)  # [D+1]

    # Extract head weights and bias
    head_weight = beta[:D].reshape(1, D)  # [1, D]
    head_bias = beta[D].reshape(1)  # [1]

    # We need two output classes. The ridge predicts y directly (0 or 1).
    # We can set head.weight[0] = -head_weight, head.weight[1] = head_weight,
    # and head.bias[0] = -head_bias, head.bias[1] = head_bias.
    # This gives logits: class 0 = - (w·x + b), class 1 = w·x + b.
    # Then argmax is equivalent to thresholding at 0.
    new_head_weight = np.zeros((2, D), dtype=np.float32)
    new_head_weight[0] = -head_weight[0]
    new_head_weight[1] = head_weight[0]
    new_head_bias = np.zeros(2, dtype=np.float32)
    new_head_bias[0] = -head_bias[0]
    new_head_bias[1] = head_bias[0]

    w["head.weight"] = new_head_weight
    w["head.bias"] = new_head_bias

    # Diagnostics
    diagnostics = {
        "train_accuracy": float(np.mean((pooled_aug @ beta > 0.5).astype(int) == y)),
        "head_weight_norm": float(np.linalg.norm(new_head_weight)),
        "head_bias": new_head_bias.tolist(),
        "w_t_mean": float(np.mean(w_t[valid])) if np.sum(valid) > 0 else 0.0,
        "w_t_std": float(np.std(w_t[valid])) if np.sum(valid) > 0 else 0.0,
        "f_t_mean": float(np.mean(f_t[valid])) if np.sum(valid) > 0 else 0.0,
        "f_t_std": float(np.std(f_t[valid])) if np.sum(valid) > 0 else 0.0,
    }

    return {"weights": w, "diagnostics": diagnostics}
