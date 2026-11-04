import numpy as np

def edit(w, f):
    # Extract training data
    X = f["x"]          # (4000, 64) int64
    y = f["y"]          # (4000,) int64
    N, L = X.shape

    # Compute token scores: smoothed log odds ratio
    # Count occurrences in positive and negative examples
    pos_counts = np.zeros(2000, dtype=np.float64)
    neg_counts = np.zeros(2000, dtype=np.float64)
    for i in range(N):
        tokens = X[i]
        if y[i] == 1:
            pos_counts[tokens] += 1
        else:
            neg_counts[tokens] += 1
    # Remove padding token (0) from counts? It's fine, but we can leave it.
    # Smoothing parameter
    alpha = 1.0
    scores = np.log((pos_counts + alpha) / (neg_counts + alpha))
    # Normalize scores to have mean 0 and std 1 for numerical stability
    scores = (scores - scores.mean()) / (scores.std() + 1e-8)

    # Set embed weights: dim 0 = score, dim 1 = 1, rest 0
    embed = np.zeros((2000, 32), dtype=np.float32)
    embed[:, 0] = scores
    embed[:, 1] = 1.0
    w["embed.weight"] = embed

    # Set pos weights to zero (already zero, but ensure)
    w["pos.weight"] = np.zeros((64, 32), dtype=np.float32)

    # Ensure blocks are identity (all zero) - they already are, but just in case
    for block in ["block0", "block1"]:
        for name in ["qkv.weight", "qkv.bias", "proj.weight", "proj.bias",
                     "ff1.weight", "ff1.bias", "ff2.weight", "ff2.bias"]:
            key = f"{block}.{name}"
            if key in w:
                w[key] = np.zeros_like(w[key])
        # Norm layers: set weight to 1, bias to 0
        for name in ["norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"]:
            key = f"{block}.{name}"
            if key in w:
                if "weight" in name:
                    w[key] = np.ones_like(w[key])
                else:
                    w[key] = np.zeros_like(w[key])
    # Final norm: weight=1, bias=0
    w["norm.weight"] = np.ones(32, dtype=np.float32)
    w["norm.bias"] = np.zeros(32, dtype=np.float32)

    # Compute pooled features for all training examples using new embed
    # Forward: embed lookup, LayerNorm per token, mean over non-pad tokens
    # We can vectorize this.
    # embed: (2000, 32)
    # X: (N, L)
    # Get embeddings: (N, L, 32)
    emb = embed[X]  # (N, L, 32)
    # LayerNorm per token (over last dim)
    mean = emb.mean(axis=2, keepdims=True)
    var = emb.var(axis=2, keepdims=True)
    ln = (emb - mean) / np.sqrt(var + 1e-5)
    # Mask: non-pad tokens (id != 0)
    mask = (X != 0).astype(np.float32)  # (N, L)
    mask_sum = mask.sum(axis=1, keepdims=True)  # (N, 1)
    mask_sum = np.maximum(mask_sum, 1.0)  # avoid division by zero
    pooled = (ln * mask[:, :, None]).sum(axis=1) / mask_sum  # (N, 32)

    # Fit head using ridge regression (closed-form)
    # We want to predict y (0 or 1). Use linear regression with ridge.
    # Add bias term by augmenting features with a column of ones.
    X_feat = np.concatenate([pooled, np.ones((N, 1), dtype=np.float32)], axis=1)  # (N, 33)
    # Ridge regression: w = (X^T X + lambda I)^{-1} X^T y
    # Use lambda = 1.0 for regularization
    lam = 1.0
    A = X_feat.T @ X_feat + lam * np.eye(33, dtype=np.float32)
    b = X_feat.T @ y.astype(np.float32)
    w_ridge = np.linalg.solve(A, b)  # (33,)

    # Set head weights: head.weight is (2, 32), head.bias is (2,)
    # We have a single linear score: logit = w_ridge[:32] @ pooled + w_ridge[32]
    # We can set class 0 logit = 0, class 1 logit = score.
    # Or set both symmetrically. Let's set:
    # head.weight[0] = -0.5 * w_ridge[:32]
    # head.weight[1] =  0.5 * w_ridge[:32]
    # head.bias[0] = -0.5 * w_ridge[32]
    # head.bias[1] =  0.5 * w_ridge[32]
    # This gives logit1 - logit0 = w_ridge[:32] @ pooled + w_ridge[32]
    head_weight = np.zeros((2, 32), dtype=np.float32)
    head_bias = np.zeros(2, dtype=np.float32)
    head_weight[0] = -0.5 * w_ridge[:32]
    head_weight[1] =  0.5 * w_ridge[:32]
    head_bias[0] = -0.5 * w_ridge[32]
    head_bias[1] =  0.5 * w_ridge[32]
    w["head.weight"] = head_weight
    w["head.bias"] = head_bias

    # Diagnostics
    # Compute training accuracy
    logits = pooled @ head_weight.T + head_bias
    preds = np.argmax(logits, axis=1)
    acc = (preds == y).mean()
    diagnostics = {
        "train_accuracy": float(acc),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "head_norm": float(np.linalg.norm(head_weight)),
        "head_bias": float(head_bias[1]),
        "pooled_std": float(pooled.std()),
    }
    return {"weights": w, "diagnostics": diagnostics}
