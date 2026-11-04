import numpy as np

def edit(w, f):
    # Extract feedback
    x = f["x"]          # [4000, 64] int64 token ids
    y = f["y"]          # [4000] int64 labels
    N, L = x.shape
    V = w["embed.weight"].shape[0]  # 2000

    # Build bag-of-words counts (binary presence per token)
    # We use binary presence to match the mean pooling over non-pad tokens.
    mask = (x != 0)     # [N, L] non-pad mask
    # One-hot bag-of-words: [N, V]
    X = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        tokens = x[i][mask[i]]
        if tokens.size > 0:
            X[i, tokens] = 1.0

    # Add a bias column (constant 1) to model length effects
    X_aug = np.concatenate([X, np.ones((N, 1), dtype=np.float32)], axis=1)  # [N, V+1]

    # Ridge regression closed-form: w = (X^T X + lambda I)^{-1} X^T y
    # Use y in {0,1} and center it for better conditioning.
    y_cent = y.astype(np.float32) - 0.5
    lam = 1.0
    A = X_aug.T @ X_aug + lam * np.eye(V + 1, dtype=np.float32)
    b = X_aug.T @ y_cent
    w_ridge = np.linalg.solve(A, b)  # [V+1]

    # Split into token weights and bias
    token_weights = w_ridge[:V]      # [V]
    bias_weight = w_ridge[V]

    # Normalize token weights to have unit norm for numerical stability
    norm = np.linalg.norm(token_weights)
    if norm > 1e-8:
        token_weights = token_weights / norm
        bias_weight = bias_weight / norm

    # Set embedding: first column = token weights, second column = 1 for non-pad, 0 for pad
    embed = np.zeros_like(w["embed.weight"], dtype=np.float32)  # [V, 32]
    embed[:, 0] = token_weights
    # For pad token (index 0), we want zero embedding so it doesn't contribute.
    # For all other tokens, second column = 1.
    embed[1:, 1] = 1.0
    # Pad token remains zero in both columns.

    w["embed.weight"] = embed

    # Zero out position embeddings
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)

    # Zero out all transformer blocks (identity)
    for block in ["block0", "block1"]:
        for sub in ["norm1", "qkv", "proj", "norm2", "ff1", "ff2"]:
            for param in ["weight", "bias"]:
                key = f"{block}.{sub}.{param}"
                if key in w:
                    w[key] = np.zeros_like(w[key], dtype=np.float32)
        # Set norm weights to 1 and biases to 0 (identity)
        w[f"{block}.norm1.weight"] = np.ones_like(w[f"{block}.norm1.weight"], dtype=np.float32)
        w[f"{block}.norm1.bias"] = np.zeros_like(w[f"{block}.norm1.bias"], dtype=np.float32)
        w[f"{block}.norm2.weight"] = np.ones_like(w[f"{block}.norm2.weight"], dtype=np.float32)
        w[f"{block}.norm2.bias"] = np.zeros_like(w[f"{block}.norm2.bias"], dtype=np.float32)

    # Final LayerNorm: set to identity (weight=1, bias=0)
    w["norm.weight"] = np.ones_like(w["norm.weight"], dtype=np.float32)
    w["norm.bias"] = np.zeros_like(w["norm.bias"], dtype=np.float32)

    # Head: we need to compute a linear combination of the pooled features.
    # After LayerNorm, the pooled vector will be a function of the two embedding columns.
    # Let's derive the exact transformation.
    # For a non-pad token, embedding = [a, 1, 0, ..., 0] where a = token_weights[token].
    # LayerNorm normalizes across 32 dims: mean = (a + 1)/32, var = (a - mean)^2 + (1 - mean)^2 + 30*(0 - mean)^2.
    # This is nonlinear in a. However, we can approximate by noting that a is small (normalized to unit norm over 2000 tokens, so typical a ~ 0.02).
    # Actually, we can just set the head to a linear combination that recovers a and the constant.
    # Since LayerNorm is nonlinear, we can't perfectly recover a and 1. But we can set the head to use the first two dimensions.
    # Better: we can avoid LayerNorm nonlinearity by setting the embedding to already be normalized? No, LayerNorm is applied.
    # To simplify, we can set the embedding for each token to be a one-hot vector of dimension 32? No, we need to encode token weights.
    # Alternative: we can set the head to a linear combination of the first two dimensions after LayerNorm.
    # Let's compute the expected values. For a token with weight a, the normalized vector z has:
    # z0 = (a - mean)/std, z1 = (1 - mean)/std, z2..z31 = -mean/std.
    # The mean over tokens of z0 is proportional to average a (since mean and std vary per token, but if a is small, mean ~ 1/32, std ~ sqrt(1/32 - 1/1024) ~ 0.17).
    # Actually, we can just set the head to a linear combination that works well. We can fit the head on the actual features produced by this embedding!
    # But we are not allowed to run the forward pass? We can compute the features for the feedback x using the new weights? We have the full x and we can compute the forward pass manually? The problem says we can use NumPy statistics, linear algebra, etc. We can compute the features for the feedback pool using the new weights, as long as we don't use autograd or iterative optimization. We can do a forward pass manually with NumPy. That is allowed. So we can set the embedding and blocks, then compute the new features for the 4000 examples, and then fit the head using closed-form least squares on those new features.
    # This is the best approach: set the embedding to the ridge weights, zero blocks, identity norm, then compute the new pooled features for all 4000 examples, and fit the head via ridge regression on those features.

    # Let's implement the forward pass for the new weights (only embed + norm + head matter, blocks are zero).
    # We'll compute the pooled features for all 4000 examples.
    # Since blocks are zero, h = embed(x) + pos(arange(64)). pos is zero.
    # Then h = norm(h) (LayerNorm over last dim, weight=1, bias=0).
    # Then pooled = masked_mean(h, x != 0).
    # Then logits = pooled @ head.weight.T + head.bias.

    # We can compute the pooled features for all N examples.
    # Embedding lookup: [N, L, 32]
    embed_matrix = w["embed.weight"]  # [V, 32]
    # For each example, gather embeddings for its tokens.
    # We can do this efficiently with numpy indexing.
    # x is [N, L] int64. We can use x as indices.
    # But we need to handle padding: for pad token (0), embedding is zero (since we set embed[0]=0).
    # So we can just gather embeddings for all tokens, including pad, and they will be zero.
    h = embed_matrix[x]  # [N, L, 32]
    # Add position embeddings (zero)
    # h = h + pos[None, :, :]  # pos is zero, skip.

    # LayerNorm over last dim (axis=-1) with weight=1, bias=0, eps=1e-5
    mean = h.mean(axis=-1, keepdims=True)
    var = h.var(axis=-1, keepdims=True)
    h_norm = (h - mean) / np.sqrt(var + 1e-5)

    # Masked mean pooling over non-pad tokens
    mask = (x != 0)  # [N, L]
    # Expand mask to [N, L, 1]
    mask_exp = mask[..., None].astype(np.float32)
    # Sum over L, divide by count
    sum_h = (h_norm * mask_exp).sum(axis=1)  # [N, 32]
    count = mask_exp.sum(axis=1)  # [N, 1]
    # Avoid division by zero (though all examples have at least one token)
    count = np.maximum(count, 1.0)
    pooled = sum_h / count  # [N, 32]

    # Now fit the head on these pooled features to predict y.
    # We use ridge regression on the 32-dim features.
    # Add bias term to the head by augmenting features with a column of ones.
    pooled_aug = np.concatenate([pooled, np.ones((N, 1), dtype=np.float32)], axis=1)  # [N, 33]
    # Ridge regression: solve for head weights [2, 33]? Actually we need two outputs (logits for class 0 and 1).
    # We can fit a linear classifier. Since it's binary, we can just fit a single score and then set logits = [0, score] or similar.
    # But the network has 2 output logits. We can set head.weight to have two rows: row0 = 0, row1 = fitted weights.
    # And head.bias = [0, bias].
    # To fit, we can use least squares on the centered labels: y_cent = y - 0.5.
    # We want score = pooled_aug @ w_score. Then logits = [0, score].
    # Then softmax gives P(1) = sigmoid(score). So we can fit w_score to predict y_cent.
    # Use ridge regression with small lambda.
    lam_head = 1e-3
    A_head = pooled_aug.T @ pooled_aug + lam_head * np.eye(33, dtype=np.float32)
    b_head = pooled_aug.T @ (y.astype(np.float32) - 0.5)
    w_score = np.linalg.solve(A_head, b_head)  # [33]

    # Set head weights: row0 all zeros, row1 = w_score[:32], bias = [0, w_score[32]]
    head_weight = np.zeros_like(w["head.weight"], dtype=np.float32)  # [2, 32]
    head_weight[1, :] = w_score[:32]
    w["head.weight"] = head_weight

    head_bias = np.zeros_like(w["head.bias"], dtype=np.float32)  # [2]
    head_bias[1] = w_score[32]
    w["head.bias"] = head_bias

    # Diagnostics
    # Compute training accuracy with the new head
    logits = pooled_aug @ w_score
    preds = (logits > 0).astype(np.int64)
    acc = (preds == y).mean()

    diagnostics = {
        "train_accuracy": float(acc),
        "ridge_lambda_embed": lam,
        "ridge_lambda_head": lam_head,
        "token_weights_norm": float(np.linalg.norm(token_weights)),
        "head_weight_norm": float(np.linalg.norm(w_score)),
        "head_bias": w_score[32].item(),
        "pooled_mean_0": pooled[y == 0].mean(axis=0).tolist(),
        "pooled_mean_1": pooled[y == 1].mean(axis=0).tolist(),
    }

    return {"weights": w, "diagnostics": diagnostics}
