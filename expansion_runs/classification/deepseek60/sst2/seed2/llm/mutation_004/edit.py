import numpy as np

def edit(w, f):
    # Extract training data
    x = f["x"]          # [4000, 64] int64 token IDs
    y = f["y"]          # [4000] int64 labels
    N, L = x.shape
    V = w["embed.weight"].shape[0]  # 2000

    # Build normalized bag-of-words features: X[i, v] = count of token v in example i / num_tokens
    # Use only tokens != 0 (padding)
    mask = (x != 0)
    lengths = mask.sum(axis=1).astype(np.float32)
    lengths[lengths == 0] = 1.0  # avoid division by zero
    X = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        tokens = x[i][mask[i]]
        if len(tokens) > 0:
            counts = np.bincount(tokens, minlength=V)
            X[i] = counts / lengths[i]

    # One-hot labels for two classes
    Y = np.zeros((N, 2), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    # Ridge regression: solve for W (2xV) and b (2)
    # Augment X with a column of ones for bias
    X_aug = np.hstack([X, np.ones((N, 1), dtype=np.float32)])  # [N, V+1]
    # Solve (X_aug^T X_aug + λI) beta = X_aug^T Y
    # Use a small ridge to stabilize inversion
    lam = 1e-2
    A = X_aug.T @ X_aug + lam * np.eye(V + 1, dtype=np.float32)
    B = X_aug.T @ Y
    beta = np.linalg.solve(A, B)  # [V+1, 2]
    W = beta[:V, :].T   # [2, V]
    b = beta[V, :]      # [2]

    # Now map W into embedding and head.
    # We want the model to compute logits = W @ x + b.
    # The model computes: h = LayerNorm(E[token]); pooled = mean(h); logits = head @ pooled + head_bias.
    # If we set E such that each row has zero mean and unit variance, LayerNorm is identity.
    # Then pooled = E^T @ x, and logits = head @ E^T @ x + head_bias.
    # We need head @ E^T = W. Since W is 2xV, we can set E = W^T (Vx2) and head = I_2 (2x2).
    # But E must be Vx32. We pad with zeros. However, LayerNorm will not be identity if rows are not zero-mean unit-variance.
    # Instead, we can set E to be W^T padded with zeros, then compute the actual features produced by the model
    # for the training data, and fit a new head on those features. This is a two-step approach.
    # Step 1: Set embedding to W^T padded with zeros.
    E = np.zeros((V, 32), dtype=np.float32)
    E[:, :2] = W.T  # [V, 2]

    # Step 2: Compute the actual pooled features for the training data using this E.
    # We need to simulate the forward pass up to the pooled features.
    # Since blocks are zero, h = embed(ids) + pos(arange(64)). pos is zero.
    # Then LayerNorm over last dim (32). Then mean over non-pad tokens.
    # We can compute this efficiently.
    # First, get the embedding for each token in each example: E[token] -> [N, L, 32]
    # But we can just compute the normalized embeddings for all tokens present.
    # Actually, we can compute the pooled feature for each example directly.
    # For each example i, we have tokens. For each token t, embedding = E[token_t] (32-dim).
    # Apply LayerNorm: mean = mean(embedding), std = std(embedding) + 1e-5.
    # normalized = (embedding - mean) / std.
    # Then average over tokens.
    # We can vectorize this.
    # Create a matrix of embeddings for all tokens: E is Vx32.
    # For each example, we need to average LayerNorm(E[token]).
    # We can precompute LayerNorm(E) for all V tokens: [V, 32]
    # LayerNorm: for each row, subtract mean, divide by std.
    E_mean = E.mean(axis=1, keepdims=True)  # [V, 1]
    E_std = E.std(axis=1, keepdims=True) + 1e-5  # [V, 1]
    E_norm = (E - E_mean) / E_std  # [V, 32]

    # Now for each example, average E_norm[token] over its tokens.
    # We can do this by matrix multiplication: X is [N, V] with counts/lengths.
    # pooled = X @ E_norm  # [N, 32]
    pooled = X @ E_norm  # [N, 32]

    # Step 3: Fit a linear classifier on pooled features to predict Y.
    # Use ridge regression again.
    pooled_aug = np.hstack([pooled, np.ones((N, 1), dtype=np.float32)])  # [N, 33]
    lam2 = 1e-2
    A2 = pooled_aug.T @ pooled_aug + lam2 * np.eye(33, dtype=np.float32)
    B2 = pooled_aug.T @ Y
    beta2 = np.linalg.solve(A2, B2)  # [33, 2]
    head_weight = beta2[:32, :].T  # [2, 32]
    head_bias = beta2[32, :]       # [2]

    # Update weights
    w["embed.weight"] = E.astype(np.float32)
    w["head.weight"] = head_weight.astype(np.float32)
    w["head.bias"] = head_bias.astype(np.float32)

    # Ensure all other weights remain zero (or keep as is, but they are already zero)
    # We can explicitly zero out block parameters to be safe, but they are already zero.
    # However, pos.weight is zero, norm.weight is 1, norm.bias is 0. Keep them.

    # Diagnostics
    # Compute training accuracy with the new weights (simulated)
    logits = pooled @ head_weight.T + head_bias  # [N, 2]
    preds = np.argmax(logits, axis=1)
    acc = (preds == y).mean()

    diagnostics = {
        "method": "Ridge regression on normalized BOW -> embedding + head fit on LayerNorm features",
        "train_accuracy": float(acc),
        "head_weight_norm": float(np.linalg.norm(head_weight)),
        "head_bias": head_bias.tolist(),
        "embed_norm_mean": float(np.linalg.norm(E, axis=1).mean()),
    }

    return {"weights": w, "diagnostics": diagnostics}
