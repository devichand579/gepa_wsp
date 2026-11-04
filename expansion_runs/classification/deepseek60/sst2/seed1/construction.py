import numpy as np

def edit(w, f):
    # Extract training data
    X = f["x"]          # [4000, 64] int64
    y = f["y"]          # [4000] int64
    N, L = X.shape
    V = 2000            # vocab size
    D = 32              # embedding dim

    # Build bag-of-words matrix (excluding padding token 0)
    # We'll use a sparse representation for efficiency, but here N=4000, V=2000 is small.
    BOW = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        for t in X[i]:
            if t != 0:
                BOW[i, t] += 1.0
    # Normalize by sentence length (number of non-padding tokens)
    lengths = np.sum(X != 0, axis=1).astype(np.float32)
    lengths[lengths == 0] = 1.0
    BOW = BOW / lengths[:, None]

    # Ridge regression to get token weights w_token (size V)
    # We solve: min ||BOW @ w_token - y_centered||^2 + lambda ||w_token||^2
    # Center y: y is 0/1, we can use y - mean(y)
    y_centered = y.astype(np.float32) - np.mean(y)
    lambda_reg = 1.0
    # Compute BOW^T BOW + lambda I
    A = BOW.T @ BOW + lambda_reg * np.eye(V, dtype=np.float32)
    b = BOW.T @ y_centered
    w_token = np.linalg.solve(A, b)  # [V]

    # Now design embedding matrix E [V, D] such that each row has mean 0 and variance 1.
    # We want the first component to be proportional to w_token, but we need to ensure
    # the row has mean 0 and variance 1. We'll set:
    # E[:, 0] = w_token / std_w  (so it has variance 1)
    # E[:, 1] = -E[:, 0] / (D-1) * (D-1)? Actually, to make mean 0, we can set
    # E[:, 0] = a, E[:, 1] = -a, and the rest 0. Then mean = 0, variance = 2a^2/D.
    # To get variance 1, we need a = sqrt(D/2). But then the magnitude of a is fixed.
    # Instead, we can use a random orthogonal basis and project w_token onto it.
    # Simpler: create a random matrix R [V, D] with mean 0, variance 1 per row.
    # Then we want E such that E @ head = w_token (approximately). We can set
    # E = w_token[:, None] * v1[None, :] + v2[None, :] where v1, v2 are fixed vectors.
    # But we need E rows to have mean 0 and variance 1. Let's do:
    # E[:, 0] = w_token / std_w
    # E[:, 1] = -w_token / std_w
    # E[:, 2:] = small random noise with mean 0 and variance 1.
    # Then mean = 0, variance = 2*(w_token/std_w)^2 / D + (D-2)/D * 1.
    # This is not exactly 1. To fix, we can normalize each row to have mean 0 and variance 1.
    # That's easy: for each row, subtract mean, divide by std.
    # But then the first component will be scaled nonlinearly. However, we can just
    # set E to be a random matrix, and then fit the head on the resulting features.
    # Actually, we can just set E to be the top 32 principal components of BOW.
    # That's a closed-form solution and likely better.
    # Let's do PCA on BOW.
    # Center BOW
    BOW_centered = BOW - np.mean(BOW, axis=0, keepdims=True)
    # Compute covariance matrix (V x V) - too big? V=2000, 2000x2000 is 4M, fine.
    # But we can use SVD on BOW_centered [N, V] with N=4000, V=2000.
    # We want the top 32 right singular vectors.
    # BOW_centered = U S Vt. The right singular vectors are Vt.T.
    # We can compute Vt via SVD of BOW_centered.T @ BOW_centered? That's V x V.
    # Better: use np.linalg.svd on BOW_centered (4000x2000) with full_matrices=False.
    # That gives U [4000, 2000], S [2000], Vt [2000, 2000]. We take first 32 columns of Vt.T.
    # This might be slow but 4000x2000 is okay.
    # Let's use a more efficient approach: compute the Gram matrix BOW_centered @ BOW_centered.T? No.
    # Actually, we can just use the token weights w_token to form the embedding.
    # Let's use a simple approach: set E[:, 0] = w_token / np.std(w_token)
    # and E[:, 1] = -E[:, 0], and E[:, 2:] = random normal.
    # Then normalize each row to have mean 0 and variance 1.
    # This will preserve some information about w_token in the direction of the row.
    # Then we compute the features and fit the head.
    # This is a heuristic but should work.

    # Let's do PCA on BOW to get 32-dimensional token embeddings.
    # We'll use the top 32 principal components of the token vectors.
    # BOW is [N, V]. We want a projection matrix P [V, D] such that BOW @ P gives [N, D].
    # We can compute P as the top D right singular vectors of BOW_centered.
    # To avoid large SVD, we can compute the covariance matrix C = BOW_centered.T @ BOW_centered [V, V].
    # Then find top D eigenvectors of C.
    # V=2000, C is 2000x2000 = 4M floats = 16MB, fine.
    # Compute C
    # C = BOW_centered.T @ BOW_centered  # [V, V]
    # Then use np.linalg.eigh to get eigenvectors.
    # This is O(V^3) = 8e9, might be slow. Let's use SVD on BOW_centered instead.
    # BOW_centered is [4000, 2000]. SVD with full_matrices=False gives U [4000, 2000], S [2000], Vt [2000, 2000].
    # This is O(N * V^2) = 4000 * 4e6 = 1.6e10, might be slow but doable in a few seconds.
    # Actually, we can just use the token weights w_token directly.
    # Let's set E[:, 0] = w_token / np.std(w_token)
    # E[:, 1] = -E[:, 0]
    # E[:, 2:] = np.random.randn(V, D-2) * 0.1
    # Then normalize each row: E = (E - mean(E, axis=1, keepdims=True)) / std(E, axis=1, keepdims=True)
    # This gives mean 0, variance 1.
    # Then we set the embedding weight to E.
    # Set pos.weight to zeros.
    # Set all transformer block weights to zero (proj, ff2) so they are identity.
    # Actually, we need to set proj.weight = 0, proj.bias = 0, ff2.weight = 0, ff2.bias = 0.
    # Also set norm1, norm2 weights to 1, biases to 0 (they are already 1 and 0).
    # Set qkv weights to 0? If qkv weights are 0, attention output is 0, so proj(0)=0. That's fine.
    # So we can just zero out proj and ff2 weights and biases.
    # Then the forward pass: h = embed(ids) + pos. pos=0.
    # block0: z = norm1(h) = layernorm(h). qkv(z) = 0. attention = 0. proj(0)=0. h = h + 0 = h.
    # norm2(h) = layernorm(h). ff1(h) = 0? If ff1 weights are 0, output is 0. relu(0)=0. ff2(0)=0. h = h + 0 = h.
    # block1: same. h remains embed(ids).
    # final norm: layernorm(h). pooled = mean over tokens. logits = head @ pooled.
    # So final features = mean(layernorm(embed(ids))).
    # Since embed(ids) has mean 0 and variance 1 per token, layernorm(embed(ids)) = embed(ids) (up to epsilon).
    # So final features = mean(embed(ids)).
    # Then we fit head on these features.

    # Let's implement this.

    # 1. Compute token weights w_token via ridge regression on BOW.
    # We already have w_token from above.

    # 2. Build embedding matrix E [V, D] with mean 0 and variance 1 per row.
    # We want E to encode w_token. Let's set:
    # E[:, 0] = w_token / np.std(w_token)
    # E[:, 1] = -E[:, 0]
    # E[:, 2:] = np.random.randn(V, D-2) * 0.1
    # Then normalize each row to mean 0, variance 1.
    np.random.seed(42)
    E = np.zeros((V, D), dtype=np.float32)
    std_w = np.std(w_token)
    if std_w == 0:
        std_w = 1.0
    E[:, 0] = w_token / std_w
    E[:, 1] = -E[:, 0]
    E[:, 2:] = np.random.randn(V, D-2).astype(np.float32) * 0.1
    # Normalize each row
    E_mean = np.mean(E, axis=1, keepdims=True)
    E_std = np.std(E, axis=1, keepdims=True)
    E_std[E_std == 0] = 1.0
    E = (E - E_mean) / E_std

    # 3. Set weights
    w["embed.weight"] = E
    w["pos.weight"] = np.zeros((64, D), dtype=np.float32)

    # Zero out transformer blocks to make them identity
    for block in ["block0", "block1"]:
        w[f"{block}.proj.weight"] = np.zeros((D, D), dtype=np.float32)
        w[f"{block}.proj.bias"] = np.zeros(D, dtype=np.float32)
        w[f"{block}.ff2.weight"] = np.zeros((D, 64), dtype=np.float32)
        w[f"{block}.ff2.bias"] = np.zeros(D, dtype=np.float32)
        # Also zero out qkv and ff1 to be safe (they won't affect if proj and ff2 are zero)
        w[f"{block}.qkv.weight"] = np.zeros((96, D), dtype=np.float32)
        w[f"{block}.qkv.bias"] = np.zeros(96, dtype=np.float32)
        w[f"{block}.ff1.weight"] = np.zeros((64, D), dtype=np.float32)
        w[f"{block}.ff1.bias"] = np.zeros(64, dtype=np.float32)
        # Norm weights are already 1, biases 0. We can leave them.

    # 4. Compute features for all training examples
    # We need to compute mean(layernorm(embed(ids))) for each example.
    # Since embed(ids) has mean 0 and variance 1, layernorm is identity (with eps=1e-5).
    # So features = mean(embed(ids)) over non-padding tokens.
    # Let's compute it directly.
    features = np.zeros((N, D), dtype=np.float32)
    for i in range(N):
        tokens = X[i]
        mask = tokens != 0
        if np.sum(mask) == 0:
            continue
        emb = E[tokens[mask]]  # [L, D]
        features[i] = np.mean(emb, axis=0)

    # 5. Fit head weights via ridge regression on features
    # We want logits = head @ features + bias. For binary classification, we can use
    # a linear model: y = features @ w_head + b. We can solve for w_head and b.
    # Add bias by augmenting features with a column of ones.
    features_aug = np.hstack([features, np.ones((N, 1), dtype=np.float32)])
    lambda_head = 1.0
    A_head = features_aug.T @ features_aug + lambda_head * np.eye(D+1, dtype=np.float32)
    b_head = features_aug.T @ y_centered
    w_head = np.linalg.solve(A_head, b_head)  # [D+1]
    head_weight = w_head[:D]  # [D]
    head_bias = w_head[D]     # scalar

    # For binary classification, we need two output logits. We can set:
    # head.weight[0] = -head_weight, head.bias[0] = -head_bias
    # head.weight[1] = head_weight, head.bias[1] = head_bias
    # This gives logit difference = 2*(head_weight @ features + head_bias)
    # So argmax will be class 1 if head_weight @ features + head_bias > 0.
    w["head.weight"] = np.vstack([-head_weight, head_weight]).astype(np.float32)
    w["head.bias"] = np.array([-head_bias, head_bias], dtype=np.float32)

    # 6. Diagnostics
    # Compute accuracy on training set
    logits = features @ head_weight + head_bias
    preds = (logits > 0).astype(np.int64)
    acc = np.mean(preds == y)
    diagnostics = {
        "train_accuracy": float(acc),
        "w_token_std": float(np.std(w_token)),
        "head_norm": float(np.linalg.norm(head_weight)),
        "head_bias": float(head_bias),
    }

    return {"weights": w, "diagnostics": diagnostics}
