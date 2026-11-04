import numpy as np
import math

def edit(w, f):
    # Extract data
    X = f["x"]          # [4000, 64] int64
    y = f["y"]          # [4000] int64
    N, L = X.shape
    V = 2000            # vocabulary size from w["embed.weight"].shape[0]
    D = 32              # model dimension

    # Build bag-of-words counts (excluding padding token 0)
    # We'll use a sparse representation for efficiency, but 4000x2000 is small enough.
    BOW = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        for t in range(L):
            tok = X[i, t]
            if tok != 0:
                BOW[i, tok] += 1.0
    # Normalize by sequence length (number of non-padding tokens)
    lengths = np.sum(X != 0, axis=1).astype(np.float32)
    lengths[lengths == 0] = 1.0
    BOW = BOW / lengths[:, None]

    # Random projection of BOW to 32 dims for token embeddings.
    # We want each token id to have a 32-dim embedding such that the average
    # of embeddings over a sequence approximates a random projection of BOW.
    # If we set embed.weight[tok] = P[tok], then average of embeddings = (1/L) sum P[tok] = P @ BOW.
    # So we need a random matrix P of shape [V, D].
    rng = np.random.RandomState(42)
    P = rng.randn(V, D).astype(np.float32) * 0.1

    # Set embedding weights: embed.weight is [V, D]
    w["embed.weight"] = P.copy()
    # Zero position embeddings (or keep small random, but zero is fine)
    w["pos.weight"] = np.zeros_like(w["pos.weight"])

    # We will use the first block's MLP as a random ReLU feature extractor.
    # The input to the MLP is the normalized embedding (after norm1, but we can set norm1 to identity).
    # Actually, the block applies norm1, then qkv (we'll zero attention), then proj (zero), then norm2, then ff1, ff2.
    # We want the final feature to be a nonlinear function of the pooled embedding.
    # But the pooling happens after the blocks. So the blocks process each token independently if we zero attention.
    # Then mean pooling averages the per-token features.
    # To get a nonlinear function of the pooled embedding, we can't easily do it per-token because pooling is linear.
    # However, we can set the blocks to identity and let the final feature be the pooled embedding.
    # Then the head is a linear classifier on the 32-dim pooled embedding.
    # That's just a linear classifier on a random projection of BOW.
    # To get nonlinearity, we can use the fact that the final norm is applied per token, but pooling is mean.
    # If we set the MLP to a nonlinear function, the pooled feature is the average of nonlinear functions of each token.
    # That is a nonlinear function of the BOW (since it's a sum of nonlinear functions of individual tokens).
    # So we can set the first block's MLP to a random ReLU network, and zero the second block.
    # Then the final feature is the average over tokens of ReLU(W1 * e_tok + b1) projected by W2.
    # This is a nonlinear function of the BOW.
    # Let's do that.

    # Zero out attention in block0 and block1
    for blk in ["block0", "block1"]:
        w[f"{blk}.qkv.weight"] = np.zeros_like(w[f"{blk}.qkv.weight"])
        w[f"{blk}.qkv.bias"] = np.zeros_like(w[f"{blk}.qkv.bias"])
        w[f"{blk}.proj.weight"] = np.zeros_like(w[f"{blk}.proj.weight"])
        w[f"{blk}.proj.bias"] = np.zeros_like(w[f"{blk}.proj.bias"])
        # Set norm layers to identity (weight=1, bias=0)
        w[f"{blk}.norm1.weight"] = np.ones_like(w[f"{blk}.norm1.weight"])
        w[f"{blk}.norm1.bias"] = np.zeros_like(w[f"{blk}.norm1.bias"])
        w[f"{blk}.norm2.weight"] = np.ones_like(w[f"{blk}.norm2.weight"])
        w[f"{blk}.norm2.bias"] = np.zeros_like(w[f"{blk}.norm2.bias"])

    # For block0, set ff1 and ff2 to random ReLU network
    # ff1: [64, 32], ff1.bias: [64]
    # ff2: [32, 64], ff2.bias: [32]
    # We want the output of block0 to be a 32-dim feature per token.
    # The block adds the residual: h = h + ff2(relu(ff1(norm2(h)))).
    # If we set norm2 to identity, then h_out = h + ff2(relu(ff1(h))).
    # We can set ff1 and ff2 to random small values, and then the output is a nonlinear function of h.
    # But we also have the residual connection. To make it purely a feature extractor, we can set ff2 to zero? No, we want nonlinearity.
    # Let's set ff1 to random, ff2 to random, and then the output is h + nonlinear(h).
    # Then block1 we can set to identity (zero ff1, ff2) so it just passes through.
    # Actually, we can use both blocks. Let's use block0 for nonlinearity, block1 identity.

    # Random weights for block0 MLP
    rng = np.random.RandomState(123)
    w["block0.ff1.weight"] = (rng.randn(64, 32) * 0.5).astype(np.float32)
    w["block0.ff1.bias"] = (rng.randn(64) * 0.1).astype(np.float32)
    w["block0.ff2.weight"] = (rng.randn(32, 64) * 0.5).astype(np.float32)
    w["block0.ff2.bias"] = (rng.randn(32) * 0.1).astype(np.float32)

    # Block1: set to identity (zero MLP)
    w["block1.ff1.weight"] = np.zeros_like(w["block1.ff1.weight"])
    w["block1.ff1.bias"] = np.zeros_like(w["block1.ff1.bias"])
    w["block1.ff2.weight"] = np.zeros_like(w["block1.ff2.weight"])
    w["block1.ff2.bias"] = np.zeros_like(w["block1.ff2.bias"])

    # Final norm: set to identity (weight=1, bias=0)
    w["norm.weight"] = np.ones_like(w["norm.weight"])
    w["norm.bias"] = np.zeros_like(w["norm.bias"])

    # Now we need to compute the features for all 4000 examples using these weights.
    # We can implement the forward pass up to the final pooled feature.
    # Since we zeroed attention, the blocks process each token independently.
    # So we can compute per-token features and then average over non-padding tokens.

    # Embedding: for each token id, embedding = P[tok] + pos[t] (pos is zero)
    # So per-token embedding e = P[tok].
    # Then block0: norm1 is identity, so h = e.
    # qkv and proj are zero, so attention output is zero, h remains e.
    # norm2 is identity, so h = e.
    # ff1: h1 = relu(ff1 @ e + ff1.bias)  [64]
    # ff2: h2 = ff2 @ h1 + ff2.bias       [32]
    # block0 output: h_out = e + h2        [32]
    # block1: norm1 identity, qkv/proj zero, norm2 identity, ff1/ff2 zero -> h_out = h_out
    # final norm: identity -> h_out
    # pooled: mean over non-padding tokens of h_out.
    # So pooled feature = mean_t ( e_t + ff2 @ relu(ff1 @ e_t + b1) + b2 )
    # = mean_t e_t + ff2 @ mean_t relu(ff1 @ e_t + b1) + b2
    # Since mean_t e_t = P @ BOW (because BOW is normalized counts, and mean over tokens is sum/L = BOW)
    # So pooled = P @ BOW + ff2 @ mean_t relu(ff1 @ e_t + b1) + b2
    # This is a nonlinear function of BOW.

    # We can compute this for all examples efficiently.
    # For each token in each example, compute e = P[tok], then h1 = relu(ff1 @ e + b1), then h2 = ff2 @ h1.
    # Then average h2 over tokens, add P @ BOW, add b2.
    # Let's compute.

    # Precompute per-token transformations
    # For each token id v, we have e_v = P[v] (32,)
    # h1_v = relu(ff1 @ e_v + b1)  (64,)
    # h2_v = ff2 @ h1_v + b2       (32,)
    # Then for a sequence, the average of h2 over tokens is sum_{t} h2_{tok_t} / L.
    # We can compute this by accumulating h2 for each token in the sequence.
    # Since V=2000, we can precompute h2 for all tokens: H2 = ff2 @ relu(ff1 @ P.T + b1) + b2  -> shape [V, 32]
    # Then for each example, the average h2 is (1/L) sum_{t} H2[tok_t].
    # This is exactly H2.T @ BOW (since BOW is counts/L).
    # So pooled = P @ BOW + H2.T @ BOW + b2 = (P + H2.T) @ BOW + b2.
    # Wait, H2 is [V, 32]. H2.T is [32, V]. So H2.T @ BOW (where BOW is [V, N]) gives [32, N].
    # So pooled features = (P + H2.T) @ BOW + b2[:, None]  -> shape [32, N].
    # This is a linear function of BOW! Because H2 is a fixed nonlinear function of P, but P is fixed.
    # So the pooled feature is just a linear projection of BOW. The nonlinearity in the MLP is absorbed into the projection matrix.
    # Because we average over tokens, the nonlinearity per token becomes a linear combination of BOW counts.
    # So the overall model is still a linear classifier on BOW! The MLP doesn't add nonlinearity in BOW space because of the mean pooling.
    # To get nonlinearity in BOW space, we need the MLP to process the pooled representation, not per-token.
    # But the architecture applies MLP per token. So mean pooling makes it linear in BOW.
    # Therefore, any per-token nonlinearity is averaged out to a linear function of BOW.
    # The only way to get nonlinearity in BOW space is to use attention to mix tokens before the MLP, or to use the fact that the MLP is applied after attention.
    # If we use attention, the per-token features depend on the whole sequence, so the average is a nonlinear function of BOW.
    # So we should use attention to compute a sequence-level representation.

    # Let's rethink: The previous best proposal achieved 0.859 by using a linear classifier on BOW with a 32-dim bottleneck.
    # To beat that, we need a nonlinear classifier on BOW. We can use the attention mechanism to compute a weighted average of token embeddings, where the weights depend on the tokens.
    # For example, we can set the query and key to be functions of the token embeddings, so the attention weights are based on token similarity.
    # Then the output is a weighted average of values. This can capture interactions.
    # But we can also use the MLP after attention. The attention mixes tokens, so the per-token features after attention depend on the whole sequence.
    # Then the MLP processes these mixed features. Then mean pooling averages them. This can be a nonlinear function of BOW.
    # So we should not zero the attention. Instead, we can set the attention to a fixed random attention, and then use the MLP.
    # But we need to fit the head. We can compute the features for all 4000 examples using a forward pass with our chosen weights, then fit the head by ridge regression.
    # Since we can compute the forward pass ourselves, we can try different random weights and pick the best? But we can't do iterative search. We can just pick one set of random weights and fit the head.
    # However, we can also use the provided features from the current network? The current features are from the current network. We can just fit a new head on the current features. But the current features are only 32-dim and might not be very good.
    # We can also use the full BOW to train a logistic regression, and then try to approximate it with the network. But that's what previous did.

    # Given the time, let's implement a strong linear classifier on BOW using the network's 32-dim bottleneck.
    # We can compute the optimal rank-32 linear classifier on BOW by doing SVD on the BOW matrix or using LDA.
    # The previous proposal used a projection that contains the optimal linear classifier weights. They achieved 0.859.
    # We can try to improve by using a better projection: instead of just the optimal linear classifier, we can use the top 32 principal components of the BOW, and then fit a linear classifier on those.
    # But the optimal linear classifier might not lie in the top 32 PCs. We can use the fact that we have 4000 examples to compute a linear classifier w (2000-dim) using ridge regression on the full BOW.
    # Then we can project w onto the 32-dim subspace spanned by the top 32 PCs? Or we can set the embedding matrix to be the top 32 PCs, and then the head is a linear classifier on those PCs.
    # Let's do: compute the top 32 principal components of the BOW matrix (centered). Then set the embedding matrix to be the transpose of the PCs? Wait, embedding is [V, 32]. If we set embed.weight = PCs (shape [V, 32]), then the pooled feature is PCs.T @ BOW. Then we fit a linear head on these 32 features.
    # This is a rank-32 linear classifier on BOW. It might be better than the previous proposal if the previous used a different projection.
    # Let's try PCA.

    # Compute PCA on BOW
    # Center BOW
    BOW_mean = np.mean(BOW, axis=0, keepdims=True)
    BOW_centered = BOW - BOW_mean
    # Compute SVD
    U, S, Vt = np.linalg.svd(BOW_centered, full_matrices=False)
    # Vt is [min(N,V), V]. We want top 32 components: Vt[:32] shape [32, V].
    # We want embed.weight to be [V, 32] such that pooled = embed.weight.T @ BOW.
    # So embed.weight = Vt[:32].T  -> shape [V, 32].
    embed_weight = Vt[:32].T.astype(np.float32)

    # Set embedding weights
    w["embed.weight"] = embed_weight
    w["pos.weight"] = np.zeros_like(w["pos.weight"])

    # Zero all transformer blocks (identity)
    for blk in ["block0", "block1"]:
        w[f"{blk}.qkv.weight"] = np.zeros_like(w[f"{blk}.qkv.weight"])
        w[f"{blk}.qkv.bias"] = np.zeros_like(w[f"{blk}.qkv.bias"])
        w[f"{blk}.proj.weight"] = np.zeros_like(w[f"{blk}.proj.weight"])
        w[f"{blk}.proj.bias"] = np.zeros_like(w[f"{blk}.proj.bias"])
        w[f"{blk}.ff1.weight"] = np.zeros_like(w[f"{blk}.ff1.weight"])
        w[f"{blk}.ff1.bias"] = np.zeros_like(w[f"{blk}.ff1.bias"])
        w[f"{blk}.ff2.weight"] = np.zeros_like(w[f"{blk}.ff2.weight"])
        w[f"{blk}.ff2.bias"] = np.zeros_like(w[f"{blk}.ff2.bias"])
        w[f"{blk}.norm1.weight"] = np.ones_like(w[f"{blk}.norm1.weight"])
        w[f"{blk}.norm1.bias"] = np.zeros_like(w[f"{blk}.norm1.bias"])
        w[f"{blk}.norm2.weight"] = np.ones_like(w[f"{blk}.norm2.weight"])
        w[f"{blk}.norm2.bias"] = np.zeros_like(w[f"{blk}.norm2.bias"])

    w["norm.weight"] = np.ones_like(w["norm.weight"])
    w["norm.bias"] = np.zeros_like(w["norm.bias"])

    # Now the pooled feature is embed_weight.T @ BOW = Vt[:32] @ BOW (since embed_weight = Vt[:32].T).
    # Wait: pooled = mean of embeddings = (1/L) sum embed_weight[tok] = embed_weight.T @ BOW (since BOW is counts/L).
    # So pooled = embed_weight.T @ BOW = Vt[:32] @ BOW.
    # So the features are the top 32 principal components of BOW (projected onto the original space? Actually Vt[:32] @ BOW gives the coordinates of BOW in the PC space).
    # We can compute these features for all 4000 examples.
    features = BOW @ embed_weight  # [N, 32]
    # Center features? The PCA components are already centered? BOW is not centered. But the embedding is Vt[:32].T, so features = BOW @ Vt[:32].T = (BOW) @ Vt[:32].T.
    # Since Vt[:32] are the right singular vectors of centered BOW, they are orthogonal to the mean? Actually, the mean of BOW is not zero. The PCA was on centered BOW. So Vt[:32] are eigenvectors of the covariance of centered BOW. They are not necessarily orthogonal to the mean. But it's fine.

    # Now fit a linear classifier on these 32 features using ridge regression on labels (0/1).
    # We'll use a small regularization.
    X_feat = features  # [N, 32]
    # Add bias term
    X_feat_b = np.hstack([X_feat, np.ones((N, 1), dtype=np.float32)])
    # Labels: 0 and 1. We can do ridge regression to predict y.
    # Solve (X^T X + lambda I) w = X^T y
    lam = 1.0
    A = X_feat_b.T @ X_feat_b + lam * np.eye(33, dtype=np.float32)
    b = X_feat_b.T @ y.astype(np.float32)
    coef = np.linalg.solve(A, b)  # [33]
    # coef[:32] are weights for features, coef[32] is bias.
    # We want logits = head.weight @ pooled + head.bias.
    # pooled = features. So logits = head.weight @ features + head.bias.
    # We have two classes. We can set head.weight[0] = coef[:32], head.weight[1] = -coef[:32] (or something).
    # Actually, we want the difference between class 1 and class 0 to be proportional to coef.
    # Let's set head.weight[1] = coef[:32], head.weight[0] = -coef[:32].
    # Then logit1 - logit0 = 2 * (coef[:32] @ features + coef[32]).
    # We can set head.bias[1] = coef[32], head.bias[0] = -coef[32].
    # But we need to scale appropriately. The softmax will handle scaling.
    # Let's set:
    head_weight = np.zeros((2, 32), dtype=np.float32)
    head_bias = np.zeros(2, dtype=np.float32)
    head_weight[1] = coef[:32]
    head_weight[0] = -coef[:32]
    head_bias[1] = coef[32]
    head_bias[0] = -coef[32]
    w["head.weight"] = head_weight
    w["head.bias"] = head_bias

    # Diagnostics
    diagnostics = {
        "method": "PCA-32 + ridge head",
        "train_accuracy": float(np.mean((X_feat_b @ coef > 0.5).astype(int) == y)),
        "embed_norm_mean": float(np.mean(np.linalg.norm(embed_weight, axis=1))),
        "head_weight_norm": float(np.linalg.norm(head_weight)),
        "head_bias": head_bias.tolist(),
    }

    return {"weights": w, "diagnostics": diagnostics}
