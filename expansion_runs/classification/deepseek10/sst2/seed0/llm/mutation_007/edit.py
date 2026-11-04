import numpy as np

def edit(w, f):
    # Extract feedback arrays
    x = f["x"]          # [4000, 64] int64
    y = f["y"]          # [4000] int64
    N, L = x.shape
    V = w["embed.weight"].shape[0]  # 2000

    # Build binary bag-of-words matrix (presence of token in document)
    # We ignore padding token 0 for feature construction, but keep it in embeddings.
    B = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        tokens = x[i]
        # unique tokens, exclude padding 0
        uniq = np.unique(tokens[tokens != 0])
        B[i, uniq] = 1.0

    # Compute class means and pooled covariance for LDA
    # Use ridge regularization for numerical stability.
    y = y.astype(np.int64)
    pos_mask = (y == 1)
    neg_mask = (y == 0)
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()

    mu_pos = B[pos_mask].mean(axis=0)
    mu_neg = B[neg_mask].mean(axis=0)
    mu_all = B.mean(axis=0)

    # Between-class scatter (for two classes, rank 1)
    diff = (mu_pos - mu_neg).reshape(-1, 1)
    Sb = diff @ diff.T

    # Within-class scatter
    B_pos = B[pos_mask] - mu_pos
    B_neg = B[neg_mask] - mu_neg
    Sw = (B_pos.T @ B_pos + B_neg.T @ B_neg) / N

    # Regularized LDA: solve for top 32 discriminative directions.
    # We want directions w such that Sw w = Sb w (generalized eigenproblem).
    # Since Sb is rank 1, the first direction is Sw^{-1} diff.
    # For more directions, we can use PCA on the class-mean-centered data
    # or simply use the top eigenvectors of the regularized Sw^{-1} Sb.
    # Here we use a simple approach: compute the top 32 principal components
    # of the class-mean-centered BOW data, then rotate them by LDA.
    # Actually, to keep it simple and robust, we use the top 32 principal
    # components of the full BOW matrix, then fit a linear classifier on them.
    # This gives a 32-dimensional linear subspace that captures most variance.
    # We then embed these PCs into the token embeddings.

    # Center the BOW matrix
    B_centered = B - mu_all
    # Compute covariance (regularized)
    reg = 1e-3
    Cov = (B_centered.T @ B_centered) / N + reg * np.eye(V, dtype=np.float32)
    # Compute top 32 eigenvectors of Cov
    # Use np.linalg.eigh for symmetric matrix
    eigvals, eigvecs = np.linalg.eigh(Cov)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    # Take top 32 components
    U = eigvecs[:, :32]  # [V, 32]

    # Now U is a 2000x32 matrix. Row v is the 32-dim embedding for token v.
    # We want the document feature to be the mean of token embeddings.
    # So for a document, feature = (1/L) * sum_{v in doc} U[v].
    # This is exactly the projection of the document's BOW onto U, divided by L.
    # We can compute these features for all training examples.
    # Note: we must exclude padding tokens from the mean. The architecture
    # uses masked mean over non-padding tokens. So we compute:
    features = np.zeros((N, 32), dtype=np.float32)
    for i in range(N):
        tokens = x[i]
        mask = tokens != 0
        if mask.sum() == 0:
            continue
        # sum of U rows for tokens in doc
        feats = U[tokens[mask]].sum(axis=0)
        features[i] = feats / mask.sum()

    # Now fit a linear classifier on these features.
    # We use ridge regression with a small regularization.
    # Add bias column
    X = np.concatenate([features, np.ones((N, 1), dtype=np.float32)], axis=1)
    # Target: +1 for positive, -1 for negative
    t = np.where(y == 1, 1.0, -1.0).astype(np.float32)
    # Ridge solution: (X^T X + lambda I)^{-1} X^T t
    lam = 1.0
    A = X.T @ X + lam * np.eye(33, dtype=np.float32)
    b = X.T @ t
    coef = np.linalg.solve(A, b)
    head_w = coef[:32]  # [32]
    head_b = coef[32]   # scalar

    # Now we need to set the network weights so that the forward pass computes
    # exactly this classifier.
    # 1. Set embed.weight: for each token v, we want the LayerNorm output
    #    to be U[v] (or a scaled version). But LayerNorm normalizes each token
    #    vector to zero mean and unit variance. So we cannot directly set
    #    embed row to U[v] because LayerNorm will change it.
    #    Instead, we use the trick: set embed row = U[v] * scale + constant_vector
    #    such that after LayerNorm, the output is approximately U[v] * scale.
    #    However, the head we fitted expects features = mean of U[v] (not normalized).
    #    So we need the LayerNorm output to be U[v] exactly. But LayerNorm forces
    #    zero mean and unit variance per token. U[v] does not have that property.
    #    So we cannot make the LayerNorm output equal to U[v] for arbitrary U[v].
    #    We must design U[v] such that it already has zero mean and unit variance.
    #    Alternatively, we can bypass LayerNorm by setting its weight to something
    #    that undoes the normalization? No, LayerNorm is fixed.
    #    The only way to get a linear function of U[v] through LayerNorm is to
    #    make the input to LayerNorm have a large constant component so that
    #    normalization is approximately linear. But that only works for small
    #    variations. Here U[v] can be large.
    #    So we need a different approach: we can set the embed weights to be
    #    the U matrix, and then set the final norm to identity, and set the
    #    head to read the pooled features. But the pooled features will be
    #    mean of LayerNorm(U[v]). This is not a linear function of U[v].
    #    To avoid this, we can set the LayerNorm parameters to make it identity?
    #    LayerNorm with weight=1, bias=0 is not identity; it normalizes.
    #    But we can set the LayerNorm weight to a large constant and bias to
    #    something? Actually, if we set the input to LayerNorm to be U[v] + C,
    #    where C is a large constant vector, then LayerNorm will subtract the
    #    mean and divide by std. If C is large, the mean and std are dominated
    #    by C, so the output is approximately (U[v] - mean(U[v])) / std(C).
    #    This is a linear function of U[v]. So we can achieve a linear mapping
    #    by adding a large constant vector to each token embedding.
    #    Let's do that: we want the final pooled feature to be a linear
    #    function of the document's token counts. We can set embed row for
    #    token v to be U[v] + C, where C is a fixed large vector (e.g., all ones
    #    times 100). Then LayerNorm will output approximately (U[v] - mean(U[v])) / std(C).
    #    The mean over tokens will be approximately mean(U[v]) / std(C).
    #    So the pooled feature is a scaled version of the mean of U[v].
    #    Then the head can be scaled accordingly.
    #    This works as long as C is large enough that the variance of U[v] is
    #    negligible compared to C. But U[v] can be up to ~0.1, so C=100 is fine.
    #    Let's implement this.

    # We will set embed.weight = U + C, where C is a constant vector.
    # But we need to be careful: the architecture adds positional embedding.
    # We set pos.weight to zero.
    # Then block0 and block1 are identity (all zero weights).
    # Then final norm is identity (weight=1, bias=0) but it will normalize.
    # Wait, the final norm is also LayerNorm! The architecture has a final norm
    # before pooling. So the pooled feature is mean of final_norm(block_output).
    # If blocks are identity, block_output = embed + pos = embed.
    # So final_norm(embed) is applied. So we have TWO LayerNorms: one inside
    # block0.norm1, but block0 is identity so it doesn't matter? Actually,
    # block0.norm1 is applied to h, then qkv etc. But since qkv, proj, ff are zero,
    # the block output is h. So the norm1 inside block0 does not affect the
    # residual h. The final norm is applied to h. So the only LayerNorm that
    # affects the pooled feature is the final norm (layer.norm).
    # So we only need to worry about the final norm.
    # Let's check: forward: h = embed + pos.
    # block0: z = norm1(h); qkv=0; attn=0; h = h + proj(attn)=h; h = h + ff2(relu(ff1(norm2(h))))=h.
    # block1: same, h = h.
    # h = norm(h)  <-- final norm
    # pooled = mean(h, mask)
    # logits = head(pooled)
    # So the final norm is the only normalization applied to the pooled features.
    # So we can set embed.weight = U + C, where C is a large constant vector.
    # Then final norm will normalize each token vector. Since C is large,
    # the normalization will be approximately linear in U.
    # Let's derive: let e = U[v] + C. mean(e) = mean(U[v]) + mean(C).
    # var(e) = var(U[v]) + var(C) + 2 cov(U[v], C). If C is constant vector,
    # var(C)=0. So var(e) = var(U[v]).
    # Wait, if C is a constant vector (same for all tokens), then mean(C) is
    # just the mean of C. But C is a vector, not a scalar. If we set C to be
    # a vector with all elements equal to c, then mean(C)=c, var(C)=0.
    # Then e = U[v] + c * ones. mean(e) = mean(U[v]) + c.
    # var(e) = var(U[v]). So the normalization is (U[v] + c - mean(U[v]) - c) / sqrt(var(U[v])+eps)
    # = (U[v] - mean(U[v])) / sqrt(var(U[v])+eps).
    # This is exactly the same as normalizing U[v] alone! The constant c cancels out.
    # So adding a constant vector does not help. We need C to have variance
    # to dominate the denominator. So we should set C to be a vector with
    # large variance, e.g., alternating +c, -c. Then mean(C)=0, var(C)=c^2.
    # Then e = U[v] + C. mean(e) = mean(U[v]) + 0 = mean(U[v]).
    # var(e) = var(U[v]) + c^2. If c is large, var(e) ≈ c^2.
    # Then normalized output = (U[v] + C - mean(U[v])) / c.
    # This is (U[v] - mean(U[v]))/c + C/c.
    # The mean over tokens of this will be mean(U[v])/c + mean(C)/c = mean(U[v])/c.
    # So the pooled feature is a scaled version of mean(U[v]).
    # This works! So we can set C to be a large alternating vector.
    # Let's choose C such that var(C) is large, e.g., c = 100.
    # But we must ensure that the final norm's weight and bias are set to
    # identity (weight=1, bias=0). Then the output is as above.
    # Then the head can be scaled to read the pooled feature.
    # The pooled feature will be mean(U[v]) / c + mean(C)/c. But mean(C)=0,
    # so it's mean(U[v]) / c. So we can just set head weights = head_w * c,
    # and head bias = head_b.
    # Let's do this.

    # Choose C: a vector of length 32 with alternating +c, -c.
    c = 100.0
    C = np.zeros(32, dtype=np.float32)
    C[0::2] = c
    C[1::2] = -c
    # Actually, we need C to have zero mean to avoid shifting the mean.
    # The alternating pattern has zero mean if 32 is even. 32 is even, so mean=0.
    # var(C) = c^2.
    # So we set embed.weight = U + C (broadcasted).
    # But wait: U is [V, 32]. C is [32]. So embed.weight = U + C.
    # However, we also need to handle padding token 0. For padding, we want
    # it to contribute zero to the mean. The architecture excludes padding
    # from the mean pooling. So we don't care about its embedding.
    # But we should set it to something that doesn't break LayerNorm.
    # We can just set it to C (so U=0 for pad). Then its normalized output
    # will be C/c = alternating +1, -1. But since it's excluded from pooling,
    # it doesn't matter.

    # Now, we need to set the final norm weight and bias to identity.
    # The final norm is layer.norm. We set weight=1, bias=0.
    # But wait: the final norm has epsilon=1e-5. The normalization is
    # (x - mean) / sqrt(var + eps). With c=100, var is ~10000, so eps is negligible.
    # So the output is approximately (U[v] + C - mean(U[v])) / c.
    # The mean over tokens (excluding pad) is mean(U[v])/c + mean(C)/c = mean(U[v])/c.
    # So the pooled feature is mean(U[v])/c.
    # We want the head to compute head_w @ mean(U[v]) + head_b.
    # So we set head.weight = head_w * c, and head.bias = head_b.
    # But head.weight is [2, 32]. We need to set it for both classes.
    # Our classifier gives a score s = head_w @ features + head_b.
    # We want logits = [s, -s] or [s, 0]? Actually we want binary classification.
    # We can set head.weight[0] = head_w * c, head.weight[1] = -head_w * c,
    # head.bias[0] = head_b, head.bias[1] = -head_b.
    # Then logits = [s, -s], argmax gives positive if s>0.
    # This is fine.

    # Let's implement this.

    # 1. Set embed.weight
    embed = np.zeros((V, 32), dtype=np.float32)
    embed[:] = U + C  # U is [V,32], C is [32]
    # For padding token 0, set to C (U=0)
    embed[0] = C
    w["embed.weight"] = embed.astype(np.float32)

    # 2. Set pos.weight to zero
    w["pos.weight"] = np.zeros((64, 32), dtype=np.float32)

    # 3. Set all block weights to zero (identity)
    for block in ["block0", "block1"]:
        for name in ["norm1.weight", "norm1.bias", "qkv.weight", "qkv.bias",
                     "proj.weight", "proj.bias", "norm2.weight", "norm2.bias",
                     "ff1.weight", "ff1.bias", "ff2.weight", "ff2.bias"]:
            key = f"{block}.{name}"
            if key in w:
                w[key] = np.zeros_like(w[key])
        # Set norm weights to 1, biases to 0? Actually if we zero them,
        # norm1.weight=0 would make norm1 output zero, but since qkv is zero,
        # it doesn't matter. But to be safe, set norm weights to 1, biases to 0.
        w[f"{block}.norm1.weight"] = np.ones(32, dtype=np.float32)
        w[f"{block}.norm1.bias"] = np.zeros(32, dtype=np.float32)
        w[f"{block}.norm2.weight"] = np.ones(32, dtype=np.float32)
        w[f"{block}.norm2.bias"] = np.zeros(32, dtype=np.float32)

    # 4. Set final norm to identity
    w["norm.weight"] = np.ones(32, dtype=np.float32)
    w["norm.bias"] = np.zeros(32, dtype=np.float32)

    # 5. Set head
    head_w_scaled = head_w * c
    head_b_scaled = head_b
    w["head.weight"] = np.zeros((2, 32), dtype=np.float32)
    w["head.weight"][0] = head_w_scaled
    w["head.weight"][1] = -head_w_scaled
    w["head.bias"] = np.array([head_b_scaled, -head_b_scaled], dtype=np.float32)

    # Diagnostics
    diag = {
        "lda_directions": 32,
        "ridge_lambda": lam,
        "c_scale": c,
        "head_w_norm": float(np.linalg.norm(head_w_scaled)),
        "head_b": float(head_b_scaled),
        "embed_norm_mean": float(np.linalg.norm(embed, axis=1).mean()),
    }
    return {"weights": w, "diagnostics": diag}
