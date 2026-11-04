import numpy as np

def edit(w, f):
    # Extract training data
    x = f["x"]          # [4000, 64] int64
    y = f["y"]          # [4000] int64
    N, L = x.shape
    V = w["embed.weight"].shape[0]  # 2000

    # Build bag-of-words counts (binary presence is often better for SST-2)
    # Use binary presence to avoid length bias
    bow = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        tokens = x[i]
        # ignore padding (0) and unknown (1) maybe keep unknown? We'll keep all non-pad
        valid = tokens[tokens != 0]
        if len(valid) > 0:
            bow[i, valid] = 1.0

    # Compute class means and pooled covariance for LDA
    mask0 = (y == 0)
    mask1 = (y == 1)
    n0 = mask0.sum()
    n1 = mask1.sum()
    mu0 = bow[mask0].mean(axis=0)
    mu1 = bow[mask1].mean(axis=0)
    # Pooled covariance (diagonal approximation for stability and speed)
    # Use diagonal LDA (Naive Bayes) which is robust for text
    var0 = bow[mask0].var(axis=0) + 1e-6
    var1 = bow[mask1].var(axis=0) + 1e-6
    # Weighted average variance
    var_pool = (n0 * var0 + n1 * var1) / (n0 + n1)
    # LDA direction: w = (mu1 - mu0) / var_pool
    w_lda = (mu1 - mu0) / var_pool
    # Bias: threshold at midpoint
    # Decision: w_lda @ x > c
    # c = 0.5 * (w_lda @ mu0 + w_lda @ mu1)
    c = 0.5 * (w_lda @ mu0 + w_lda @ mu1)

    # Normalize w_lda to have reasonable scale (avoid huge values)
    # We will encode this direction into the embedding.
    # We need to set embed.weight such that the pooled feature after LayerNorm
    # is proportional to w_lda. Since LayerNorm normalizes each token vector,
    # we can use a 32-dim embedding where the first dimension is w_lda[token],
    # and we set the final norm to identity by making each token vector have
    # zero mean and unit variance across the 32 dims. To achieve this, we
    # construct a 32-dim vector for each token: we set the first component to
    # w_lda[token] and the remaining 31 components to values that make the
    # vector have mean 0 and variance 1. However, LayerNorm will then normalize
    # the vector, which will remove the magnitude of w_lda[token]. To preserve
    # the magnitude, we can instead use a different trick: set the final norm
    # weight to a large value and bias to 0? No, LayerNorm still divides by std.
    # Actually, we can bypass LayerNorm by setting its weight to 0 and bias to 0?
    # That kills signal. The only way to get a linear function of w_lda is to
    # make the input to LayerNorm have a constant variance independent of w_lda.
    # We can do this by using a 2D subspace: E[t] = [w_lda[t], 1]. Then mean = (w+1)/2,
    # variance = (w-1)^2/2. LayerNorm output = (w - (w+1)/2) / sqrt((w-1)^2/2) = sign(w-1)*sqrt(2)/2.
    # This only gives sign. To get magnitude, we can use a thermometer code:
    # represent w_lda[t] in binary across 32 dimensions. Then the mean of these bits
    # is a linear function of w_lda[t] (if we use a binary representation). Since we
    # have 32 dimensions, we can quantize w_lda to 32 levels using 5 bits (for magnitude)
    # and 1 bit for sign. Then the pooled feature is the average of these bits.
    # The head can then reconstruct the weighted sum by taking a linear combination
    # of the bits. This is a bit involved but doable. However, a simpler approach:
    # we can set the final norm to have weight=1, bias=0, and set the embedding
    # to be the LDA direction itself, but then LayerNorm will normalize each token
    # vector to unit variance. If we set the embedding for token t to be a vector
    # that is a scaled version of a fixed unit vector u, then LayerNorm will output
    # sign(w_lda[t]) * u. This gives a binary classifier on the sign of w_lda.
    # For SST-2, a binary classifier on the sign of LDA weights (i.e., using only
    # the sign of the token weights) might still be quite good. But we can do better
    # by using multiple dimensions to encode the magnitude. Let's use a 32-bit
    # binary representation of w_lda[t]. We'll quantize w_lda to 32 levels.
    # Actually, we can just set the embedding to be the LDA direction in the first
    # dimension, and set the final norm weight to 0? No.
    # Let's reconsider: we can set the final norm to have weight=0 and bias=0?
    # Then output is 0. No.
    # What if we set the final norm to have weight=1, bias=0, and we set the
    # embedding for token t to be a vector that already has zero mean and unit
    # variance, and we want the pooled feature to be linear in w_lda[t].
    # If we set E[t] = w_lda[t] * u, then variance = w_lda[t]^2 * var(u). LayerNorm
    # divides by |w_lda[t]| * std(u), so output = sign(w_lda[t]) * u/std(u).
    # So magnitude is lost. To preserve magnitude, we need the variance to be
    # independent of w_lda[t]. We can achieve this by adding a constant vector v
    # such that E[t] = w_lda[t] * u + v, and we choose u and v such that
    # var(E[t]) = constant. This requires w_lda[t]^2 var(u) + 2 w_lda[t] cov(u,v) + var(v) = C.
    # This can only hold for all w_lda[t] if var(u)=0 and cov(u,v)=0. So E[t] = v,
    # independent of w_lda[t]. So impossible.
    # Therefore, we cannot get a linear function of w_lda through LayerNorm.
    # But we can bypass LayerNorm by setting its weight to 0? No.
    # Wait! We can set the final norm to be identity by making the input to it
    # already have zero mean and unit variance per token. But we just showed that
    # if E[t] depends on w_lda[t], its variance will depend on w_lda[t] unless
    # we use a nonlinear encoding. So we must use a nonlinear encoding.
    # Let's use a simple nonlinear encoding: we set E[t] to be a 32-dim vector
    # where the first 16 dims are a binary representation of w_lda[t] (positive part)
    # and the next 16 dims are a binary representation of -w_lda[t] (negative part).
    # Then the mean of these bits is a linear function of w_lda[t] (if we use
    # a linear combination of bits). Actually, if we use a binary representation,
    # the value w_lda[t] is a linear combination of the bits with powers of 2.
    # So if we set E[t] to be the bits of w_lda[t], then the pooled feature is
    # the average of these bits. Then the head can compute the weighted sum by
    # taking a linear combination of the bits with the appropriate powers of 2.
    # This works! Let's do this.
    # We have 32 dimensions. We can use 5 bits for magnitude (0-31) and 1 bit for sign.
    # Actually, we can just use 32 bits to represent w_lda[t] in a fixed-point format.
    # But w_lda[t] can be any real number. We can quantize it to 32 levels.
    # Let's quantize w_lda to 32 levels using 5 bits. We'll use 5 dimensions for
    # the magnitude bits, and 1 dimension for the sign. The remaining 26 dimensions
    # can be set to 0. Then the pooled feature will have these bits averaged.
    # The head can then compute the weighted sum by multiplying each bit by its
    # corresponding power of 2 and summing. This is a linear operation.
    # However, LayerNorm will normalize each token's vector. If we set the vector
    # to be the bits, then the mean and variance of the bits will depend on w_lda[t].
    # LayerNorm will then normalize them, which will distort the bits. To avoid this,
    # we can set the final norm to have weight=0 and bias=0? No.
    # We can set the final norm to have weight=1, bias=0, but we need the input
    # to have zero mean and unit variance per token. If we set the bits to be
    # symmetric around 0 (e.g., use -1 and +1 instead of 0 and 1), then the mean
    # might be 0. But variance will still depend on w_lda[t]. So LayerNorm will
    # still normalize. The only way to avoid LayerNorm's normalization is to
    # set the final norm weight to 0? No.
    # Wait, we can set the final norm weight to a very large value and bias to 0?
    # No, LayerNorm always divides by std. The weight multiplies the normalized
    # output. So if we set weight to a large value, it just scales the normalized
    # output. The normalized output is (x - mean)/std. If we set weight to a large
    # value, it amplifies the normalized output. But the normalized output is
    # still a nonlinear function of x. So it doesn't help.
    # The only way to bypass LayerNorm is to set its weight to 0 and bias to
    # something that carries the signal? But bias is constant per dimension.
    # If we set weight=0, output = bias. That's constant. No signal.
    # So we cannot bypass LayerNorm.
    # Therefore, we must work with the normalized vectors. The normalized vector
    # for token t is a function of the bits. If we use a binary representation,
    # the normalized vector will be some nonlinear function of w_lda[t]. But we
    # can still fit a linear head on the pooled normalized vectors. Since we have
    # the full training data, we can compute the pooled normalized vectors for
    # all 4000 examples, and then fit a linear classifier (e.g., LDA) on those
    # pooled features. This is exactly what the previous proposals did: they fit
    # a head on the existing features. But we can do better by choosing the
    # embeddings to make the pooled features more informative. We can set the
    # embeddings to be the LDA direction itself, but as we saw, LayerNorm will
    # normalize each token vector. If we set the embedding for token t to be
    # w_lda[t] * u, then the normalized vector is sign(w_lda[t]) * u_norm.
    # So the pooled feature is (1/L) sum sign(w_lda[t]) * u_norm. This is a
    # linear function of the signs. The head can then learn a linear classifier
    # on this. This is equivalent to using binary features: sign(w_lda[t]).
    # This might be quite good. Let's just do that: set embed.weight[:, 0] = w_lda,
    # and set the final norm to identity? No, LayerNorm will normalize. But if we
    # set embed.weight[:, 0] = w_lda, and all other columns 0, then each token
    # vector is [w_lda[t], 0, 0, ...]. LayerNorm will compute mean = w_lda[t]/32,
    # variance = w_lda[t]^2 * 31/32. Normalized first component = (w_lda[t] - w_lda[t]/32) / (|w_lda[t]| * sqrt(31/32)) = sign(w_lda[t]) * sqrt(31/32).
    # So the pooled feature's first component is (1/L) sum sign(w_lda[t]) * sqrt(31/32).
    # The head can then just use this first component. So we get a classifier on
    # the average sign of w_lda[t]. This is a binary bag-of-words classifier.
    # We can also set the other 31 dimensions to encode other information, e.g.,
    # the magnitude of w_lda[t] in a way that survives normalization. But as we
    # saw, any scaling is removed. So we can only get sign. However, we can use
    # multiple dimensions to encode multiple binary features. For example, we can
    # set the embedding for token t to be a 32-bit binary code that is a hash of
    # the token. Then the pooled feature is the average of these codes. Then we
    # can fit a linear classifier on this 32-dim pooled feature. This is a random
    # projection of the bag-of-words counts. But we can do better: we can set the
    # codes to be the top 32 principal components of the bag-of-words data? But
    # we don't have the full bag-of-words matrix? We do have f["x"] which is
    # [4000, 64]. We can compute the bag-of-words matrix for the 4000 training
    # examples. That's a 4000x2000 matrix. We can compute the LDA direction on
    # this matrix. But we can also compute the top 32 principal components of
    # the bag-of-words matrix. Then we can set the embedding for each token to
    # be its loading on these 32 components. Then the pooled feature (after
    # LayerNorm) will be approximately the projection of the bag-of-words counts
    # onto these 32 components. Then the head can be a linear classifier on this
    # projection. This is a standard technique: use PCA on bag-of-words, then
    # linear classifier. But we have to account for LayerNorm. If we set the
    # embedding to be the PCA loadings, then each token vector is a 32-dim vector.
    # LayerNorm will normalize each token vector. This will distort the PCA
    # projection. But we can just compute the pooled features after LayerNorm
    # for all 4000 examples, and then fit a linear classifier on those pooled
    # features. This is exactly what the previous proposals did: they fit a head
    # on the existing features. But we can change the embeddings to make the
    # pooled features more linearly separable. We can do this by setting the
    # embeddings to be the LDA direction itself, but as we saw, LayerNorm will
    # binarize it. So we can set the embeddings to be the LDA direction, and then
    # fit a linear classifier on the resulting pooled features. This is a two-step
    # process: first set embeddings, then compute pooled features, then fit head.
    # Since we can compute the forward pass ourselves for all 4000 examples, we
    # can do this. We have the full x and y. We can set the embeddings to some
    # initial guess, compute the pooled features, then fit a linear classifier
    # (LDA) on those pooled features. We can even iterate this a few times? But
    # no iterative optimizer. We can do a closed-form fit of the head given the
    # embeddings. But we can also choose the embeddings to be the LDA direction
    # on bag-of-words, and then fit the head on the resulting pooled features.
    # This is a single step: set embed = w_lda (as first column), zero other
    # weights, set blocks to identity, set norm to identity? No, norm is not
    # identity. But we can compute the pooled features after LayerNorm and then
    # fit the head. Let's do that.

    # Actually, the simplest and most effective approach: we can set the entire
    # network to be a linear classifier on bag-of-words by setting the embedding
    # to be the identity? But embed is [2000, 32], we can't have identity.
    # We can set the embedding to be a random projection, but we can do better:
    # we can set the embedding to be the LDA direction, and then fit the head
    # on the pooled features. Let's do this:
    # 1. Compute LDA direction w_lda on bag-of-words (binary presence).
    # 2. Set embed.weight[:, 0] = w_lda, other columns 0.
    # 3. Set all block weights to 0, biases to 0, and set norm1/norm2 weights to 0
    #    so blocks are identity.
    # 4. Set final norm weight=1, bias=0 (default).
    # 5. Compute the pooled features for all 4000 examples using this setup.
    #    Since blocks are identity, h = embed(ids) + pos. Set pos=0.
    #    Then h = embed(ids). Then final norm: LayerNorm(h).
    #    Then pooled = mean over non-pad tokens of LayerNorm(h).
    # 6. Fit a linear classifier (LDA) on these pooled features to get head weights.
    # This gives a classifier that uses the sign of w_lda[t] (since LayerNorm
    # binarizes). But we can also set embed.weight to be a 32-dim PCA projection
    # of the bag-of-words, then compute pooled features, then fit head. Let's
    # try the LDA direction first.

    # Compute LDA direction on bag-of-words (binary presence)
    # We already have w_lda from above.
    # Normalize w_lda to have unit norm to avoid huge values
    w_lda = w_lda / (np.linalg.norm(w_lda) + 1e-8)

    # Set embed.weight: first column = w_lda, rest 0
    embed = np.zeros_like(w["embed.weight"])
    embed[:, 0] = w_lda
    w["embed.weight"] = embed.astype(np.float32)

    # Set pos.weight to 0
    w["pos.weight"] = np.zeros_like(w["pos.weight"])

    # Zero out all block weights and biases to make blocks identity
    for block in ["block0", "block1"]:
        for param in ["qkv.weight", "qkv.bias", "proj.weight", "proj.bias",
                      "ff1.weight", "ff1.bias", "ff2.weight", "ff2.bias"]:
            key = f"{block}.{param}"
            if key in w:
                w[key] = np.zeros_like(w[key])
        # Set norm1 and norm2 weights to 0 and biases to 0 to make them output 0
        for norm in ["norm1", "norm2"]:
            w[f"{block}.{norm}.weight"] = np.zeros_like(w[f"{block}.{norm}.weight"])
            w[f"{block}.{norm}.bias"] = np.zeros_like(w[f"{block}.{norm}.bias"])

    # Final norm: set weight=1, bias=0 (default)
    w["norm.weight"] = np.ones_like(w["norm.weight"])
    w["norm.bias"] = np.zeros_like(w["norm.bias"])

    # Now compute pooled features for all 4000 examples using this setup.
    # We need to implement the forward pass up to the final norm and pooling.
    # Since blocks are identity, h = embed(ids) + pos. pos=0.
    # So h = embed(ids). Shape: [N, L, 32]
    # Then final norm: LayerNorm over last dim.
    # Then pooled = mean over non-pad tokens.
    N = x.shape[0]
    L = x.shape[1]
    # Embedding lookup
    h = embed[x]  # [N, L, 32]
    # LayerNorm
    eps = 1e-5
    mean = h.mean(axis=-1, keepdims=True)
    var = h.var(axis=-1, keepdims=True)
    h_norm = (h - mean) / np.sqrt(var + eps)
    # Pooling: mean over non-pad tokens
    mask = (x != 0).astype(np.float32)  # [N, L]
    mask_sum = mask.sum(axis=1, keepdims=True)  # [N, 1]
    mask_sum = np.maximum(mask_sum, 1.0)  # avoid div by zero
    pooled = (h_norm * mask[:, :, None]).sum(axis=1) / mask_sum  # [N, 32]

    # Now fit a linear classifier (LDA) on pooled features
    # We have pooled [4000, 32] and y [4000]
    # Compute class means
    mu0_p = pooled[mask0].mean(axis=0)
    mu1_p = pooled[mask1].mean(axis=0)
    # Pooled covariance (diagonal)
    var0_p = pooled[mask0].var(axis=0) + 1e-6
    var1_p = pooled[mask1].var(axis=0) + 1e-6
    var_pool_p = (n0 * var0_p + n1 * var1_p) / (n0 + n1)
    # LDA direction
    w_head = (mu1_p - mu0_p) / var_pool_p
    # Bias: threshold at midpoint
    c_head = 0.5 * (w_head @ mu0_p + w_head @ mu1_p)
    # We want logits: class 0 = 0, class 1 = w_head @ pooled - c_head
    # So head.weight[0] = 0, head.bias[0] = 0
    # head.weight[1] = w_head, head.bias[1] = -c_head
    head_w = np.zeros((2, 32), dtype=np.float32)
    head_b = np.zeros(2, dtype=np.float32)
    head_w[1] = w_head
    head_b[1] = -c_head
    w["head.weight"] = head_w
    w["head.bias"] = head_b

    # Diagnostics
    # Compute training accuracy with this setup
    logits = pooled @ head_w.T + head_b
    preds = np.argmax(logits, axis=1)
    acc = (preds == y).mean()

    diagnostics = {
        "train_accuracy": float(acc),
        "lda_weight_norm": float(np.linalg.norm(w_lda)),
        "head_weight_norm": float(np.linalg.norm(w_head)),
        "head_bias": head_b.tolist(),
        "pooled_mean_0": mu0_p.tolist(),
        "pooled_mean_1": mu1_p.tolist(),
    }

    return {"weights": w, "diagnostics": diagnostics}
