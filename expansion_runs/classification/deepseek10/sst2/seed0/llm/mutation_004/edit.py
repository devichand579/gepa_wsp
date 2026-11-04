import numpy as np

def edit(w, f):
    # Read feedback pool
    x = f["x"].astype(np.int64)          # [N, 64]
    y = f["y"].astype(np.int64)          # [N]
    N, L = x.shape
    V = w["embed.weight"].shape[0]       # 2000

    # Build BOW counts and document lengths
    mask = (x != 0)
    lengths = mask.sum(axis=1).astype(np.float64)
    lengths[lengths == 0] = 1.0

    # Count occurrences of each token in each document
    # x is [N, 64], values in [0, V-1]
    # We can use np.add.at for efficiency
    bow = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), L)
    cols = x.reshape(-1)
    np.add.at(bow, (rows, cols), 1.0)

    # Remove padding counts (token 0) from BOW
    bow[:, 0] = 0.0

    # Compute IDF from the full feedback pool
    df = (bow > 0).sum(axis=0).astype(np.float64)
    idf = np.log((1.0 + N) / (1.0 + df)) + 1.0
    idf[0] = 0.0  # padding

    # TF-IDF features: term frequency = count / length, then multiply by IDF
    tf = bow / lengths[:, None]
    tfidf = tf * idf[None, :]

    # Fit a linear classifier on TF-IDF using ridge regression on labels 0/1
    # We solve for w_cls in: tfidf @ w_cls + b ~ y
    # Use closed-form ridge: [w; b] = (X^T X + λI)^-1 X^T y
    X = np.concatenate([tfidf, np.ones((N, 1))], axis=1)  # [N, V+1]
    y_reg = y.astype(np.float64)
    lam = 1.0  # regularization strength
    A = X.T @ X + lam * np.eye(V + 1)
    b_vec = X.T @ y_reg
    coef = np.linalg.solve(A, b_vec)
    w_cls = coef[:V]
    b_cls = coef[V]

    # We want the network's pooled feature to be approximately the TF-IDF score.
    # The network computes: pooled = mean( LayerNorm(embed[token]) )
    # We set embed.weight[:, 0] = w_cls * idf? Wait: the pooled feature is mean of embed.
    # We want pooled[0] = sum_t (count_t / L) * w_cls_t * idf_t? Actually we already have tfidf = (count/L)*idf.
    # So we want pooled[0] = sum_t tfidf_t * w_cls_t = sum_t (count_t/L) * (idf_t * w_cls_t).
    # The network's pooled[0] = (1/L) sum_t embed[t,0] * count_t.
    # So we need embed[t,0] = w_cls_t * idf_t.
    embed_col0 = w_cls * idf

    # To make LayerNorm approximately identity for the first coordinate, we use the same trick:
    # set the remaining 31 coordinates to a constant alternating pattern that dominates the mean and variance.
    # The pattern must have zero mean and unit variance across the 32 dims for each token.
    # We can use a fixed pattern: +1, -1, +1, -1, ... for indices 1..31.
    # But we need 31 values. Let's use alternating +1, -1.
    pattern = np.ones(31, dtype=np.float64)
    pattern[1::2] = -1.0
    # Scale pattern so that its variance is large compared to embed_col0.
    # We can just set it to a large constant, e.g., 1.0, but then LayerNorm will scale down embed_col0.
    # Actually, the previous solution used 1.0328 for the pattern and small values for the first two coords.
    # The scale of embed_col0 might be small. We can scale the pattern to be large, e.g., 100.0,
    # so that LayerNorm's mean and variance are dominated by the pattern, making the first coord
    # approximately linear. But we must keep absolute values <= 1000.
    scale = 100.0
    pattern = pattern * scale

    # Build new embed.weight
    new_embed = np.zeros_like(w["embed.weight"])
    new_embed[:, 0] = embed_col0
    # For the remaining 31 dims, we need to set them so that for each token, the 32-dim vector
    # has the pattern in dims 1..31. But the pattern is the same for all tokens.
    # So we can just broadcast the pattern to all tokens.
    new_embed[:, 1:] = pattern[None, :]

    # Now the head: we want logit0 = -score, logit1 = +score, or something.
    # The network's head reads pooled features. We want the difference between logits to be
    # proportional to the TF-IDF score. Let's set head.weight[1,0] = C, head.weight[0,0] = -C.
    # We also need to incorporate the bias b_cls. The score is pooled[0] * C + bias.
    # We want score = pooled[0] * C + bias to match w_cls·tfidf + b_cls.
    # But pooled[0] is approximately (1/L) sum embed_col0 * count = w_cls·tfidf (if LayerNorm is identity).
    # Actually, LayerNorm will scale embed_col0 by some factor. Let's compute the scaling factor.
    # For a token, the 32-dim vector is [v, p1, p2, ..., p31] where v = embed_col0[t], p = pattern.
    # The mean is (v + sum(p))/32. sum(p) = 0 (since pattern alternates +1,-1, but 31 is odd, so sum is +1? Let's check: 31 elements, starting with +1, alternating. Indices 1..31: 1:+1, 2:-1, 3:+1, ..., 31:+1. There are 16 +1 and 15 -1, sum = 1. So mean = (v + 1*scale)/32.
    # Variance = ( (v-mean)^2 + sum((p_i - mean)^2) ) / 32.
    # Since scale is large, mean ≈ scale/32, variance ≈ scale^2 * (31/32 - (1/32)^2) ≈ scale^2 * 0.96875.
    # So std ≈ scale * 0.984.
    # After LayerNorm, the first coord becomes (v - mean)/std ≈ (v - scale/32) / (scale*0.984) ≈ v/(scale*0.984) - 1/(32*0.984).
    # So it's approximately linear in v, with a constant offset.
    # The offset is the same for all tokens, so it contributes a constant to the pooled feature.
    # We can absorb that into the head bias.
    # The scaling factor is about 1/(scale*0.984). Let's call it alpha.
    # We want the head to multiply by C such that C * alpha * (w_cls·tfidf) + bias = w_cls·tfidf + b_cls.
    # So C * alpha = 1  => C = 1/alpha = scale * 0.984.
    # Let's set C = scale * 0.984.
    # The constant offset from LayerNorm: for each token, the first coord after LayerNorm has an extra -1/(32*0.984) term.
    # When we average over L tokens, this adds a constant -1/(32*0.984) to pooled[0].
    # So pooled[0] ≈ alpha * (w_cls·tfidf) - 1/(32*0.984).
    # Then logit1 - logit0 = C * pooled[0] + (bias1 - bias0).
    # We want this to equal w_cls·tfidf + b_cls.
    # So C * alpha = 1 => C = 1/alpha.
    # C * (-1/(32*0.984)) + (bias1 - bias0) = b_cls.
    # So bias1 - bias0 = b_cls + C/(32*0.984).
    # We can set bias0 = 0, bias1 = b_cls + C/(32*0.984).
    # But we also need to ensure logits are reasonable. Let's just set:
    alpha = 1.0 / (scale * 0.984)
    C = 1.0 / alpha
    # Actually, let's compute alpha more precisely by simulating LayerNorm on a typical vector.
    # We can just use a simple approach: set C = 1.0, and let the head learn? But we are not training.
    # Better: we can compute the exact scaling by doing a quick calculation.
    # For a token with v=0, the vector is [0, pattern]. LayerNorm output first coord = (0 - mean)/std.
    # mean = sum(pattern)/32 = scale/32.
    # var = ( (0-mean)^2 + sum((p_i-mean)^2) ) / 32.
    # We can compute this exactly for the pattern.
    # Let's just compute the LayerNorm output for a zero v and use that to calibrate.
    # We'll create a dummy vector of length 32: [0, pattern].
    # Compute mean, var, std, then normalized first coord.
    dummy = np.concatenate([[0.0], pattern])
    mean_dummy = dummy.mean()
    var_dummy = dummy.var()
    std_dummy = np.sqrt(var_dummy + 1e-5)
    norm_first = (0.0 - mean_dummy) / std_dummy
    # For a general v, the normalized first coord is (v - mean)/std, where mean and std depend on v.
    # But since v is small compared to scale, we can approximate mean and std as constants.
    # Actually, let's just compute the scaling factor by linearizing around v=0.
    # We can compute the derivative of (v - mean(v))/std(v) with respect to v at v=0.
    # But simpler: we can just set the head weight to a large value and rely on the fact that
    # the previous solution worked with a similar trick. The previous solution used scale=1.0328
    # and head weight 260.2271. So the product was about 260.2271 * 1.0328 ≈ 268.7.
    # In our case, we can just set C = 260.2271 * (scale / 1.0328) to keep similar scaling.
    # Let's do that: C = 260.2271 * (scale / 1.0328)
    C = 260.2271 * (scale / 1.0328)
    # Now set head weights: we want logit1 - logit0 = C * pooled[0] + (bias1 - bias0).
    # We set head.weight[0,0] = -C/2, head.weight[1,0] = C/2.
    # Then logit1 - logit0 = C * pooled[0].
    # We need to add b_cls to the difference. So set bias0 = -b_cls/2, bias1 = b_cls/2.
    # But we also need to account for the constant offset from LayerNorm.
    # The offset is norm_first (the value of pooled[0] when all v=0).
    # So pooled[0] = alpha * (w_cls·tfidf) + norm_first.
    # We want C * pooled[0] + (bias1 - bias0) = w_cls·tfidf + b_cls.
    # So C * alpha = 1 => alpha = 1/C.
    # C * norm_first + (bias1 - bias0) = b_cls.
    # So bias1 - bias0 = b_cls - C * norm_first.
    # We can set bias0 = -(b_cls - C * norm_first)/2, bias1 = (b_cls - C * norm_first)/2.
    # Let's compute alpha from our C: alpha = 1/C.
    # But we also need to ensure that the actual alpha matches. We can compute alpha by simulating LayerNorm
    # on a vector with v=1 and v=0.
    # Let's do a quick calibration:
    v_test = 1.0
    vec1 = np.concatenate([[v_test], pattern])
    mean1 = vec1.mean()
    var1 = vec1.var()
    std1 = np.sqrt(var1 + 1e-5)
    norm1 = (v_test - mean1) / std1
    # For v=0:
    vec0 = np.concatenate([[0.0], pattern])
    mean0 = vec0.mean()
    var0 = vec0.var()
    std0 = np.sqrt(var0 + 1e-5)
    norm0 = (0.0 - mean0) / std0
    # The scaling factor alpha is approximately (norm1 - norm0) / v_test.
    alpha_actual = (norm1 - norm0) / v_test
    # We want C * alpha_actual = 1, so C = 1/alpha_actual.
    C = 1.0 / alpha_actual
    # Now compute the offset: pooled[0] when all v=0 is norm0.
    # So we want C * norm0 + (bias1 - bias0) = b_cls.
    # Let's set bias0 = 0, bias1 = b_cls - C * norm0.
    # But we need to be careful: the head has two outputs. We can set:
    # head.weight[0,0] = 0, head.weight[1,0] = C.
    # Then logit0 = bias0, logit1 = C * pooled[0] + bias1.
    # We want logit1 - logit0 = C * pooled[0] + (bias1 - bias0) = w_cls·tfidf + b_cls.
    # Since pooled[0] ≈ alpha_actual * (w_cls·tfidf) + norm0, we have:
    # C * (alpha_actual * score + norm0) + (bias1 - bias0) = score + b_cls.
    # With C = 1/alpha_actual, this becomes score + C*norm0 + (bias1 - bias0) = score + b_cls.
    # So bias1 - bias0 = b_cls - C*norm0.
    # We can set bias0 = 0, bias1 = b_cls - C*norm0.
    # But we also want the logits to be centered. Let's set:
    bias0 = -0.5 * (b_cls - C * norm0)
    bias1 = 0.5 * (b_cls - C * norm0)
    # Actually, the difference is what matters. We can set bias0 = 0, bias1 = b_cls - C*norm0.
    # But to keep logits balanced, we can set them symmetric.
    # Let's just set:
    head_w = np.zeros_like(w["head.weight"])
    head_w[0, 0] = -C / 2.0
    head_w[1, 0] = C / 2.0
    head_b = np.array([-0.5 * (b_cls - C * norm0), 0.5 * (b_cls - C * norm0)], dtype=np.float32)

    # Update weights
    w["embed.weight"] = new_embed.astype(np.float32)
    w["head.weight"] = head_w.astype(np.float32)
    w["head.bias"] = head_b.astype(np.float32)

    # Zero out positional embedding and all transformer block weights to keep identity
    for k in list(w.keys()):
        if k.startswith("pos.") or k.startswith("block0.") or k.startswith("block1."):
            w[k] = np.zeros_like(w[k])
    # Set norm weights to 1, biases to 0
    for k in ["norm.weight", "norm.bias"]:
        if k in w:
            w[k] = np.ones_like(w[k]) if "weight" in k else np.zeros_like(w[k])
    # Also set all norm1/norm2 weights to 1, biases to 0 (already zeroed above, but ensure)
    for k in w:
        if "norm" in k and "weight" in k:
            w[k] = np.ones_like(w[k])
        if "norm" in k and "bias" in k:
            w[k] = np.zeros_like(w[k])

    # Diagnostics
    diag = {
        "train_accuracy": float(( (tfidf @ w_cls + b_cls) > 0.5 ).astype(np.int64) == y).mean() if False else None,
        "idf_mean": float(idf[1:].mean()),
        "idf_max": float(idf[1:].max()),
        "w_cls_norm": float(np.linalg.norm(w_cls)),
        "C": float(C),
        "norm0": float(norm0),
        "b_cls": float(b_cls),
        "head_bias": head_b.tolist(),
    }
    # Compute actual train accuracy of the embedded model? We can't run forward here, but we can estimate.
    # We'll just return diagnostics.
    return {"weights": w, "diagnostics": diag}
