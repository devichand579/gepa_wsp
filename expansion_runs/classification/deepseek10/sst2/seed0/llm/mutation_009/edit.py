import numpy as np
import math

def edit(w, f):
    # Extract feedback arrays
    X = f["x"]          # [N, 64] int64 token ids
    y = f["y"]          # [N] int64 labels
    N, L = X.shape
    V = w["embed.weight"].shape[0]  # 2000
    D = w["embed.weight"].shape[1]  # 32

    # Build term-frequency matrix: count of each token per example, divided by number of non-pad tokens
    # We use only tokens != 0 (pad). Unknown token 1 is included.
    counts = np.zeros((N, V), dtype=np.float64)
    for i in range(N):
        row = X[i]
        mask = row != 0
        if mask.sum() == 0:
            continue
        tokens = row[mask]
        for t in tokens:
            if 0 <= t < V:
                counts[i, t] += 1.0
        counts[i] /= mask.sum()

    # Ridge regression: fit linear classifier on term frequencies.
    # We want to predict y (0/1). Use ridge with small lambda.
    # Add bias column.
    A = np.concatenate([counts, np.ones((N, 1))], axis=1)  # [N, V+1]
    # Solve (A^T A + lambda I) w = A^T y
    lam = 1.0
    AtA = A.T @ A
    AtA += lam * np.eye(V + 1)
    Aty = A.T @ y.astype(np.float64)
    coef = np.linalg.solve(AtA, Aty)
    token_weights = coef[:V]   # [V]
    bias = coef[V]

    # We want the network to compute logit difference = sum_t count_t * token_weights[t] + bias
    # The network computes: pooled = mean over tokens of LayerNorm(embed[token])
    # logit_diff = head_diff · pooled + head_bias_diff
    # We will set head_diff = [C, 0, ..., 0] with C large enough.
    # Then logit_diff = C * mean(LayerNorm(embed[token])[0]) + head_bias_diff
    # We want C * mean(LayerNorm(embed[token])[0]) = sum_t count_t * token_weights[t] + bias
    # Since mean over tokens = (1/L) sum_t count_t, we need:
    # C * (1/L) sum_t count_t * s_t = sum_t count_t * token_weights[t] + bias
    # This is not exactly possible because of the 1/L factor and the bias.
    # However, the network's pooling is mean, so it inherently divides by length.
    # To match a linear classifier on term frequencies, we should fit the classifier on term frequencies
    # and then the network's mean pooling will exactly compute the term-frequency-weighted sum.
    # Wait: the network computes mean over tokens of LayerNorm(embed[token]).
    # If we set LayerNorm(embed[token])[0] = s_t, then pooled[0] = (1/L) sum_t count_t * s_t.
    # We want logit_diff = sum_t count_t * token_weights[t] + bias.
    # So we need C * (1/L) sum_t count_t * s_t = sum_t count_t * token_weights[t] + bias.
    # This cannot hold for all L unless we include L in the model.
    # But the network's mean pooling is fixed. So we should instead fit a linear classifier
    # on the mean token counts (i.e., term frequencies). That is exactly what we did:
    # counts[i, t] = count_t / L. So the model predicts y from term frequencies.
    # Then the network's pooled[0] = sum_t (count_t / L) * s_t = sum_t counts[i,t] * s_t.
    # So if we set s_t = token_weights[t], then pooled[0] = sum_t counts[i,t] * token_weights[t].
    # Then logit_diff = C * pooled[0] + head_bias_diff = C * (sum_t counts[i,t] * token_weights[t]) + head_bias_diff.
    # We want this to equal sum_t counts[i,t] * token_weights[t] + bias.
    # So we set C = 1, and head_bias_diff = bias.
    # But wait, we need LayerNorm(embed[token])[0] = s_t. Since LayerNorm output has mean 0 and variance 1,
    # its first coordinate can be any value in [-sqrt(31), sqrt(31)] ≈ [-5.568, 5.568].
    # So we need |s_t| <= 5.568. The fitted token_weights may be larger.
    # To allow larger s_t, we can set C > 1 and scale s_t = token_weights[t] / C.
    # Then pooled[0] = (1/C) * sum_t counts[i,t] * token_weights[t].
    # Then logit_diff = C * pooled[0] + head_bias_diff = sum_t counts[i,t] * token_weights[t] + head_bias_diff.
    # So we can choose C to be large enough so that |token_weights[t] / C| <= 5.568.
    # Let C = max(1.0, max(|token_weights|) / 5.0). Then s_t = token_weights[t] / C.
    # We also need to set head_bias_diff = bias.
    # The head has two rows: head[0] and head[1]. We want logit_diff = logit1 - logit0.
    # So we set head[1] - head[0] = [C, 0, ..., 0], and head.bias[1] - head.bias[0] = bias.
    # We can set head[0] = 0, head[1] = [C, 0, ..., 0], head.bias[0] = 0, head.bias[1] = bias.
    # But we must keep the original shapes and not exceed 1000 in absolute value.
    # So we need to ensure C <= 1000 and bias <= 1000.

    max_w = np.max(np.abs(token_weights))
    C = max(1.0, max_w / 5.0)
    if C > 1000.0:
        C = 1000.0
    s = token_weights / C

    # Now construct embed[token] such that LayerNorm(embed[token])[0] = s[token].
    # We need a vector v of dimension D=32 with mean 0 and variance 1, and v[0] = s[token].
    # We can set v[0] = s, and the remaining 31 coordinates to some values that make mean 0 and variance 1.
    # Let the remaining coordinates be a constant a. Then mean = (s + 31a)/32 = 0 => a = -s/31.
    # Variance = (1/32) * ( (s - 0)^2 + 31*(a - 0)^2 ) = (1/32) * ( s^2 + 31*(s^2/961) ) = (1/32) * s^2 * (1 + 1/31) = s^2 / 31.
    # So variance = s^2 / 31. To make variance 1, we need to scale the whole vector by sqrt(31)/|s|.
    # But then v[0] becomes s * sqrt(31)/|s| = sqrt(31) * sign(s), losing magnitude.
    # So we cannot have both mean 0, variance 1, and v[0] = s with arbitrary s if we set the rest equal.
    # We need to use the other coordinates to absorb the variance while keeping v[0] = s.
    # We want a vector v with mean 0, variance 1, and v[0] = s.
    # Let v[0] = s. Let the other 31 coordinates be some values. We need sum(v) = 0 => sum_{i=1}^{31} v[i] = -s.
    # We need sum(v^2) = 32 (since variance 1 means sum((v - mean)^2) = 32, and mean=0 so sum(v^2)=32).
    # So we need sum_{i=1}^{31} v[i]^2 = 32 - s^2.
    # This is possible as long as 32 - s^2 >= 0, i.e., |s| <= sqrt(32) ≈ 5.657.
    # So we can set the other 31 coordinates to be equal to -s/31, but then their sum of squares is 31*(s/31)^2 = s^2/31.
    # Then total sum of squares = s^2 + s^2/31 = s^2 * 32/31. We need this to equal 32, so s^2 * 32/31 = 32 => s^2 = 31 => |s| = sqrt(31) ≈ 5.568.
    # So if we set the other coordinates equal, we are forced to |s| = sqrt(31). To get any |s| <= sqrt(32), we need to distribute the remaining sum of squares differently.
    # We can simply set the other 31 coordinates to have sum = -s, and sum of squares = 32 - s^2.
    # For example, we can set 30 of them to some value a, and one to b, such that 30a + b = -s and 30a^2 + b^2 = 32 - s^2.
    # This is solvable. But we can also just use a simple construction: let v[0] = s. Let v[1] = -s. Let the remaining 30 coordinates be 0.
    # Then mean = 0. Sum of squares = s^2 + s^2 = 2s^2. We need this to be 32, so s^2 = 16 => |s| = 4. Not general.
    # Better: we can set v[0] = s. Let the other 31 coordinates be a constant c. Then mean = (s + 31c)/32 = 0 => c = -s/31.
    # Then sum of squares = s^2 + 31*(s/31)^2 = s^2 * (1 + 1/31) = s^2 * 32/31.
    # We want this to be 32, so s^2 = 31. So this only works for |s| = sqrt(31).
    # To get arbitrary s, we can add a vector orthogonal to the all-ones vector and to the first coordinate.
    # Let v = [s, -s/31, -s/31, ..., -s/31] + u, where u has mean 0 and u[0] = 0.
    # Then mean(v) = 0. We need sum(v^2) = 32.
    # sum(v^2) = s^2 + 31*(s/31)^2 + sum(u^2) = s^2 * 32/31 + sum(u^2).
    # We need s^2 * 32/31 + sum(u^2) = 32 => sum(u^2) = 32 - s^2 * 32/31 = 32 * (1 - s^2/31).
    # This is non-negative as long as s^2 <= 31. So for |s| <= sqrt(31), we can set u = 0 and scale the whole vector?
    # Wait, if we set u = 0, then sum(v^2) = s^2 * 32/31. To make it 32, we need to scale v by a factor k such that k^2 * s^2 * 32/31 = 32 => k^2 = 31/s^2 => k = sqrt(31)/|s|.
    # But then v[0] becomes s * sqrt(31)/|s| = sqrt(31) * sign(s). So the first coordinate becomes fixed magnitude. That's the problem.
    # So we cannot scale the whole vector. We must keep v[0] = s exactly. So we need to add u with u[0] = 0 to adjust the norm without changing v[0].
    # So we need sum(u^2) = 32 - s^2 * 32/31. This is positive for |s| < sqrt(31). For |s| > sqrt(31), this becomes negative, meaning we cannot have v[0] = s with mean 0 and variance 1 if |s| > sqrt(31). Actually, the maximum possible value of a single coordinate in a mean-zero unit-variance vector of dimension 32 is sqrt(31) ≈ 5.568. So we must have |s| <= sqrt(31). So we need to choose C such that |token_weights| / C <= sqrt(31). So C >= max|token_weights| / sqrt(31). Let's use C = max(1.0, max_w / 5.0) to be safe. Then |s| <= 5.0. So we can construct v.
    # Construction: v[0] = s. Let the other 31 coordinates be a constant c = -s/31. Then sum(v^2) = s^2 * 32/31. We need this to be 32. If it's not, we can add a vector u with u[0]=0, mean 0, and sum(u^2) = 32 - s^2 * 32/31. We can just set u[1] = sqrt(32 - s^2 * 32/31), and the rest 0. But we must ensure mean(u) = 0. If we set u[1] = d, and u[2] = -d, then mean(u)=0, sum(u^2)=2d^2. We need 2d^2 = 32 - s^2 * 32/31. So d = sqrt((32 - s^2 * 32/31)/2). This works as long as the argument is non-negative. Since |s| <= 5, s^2 <= 25, 32 - 25*32/31 = 32*(1 - 25/31) = 32*(6/31) ≈ 6.19, positive. So we can do this.
    # Actually, we can simplify: we don't need to pre-normalize the embedding. We can just set embed[token] to some vector, and let LayerNorm normalize it. But as we saw, LayerNorm of a vector with a single large coordinate and rest equal gives a fixed normalized first coordinate. To get a specific normalized first coordinate, we need to design the vector such that after normalization, the first coordinate is s. This is equivalent to finding a vector v such that (v[0] - mean(v))/std(v) = s. We can solve this directly. Let v = [x, y, y, ..., y] (32 dims). We already derived that (x - mean)/std = sqrt(31) * sign(x - y). So with this form, we can only get ±sqrt(31). To get arbitrary s, we need to vary the other coordinates. So we can set v = [x, y, z, z, ..., z] with 30 z's. Then we have more freedom. But it's easier to just construct the normalized vector we want, and then set embed[token] = that normalized vector. Since LayerNorm of a vector that already has mean 0 and variance 1 is itself (if weight=1, bias=0), we can just set embed[token] to be the desired normalized vector. So we need to construct a vector v of dimension 32 with mean 0, variance 1, and v[0] = s. As argued, this is possible for |s| <= sqrt(31). Let's construct it simply:
    # v[0] = s
    # v[1] = -s
    # v[2] = sqrt(32 - 2*s^2) / sqrt(2) ? Wait, we need sum(v) = 0 and sum(v^2) = 32.
    # Let v[0] = s, v[1] = -s. Then sum = 0. Sum of squares = 2s^2.
    # We need remaining 30 coordinates to have sum 0 and sum of squares = 32 - 2s^2.
    # We can set v[2] = a, v[3] = -a, and the rest 0. Then sum = 0, sum of squares = 2a^2.
    # We need 2a^2 = 32 - 2s^2 => a^2 = 16 - s^2. So a = sqrt(16 - s^2).
    # This works as long as 16 - s^2 >= 0, i.e., |s| <= 4.
    # For |s| up to 5.5, we need more coordinates. We can use 4 coordinates: v[0]=s, v[1]=-s, v[2]=a, v[3]=-a, v[4]=b, v[5]=-b, etc.
    # Actually, we can just use the following simple method: set v[0] = s. Set the other 31 coordinates to be equal to -s/31. Then mean = 0. Sum of squares = s^2 + 31*(s/31)^2 = s^2 * 32/31. We need this to be 32. If it's not, we can scale the other 31 coordinates by a factor k such that mean remains 0 and sum of squares becomes 32. Let v[0] = s. Let v[i] = c for i>=1. Then mean = (s + 31c)/32 = 0 => c = -s/31. Sum of squares = s^2 + 31c^2 = s^2 + 31*(s^2/961) = s^2 * (1 + 1/31) = s^2 * 32/31. To make this 32, we need s^2 = 31. So this only works for |s| = sqrt(31). To get arbitrary s, we can set v[0] = s, and set the other 31 coordinates to be c + d_i, where sum(d_i) = 0 and sum((c+d_i)^2) = 32 - s^2. We can just set c = -s/31, and then add a vector d with sum(d)=0 and sum(d^2) = 32 - s^2 * 32/31. We can set d[1] = e, d[2] = -e, rest 0. Then sum(d)=0, sum(d^2)=2e^2. So 2e^2 = 32 - s^2 * 32/31 => e = sqrt(16 - s^2 * 16/31). This is real for |s| <= sqrt(31). So we can construct v as:
    # v[0] = s
    # v[1] = -s/31 + e
    # v[2] = -s/31 - e
    # v[3..31] = -s/31
    # where e = sqrt(16 - s^2 * 16/31).
    # Let's check: mean = (s + 31*(-s/31) + e - e)/32 = 0. Sum of squares = s^2 + (-s/31+e)^2 + (-s/31-e)^2 + 29*(s^2/31^2) = s^2 + 2*(s^2/31^2 + e^2) + 29*s^2/961 = s^2 + 2s^2/961 + 2e^2 + 29s^2/961 = s^2 + 31s^2/961 + 2e^2 = s^2 + s^2/31 + 2e^2 = s^2 * 32/31 + 2e^2. We want this to be 32. So 2e^2 = 32 - s^2 * 32/31 => e^2 = 16 - s^2 * 16/31. Yes.
    # So we can compute v for each token. Then set embed[token] = v. Since v already has mean 0 and variance 1, LayerNorm will leave it unchanged (if norm.weight=1, bias=0). So we set norm.weight=1, norm.bias=0. Also set all block weights to zero, pos to zero, and head as described.

    # Let's implement this construction.
    embed = np.zeros((V, D), dtype=np.float32)
    for t in range(V):
        s_t = s[t]
        # clamp s_t to [-sqrt(31)+eps, sqrt(31)-eps] to avoid numerical issues
        s_t = np.clip(s_t, -5.5, 5.5)
        e2 = 16.0 - s_t * s_t * 16.0 / 31.0
        if e2 < 0:
            e2 = 0.0
        e = math.sqrt(e2)
        v = np.zeros(D, dtype=np.float64)
        v[0] = s_t
        v[1] = -s_t / 31.0 + e
        v[2] = -s_t / 31.0 - e
        v[3:] = -s_t / 31.0
        embed[t] = v.astype(np.float32)

    # Set pos to zero
    w["pos.weight"][:] = 0.0

    # Zero out all block weights and biases to make blocks identity
    for name in w:
        if name.startswith("block0.") or name.startswith("block1."):
            if name.endswith(".weight") or name.endswith(".bias"):
                w[name][:] = 0.0
    # But we must keep norm weights in blocks? Actually if we zero all block weights, the blocks are identity.
    # However, the norm1 and norm2 inside blocks are applied. If we zero their weights, they become zero output.
    # Wait, the forward pass: h = embed + pos. Then block0: z = norm1(h); q,k,v = qkv(z); ...; h = h + proj(attention); h = h + ff2(relu(ff1(norm2(h)))).
    # If we zero qkv, proj, ff1, ff2, then the block just passes h through. But norm1 and norm2 are still applied? No, norm1 is applied to h to produce z, but z is only used in qkv. If qkv weights are zero, z doesn't affect anything. norm2 is applied to h to produce input to ff1, but ff1 weights zero, so no effect. So the block output is just h. So we don't need to zero norm1/norm2 weights. They can stay as identity. But to be safe, we can leave them as is (they are already 1 and 0). Actually, the current weights have block0.norm1.weight=1, bias=0, etc. So they are identity. So we can just zero the linear layers.

    # Set final norm to identity (weight=1, bias=0)
    w["norm.weight"][:] = 1.0
    w["norm.bias"][:] = 0.0

    # Set head weights and bias
    # We want logit_diff = C * pooled[0] + bias
    # pooled[0] = mean over tokens of LayerNorm(embed[token])[0] = mean of s_t = sum_t count_t * s_t / L = sum_t counts[i,t] * s_t
    # We set s_t = token_weights[t] / C.
    # So pooled[0] = (1/C) * sum_t counts[i,t] * token_weights[t]
    # Then logit_diff = C * pooled[0] + bias = sum_t counts[i,t] * token_weights[t] + bias.
    # So we set head[1] - head[0] = [C, 0, ..., 0], and head.bias[1] - head.bias[0] = bias.
    # We can set head[0] = 0, head[1] = [C, 0, ..., 0], head.bias[0] = 0, head.bias[1] = bias.
    # But we must keep values within [-1000, 1000]. C is at most 1000. bias is typically small.
    head_w = np.zeros((2, D), dtype=np.float32)
    head_b = np.zeros(2, dtype=np.float32)
    head_w[1, 0] = C
    head_b[1] = bias
    w["head.weight"][:] = head_w
    w["head.bias"][:] = head_b

    # Also ensure embed weights are within [-1000, 1000]
    embed = np.clip(embed, -1000.0, 1000.0)
    w["embed.weight"][:] = embed

    # Diagnostics
    diag = {
        "C": float(C),
        "bias": float(bias),
        "max_abs_token_weight": float(max_w),
        "embed_mean": float(np.mean(embed)),
        "embed_std": float(np.std(embed)),
        "head_w": head_w.tolist(),
        "head_b": head_b.tolist()
    }
    return {"weights": w, "diagnostics": diag}
