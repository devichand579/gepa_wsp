import numpy as np

def edit(w, f):
    # Extract training data
    X = f["x"]          # (4000, 64) int64
    y = f["y"]          # (4000,) int64
    N, L = X.shape
    V = 2000            # vocabulary size

    # Build normalized BOW features: count / length (excluding padding token 0)
    # mask for non-padding
    mask = (X != 0)     # (N, 64)
    lengths = mask.sum(axis=1).astype(np.float32)  # (N,)
    lengths[lengths == 0] = 1.0  # avoid division by zero

    # Accumulate counts
    X_bow = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        # Get token ids for this document, ignore padding
        ids = X[i][mask[i]]
        if len(ids) > 0:
            # np.add.at is safe for repeated indices
            np.add.at(X_bow[i], ids, 1.0)
    # Normalize by document length
    X_bow = X_bow / lengths[:, None]  # (N, V)

    # Convert labels to +1 / -1 for binary classification
    y_bin = np.where(y == 1, 1.0, -1.0)  # (N,)

    # Ridge regression: solve (X^T X + lambda I) w = X^T y
    # Use a moderate lambda for regularization
    lam = 1.0
    # Compute X^T X and X^T y
    # X_bow is (N, V), V=2000. X^T X is (V, V) = 2000x2000, which is 4M floats ~16MB, fine.
    XtX = X_bow.T @ X_bow
    Xty = X_bow.T @ y_bin
    # Add regularization
    XtX += lam * np.eye(V, dtype=np.float32)
    # Solve linear system
    w_target = np.linalg.solve(XtX, Xty)  # (V,)

    # Now we need to map w_target into the network parameters.
    # The model computes: pooled = mean_t LN(e_{w_t}), logits = head @ pooled + bias.
    # For binary classification, we care about the difference logit1 - logit0.
    # Let w_eff = head[1] - head[0] (32-dim). Then score = w_eff @ pooled + (bias[1] - bias[0]).
    # We want score = sum_w (c_w/L) * W_target[w] + b_eff.
    # So we need w_eff @ LN(e_w) = W_target[w] for all w.
    # We can choose w_eff = (1, 0, ..., 0) and set LN(e_w)[0] = W_target[w].
    # To achieve this, we set the final LayerNorm gamma = (c, 0, ..., 0) and beta = (b, 0, ..., 0).
    # Then LN(e)[0] = c * (e[0] - mean(e)) / std(e) + b.
    # We can choose e_w such that (e_w[0] - mean(e_w)) / std(e_w) = (W_target[w] - b) / c.
    # Let's set c = 1.0, b = 0.0 for simplicity, and scale W_target if needed.
    # But W_target can be large. We can scale c to accommodate.
    # Let's find the max absolute value of W_target.
    max_abs = np.max(np.abs(w_target))
    if max_abs < 1e-6:
        max_abs = 1.0
    # We want the normalized value to be within [-sqrt(31), sqrt(31)] to be achievable with unit variance.
    # Actually, we can set c = max_abs / sqrt(31) * 1.1 to be safe.
    # But simpler: set c = max_abs / 5.0 (since sqrt(31) ~ 5.57). Then the required normalized value is W_target / c.
    # We'll set c = max_abs / 5.0, so that W_target / c is within [-5, 5], which is achievable.
    c = max_abs / 5.0
    if c < 1e-6:
        c = 1.0
    # Set norm.weight and norm.bias
    w["norm.weight"] = np.zeros(32, dtype=np.float32)
    w["norm.weight"][0] = c
    w["norm.bias"] = np.zeros(32, dtype=np.float32)

    # Set head: we want w_eff = (1, 0, ..., 0). So head[1] - head[0] = (1, 0, ..., 0).
    # Let head[0] = (0, 0, ..., 0), head[1] = (1, 0, ..., 0).
    # Then logit1 - logit0 = pooled[0] + (bias[1] - bias[0]).
    # We want this to equal sum_w (c_w/L) * W_target[w] + b_eff.
    # pooled[0] = mean_t LN(e_{w_t})[0] = mean_t (c * (e[0]-mean)/std + 0) = c * mean_t ((e[0]-mean)/std).
    # We want c * mean_t ((e[0]-mean)/std) = sum_w (c_w/L) * W_target[w].
    # So mean_t ((e[0]-mean)/std) = sum_w (c_w/L) * (W_target[w] / c).
    # So we need for each word w: (e_w[0] - mean(e_w)) / std(e_w) = W_target[w] / c.
    # Let's denote target_norm[w] = W_target[w] / c.
    # We need to construct e_w (32-dim) such that its first component after normalization equals target_norm[w].
    # For a vector e, let mu = mean(e), sigma = std(e). Then (e[0] - mu) / sigma = target.
    # We can choose e to have zero mean and unit variance? Actually, we can just set e_w = [target_norm[w], 0, 0, ..., 0] and then adjust to have zero mean and unit variance.
    # But LayerNorm subtracts mean and divides by std. If we set e_w = [x, 0, ..., 0], then mean = x/32, std = sqrt((x^2 - x^2/32)/32) = sqrt(31/32^2 * x^2) = |x| * sqrt(31)/32.
    # Then (e[0] - mean)/std = (x - x/32) / (|x| * sqrt(31)/32) = (31x/32) / (|x| * sqrt(31)/32) = 31x / (|x| * sqrt(31)) = sign(x) * sqrt(31).
    # So the normalized value is always ±sqrt(31), independent of x! That's because the vector has only one non-zero component, so its normalized value is fixed.
    # To get a specific normalized value, we need a vector with more components. We can use a vector of the form [a, b, b, ..., b].
    # Let e = [a, b, b, ..., b] (32 elements). Then mean = (a + 31b)/32. 
    # Let's set b = 0 for simplicity? Then we get the fixed ±sqrt(31). So we need b != 0.
    # We want (a - mean) / std = t, where t = target_norm[w].
    # Let's set the vector to have zero mean and unit variance. Then (e[0] - 0)/1 = e[0] = t. So we need a vector with zero mean, unit variance, and first component t.
    # Such a vector exists if |t| <= sqrt(31). Since we set c such that |t| <= 5, and sqrt(31) ≈ 5.57, it's achievable.
    # We can construct it as follows: let the remaining 31 components be equal to -t/31. Then sum = t + 31*(-t/31) = 0. 
    # Variance: sum of squares = t^2 + 31*(t/31)^2 = t^2 + t^2/31 = t^2 * 32/31. 
    # To have unit variance, we need sum of squares / 32 = 1 => sum of squares = 32. 
    # So we need t^2 * 32/31 = 32 => t^2 = 31 => t = ±sqrt(31). So this construction only works for t = ±sqrt(31).
    # To get arbitrary t, we need to add an extra dimension. We can use a vector of the form [t, u, u, ..., u] and then scale it.
    # Let's find a vector e such that (e[0] - mean(e)) / std(e) = t.
    # Let e = [t, 0, 0, ..., 0] + s * [1, 1, ..., 1]? That doesn't change the normalized value.
    # Actually, the normalized value only depends on the direction of the vector, not its scale. So we can first construct a vector with zero mean and unit variance, then scale it? No, scaling doesn't change the normalized value.
    # So we need to find a vector on the unit sphere (zero mean, unit variance) whose first component is t.
    # This is possible for any t in [-sqrt(31), sqrt(31)]. We can construct it by taking a vector with first component t, and the remaining 31 components summing to -t, and having the correct sum of squares.
    # Let the remaining components be all equal to v. Then 31v = -t => v = -t/31.
    # Sum of squares = t^2 + 31*(t/31)^2 = t^2 + t^2/31 = t^2 * 32/31.
    # We need sum of squares = 32 (since variance = 1, mean = 0, so sum of squares = 32).
    # So t^2 * 32/31 = 32 => t^2 = 31. So this only works for t = ±sqrt(31).
    # To get other t, we need to vary the remaining components. We can set the remaining 31 components to have a sum of -t, and sum of squares = 32 - t^2.
    # This is possible as long as 32 - t^2 >= (t^2)/31 (by Cauchy-Schwarz). Since |t| <= sqrt(31), 32 - t^2 >= 1, and (t^2)/31 <= 1, so it's always possible.
    # So we can construct such a vector. For simplicity, we can just use a numerical method: start with a vector with first component t, and the rest zero, then subtract the mean and normalize. But that gives a fixed normalized value. 
    # Actually, we can just set e_w to be a vector where the first component is t, and the rest are chosen to make the mean zero and variance 1. 
    # Let's do this: we want a vector e such that (e[0] - mean(e)) / std(e) = t.
    # Let's set e[0] = t. Let the other 31 components be a constant v. Then mean = (t + 31v)/32.
    # We want (t - (t+31v)/32) / std = t. This is getting complicated.
    # Alternative: we can just set the embedding matrix to be a random matrix, and then use the head to fit? No, we need exact mapping.
    # Actually, we don't need to use the final LayerNorm to do the normalization. We can set the final LayerNorm to be identity by setting gamma=1, beta=0, and ensuring each token embedding has zero mean and unit variance. But then we can't control the first component arbitrarily because it's constrained to be on the sphere. But we can use the head to project. Wait, we have w_eff = head[1] - head[0]. We can choose w_eff to be any 32-dim vector. Then we need w_eff @ LN(e_w) = W_target[w]. If we set LN(e_w) to be a vector on the sphere, then w_eff @ LN(e_w) is a linear function of the sphere. We can choose e_w to be any point on the sphere. So for a given w_eff, the set of achievable W_target[w] is the interval [-||w_eff||, ||w_eff||] (since the dot product of a fixed vector with a unit vector ranges from -||w_eff|| to ||w_eff||). So if we choose w_eff to be a vector of norm R, we can achieve any W_target[w] in [-R, R] by choosing e_w appropriately. So we don't need to set w_eff = (1,0,...,0). We can just set w_eff to be a random vector of large norm, and then for each word, we can find a point on the sphere that gives the desired dot product. This is much simpler! 
    # Let's do that: set w_eff = (R, 0, 0, ..., 0) where R is large enough. Then we need LN(e_w)[0] = W_target[w] / R. We can choose R = max_abs / 5.0, so that W_target[w]/R is in [-5, 5]. Then we need to find e_w such that LN(e_w)[0] = t_w = W_target[w] / R. 
    # We can set the final LayerNorm to be identity (gamma=1, beta=0). Then LN(e) = (e - mean(e)) / std(e). So we need (e[0] - mean(e)) / std(e) = t_w.
    # We can construct such an e_w as follows: let e_w be a vector with zero mean and unit variance, and first component t_w. As discussed, this is possible if |t_w| <= sqrt(31). Since we set R such that |t_w| <= 5, and sqrt(31) ≈ 5.57, it's possible.
    # To construct e_w: we want sum(e) = 0, sum(e^2) = 32, and e[0] = t_w.
    # Let the remaining 31 components be all equal to v. Then 31v = -t_w => v = -t_w/31.
    # Sum of squares = t_w^2 + 31*(t_w/31)^2 = t_w^2 + t_w^2/31 = t_w^2 * 32/31.
    # We need this to equal 32. So t_w^2 * 32/31 = 32 => t_w^2 = 31. This only works for t_w = ±sqrt(31).
    # So we need to vary the remaining components. Let's set the remaining 31 components to be a constant v plus a small perturbation? Actually, we can just use a numerical approach: we want a vector e such that (e - mean(e)) / std(e) = t_w. This is equivalent to e = mean(e) + std(e) * t_w * u, where u is a unit vector with first component 1? No.
    # Simpler: we can just set e_w = [t_w, 0, 0, ..., 0] and then apply LayerNorm. But LayerNorm will normalize it, giving a fixed value. So that doesn't work.
    # Wait, we can set the final LayerNorm gamma and beta to help. We have LN(e) = (e - mean(e)) / std(e) * gamma + beta.
    # We want LN(e)[0] = t_w. We can set gamma = (1, 0, ..., 0) and beta = (0, 0, ..., 0). Then LN(e)[0] = (e[0] - mean(e)) / std(e). So we still need the normalized first component to be t_w.
    # Alternatively, we can set gamma = (g, 0, ..., 0) and beta = (b, 0, ..., 0). Then LN(e)[0] = g * (e[0] - mean(e))/std(e) + b. We can choose e to be a fixed vector (e.g., all zeros except one) and then adjust g and b? No, g and b are shared across all words. So we can't set individual t_w with a single g and b unless we vary e_w.
    # So we need to vary e_w to achieve different t_w.
    # Let's construct e_w properly. We want a vector e of length 32 such that its mean is 0, its variance is 1, and its first component is t. 
    # This is a standard problem: find a vector on the unit sphere (after centering and scaling) with a given first component. 
    # We can do this: let e = [t, a, a, ..., a] + [0, d, -d, 0, ..., 0]? We need to satisfy sum = 0 and sum of squares = 32.
    # Let's set e[0] = t. Let the remaining 31 components be such that their sum is -t and their sum of squares is 32 - t^2.
    # We can choose them to be all equal to -t/31, but then the sum of squares is t^2/31, which is too small. We need to add some variation.
    # Let's set the remaining components to be: v = -t/31 + delta_i, where sum(delta_i) = 0 and sum(delta_i^2) = 32 - t^2 - t^2/31 = 32 - t^2 * 32/31.
    # We can choose delta_i to be a simple pattern: e.g., let delta_1 = d, delta_2 = -d, and the rest 0. Then sum(delta) = 0. Sum of squares = 2d^2.
    # So we need 2d^2 = 32 - t^2 * 32/31 = 32 * (1 - t^2/31) = 32 * (31 - t^2)/31.
    # So d^2 = 16 * (31 - t^2)/31.
    # This is valid as long as 31 - t^2 >= 0, i.e., |t| <= sqrt(31). Since we set |t| <= 5, it's valid.
    # So we can construct e_w as follows:
    # e_w[0] = t_w
    # e_w[1] = -t_w/31 + d
    # e_w[2] = -t_w/31 - d
    # e_w[3..31] = -t_w/31
    # where d = sqrt(16 * (31 - t_w^2)/31).
    # Let's check: sum = t_w + 31*(-t_w/31) + d - d = 0. Good.
    # Sum of squares = t_w^2 + 31*(t_w/31)^2 + 2d^2 = t_w^2 + t_w^2/31 + 2d^2 = t_w^2 * 32/31 + 32*(31 - t_w^2)/31 = (32 t_w^2 + 32*31 - 32 t_w^2)/31 = 32*31/31 = 32. Good.
    # So this construction works! 
    # We can do this for each word w. We have 2000 words. We can compute t_w = W_target[w] / R, where R is chosen so that |t_w| <= 5.
    # Let's set R = max_abs / 5.0. Then |t_w| <= 5. 
    # Then we set the embedding matrix E (2000x32) with these vectors.
    # We also need to set the final LayerNorm to be identity: norm.weight = 1, norm.bias = 0.
    # Then LN(e_w) = (e_w - mean(e_w)) / std(e_w) = e_w (since we constructed e_w to have mean 0 and std 1). So LN(e_w) = e_w.
    # Then pooled = mean_t e_{w_t}. Then logits = head @ pooled + bias.
    # We want logit1 - logit0 = R * pooled[0] + (bias[1] - bias[0]).
    # We set head[0] = (0, 0, ..., 0), head[1] = (R, 0, ..., 0).
    # Then logit1 - logit0 = R * pooled[0] + (bias[1] - bias[0]).
    # We want this to equal sum_w (c_w/L) * W_target[w] + b_eff.
    # pooled[0] = mean_t e_{w_t}[0] = mean_t t_{w_t} = mean_t (W_target[w_t] / R) = (1/R) * mean_t W_target[w_t] = (1/R) * sum_w (c_w/L) * W_target[w].
    # So R * pooled[0] = sum_w (c_w/L) * W_target[w]. Perfect!
    # So we just need to set bias[1] - bias[0] = b_eff. We can set bias[0] = 0, bias[1] = b_eff. But what should b_eff be? The ridge regression gives a weight vector w_target, but it doesn't give a bias. We can compute the optimal bias by taking the mean of the scores? Actually, we can just set bias[1] - bias[0] = 0, or we can compute it from the data. Since we used ridge regression without bias, the model will have no bias. But we can add a bias by centering the features. Let's just set bias to 0. The accuracy might be slightly lower without bias, but it's fine. Alternatively, we can compute the optimal bias by finding the threshold that maximizes accuracy on the training set. We can do that: after computing the scores for all training examples using the new weights, we can find the best threshold. But we can't run forward pass. However, we can compute the scores ourselves: score_i = sum_w (c_{i,w}/L_i) * W_target[w]. Then we can find the best bias by trying a range of values. But that's not necessary; we can just set bias[1] - bias[0] = -mean(score) or something. Actually, we can just set bias[0] = 0, bias[1] = 0. The ridge regression already includes an implicit bias? No, we didn't include a bias term in the ridge regression. We can add a constant feature to X_bow. But X_bow is normalized by length, so the sum of features is 1. So a bias term is redundant if we include a constant? Actually, the sum of (c_w/L) is 1. So the features sum to 1. So a bias term is not needed if we have a weight for each word? Wait, the sum of features is 1, so any constant can be absorbed by adding a constant to all weights? No, if we add a constant to all weights, the score changes by that constant times sum(c_w/L) = constant. So a bias is equivalent to adding a constant to all weights. But ridge regression without bias will find weights that sum to something. We can just set bias[1] - bias[0] = 0. The accuracy might be slightly off, but we can also compute the optimal bias by finding the threshold that maximizes accuracy on the training set. Since we have the training labels, we can compute the scores for all training examples and then find the best threshold. That's easy: we can compute score_i = X_bow[i] @ w_target. Then we can sort the scores and find the threshold that best separates the classes. Then set bias[1] - bias[0] = -threshold. Let's do that.

    # Compute scores for training data
    scores = X_bow @ w_target  # (N,)
    # Find best threshold
    # Sort scores and labels
    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_y = y_bin[order]
    # We want to find threshold t such that sign(score - t) predicts y_bin.
    # We can try all midpoints between consecutive scores.
    # For efficiency, we can compute the number of correct predictions for each threshold.
    # Since N=4000, we can do a simple loop or vectorized.
    # Let's just use a simple approach: try thresholds at each score value.
    best_acc = 0
    best_thresh = 0
    # We can compute cumulative sums of y_bin to quickly evaluate.
    # Actually, we want to maximize accuracy = (TP + TN) / N.
    # For a threshold t, predict +1 if score > t, else -1.
    # We can compute for each t: correct = sum_{i: score_i > t} (y_bin_i == 1) + sum_{i: score_i <= t} (y_bin_i == -1).
    # Let's just do a simple loop over unique scores.
    unique_scores = np.unique(scores)
    for t in unique_scores:
        pred = np.where(scores > t, 1.0, -1.0)
        acc = np.mean(pred == y_bin)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
    # Also try t = -inf (all +1) and t = +inf (all -1)
    acc_all_pos = np.mean(y_bin == 1.0)
    acc_all_neg = np.mean(y_bin == -1.0)
    if acc_all_pos > best_acc:
        best_acc = acc_all_pos
        best_thresh = -np.inf
    if acc_all_neg > best_acc:
        best_acc = acc_all_neg
        best_thresh = np.inf
    # Set bias so that score > best_thresh predicts +1.
    # Our model computes logit1 - logit0 = R * pooled[0] + (bias[1] - bias[0]).
    # We want this to be > 0 for +1. So we set bias[1] - bias[0] = -R * best_thresh? Wait, pooled[0] = (1/R) * score.
    # So logit1 - logit0 = score + (bias[1] - bias[0]).
    # We want score + (bias[1] - bias[0]) > 0 for +1, i.e., score > -(bias[1] - bias[0]).
    # So we set -(bias[1] - bias[0]) = best_thresh => bias[1] - bias[0] = -best_thresh.
    # Let's set bias[0] = 0, bias[1] = -best_thresh.

    # Now set the weights in w.
    # 1. Zero out all transformer blocks and pos.
    for key in w.keys():
        if key.startswith("block") or key == "pos.weight":
            w[key] = np.zeros_like(w[key])
    # 2. Set embed.weight to the constructed E.
    E = np.zeros((V, 32), dtype=np.float32)
    for w_idx in range(V):
        t = w_target[w_idx] / R
        # Clamp t to [-sqrt(31), sqrt(31)] to be safe
        t = np.clip(t, -np.sqrt(31) + 1e-6, np.sqrt(31) - 1e-6)
        d = np.sqrt(16.0 * (31.0 - t*t) / 31.0)
        e = np.zeros(32, dtype=np.float32)
        e[0] = t
        e[1] = -t/31.0 + d
        e[2] = -t/31.0 - d
        e[3:] = -t/31.0
        E[w_idx] = e
    w["embed.weight"] = E
    # 3. Set final LayerNorm to identity
    w["norm.weight"] = np.ones(32, dtype=np.float32)
    w["norm.bias"] = np.zeros(32, dtype=np.float32)
    # 4. Set head
    head = np.zeros((2, 32), dtype=np.float32)
    head[1, 0] = R
    w["head.weight"] = head
    # 5. Set head bias
    bias = np.zeros(2, dtype=np.float32)
    bias[1] = -best_thresh
    w["head.bias"] = bias

    # Diagnostics
    diagnostics = {
        "method": "ridge regression on normalized BOW -> embedding + head",
        "train_accuracy": float(best_acc),
        "best_threshold": float(best_thresh) if np.isfinite(best_thresh) else str(best_thresh),
        "max_abs_weight": float(max_abs),
        "R": float(R),
        "embed_norm_mean": float(np.mean(np.linalg.norm(E, axis=1))),
        "head_weight_norm": float(np.linalg.norm(head)),
    }
    return {"weights": w, "diagnostics": diagnostics}
