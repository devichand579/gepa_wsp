import numpy as np

def edit(w, f):
    # Extract training data
    x = f["x"]          # [N, 64] int64
    y = f["y"]          # [N] int64
    N = x.shape[0]
    V = w["embed.weight"].shape[0]  # 2000

    # Build bag-of-words matrix, excluding padding token 0
    BOW = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        counts = np.bincount(x[i], minlength=V).astype(np.float32)
        counts[0] = 0.0
        BOW[i] = counts

    # Number of non-pad tokens per example
    n = BOW.sum(axis=1, keepdims=True)
    n[n == 0] = 1.0  # avoid division by zero (should not happen)

    # Normalized BOW features
    X = BOW / n  # [N, V]

    # Targets: -1 for negative (y=0), +1 for positive (y=1)
    y_pm = 2.0 * y.astype(np.float32) - 1.0

    # Center for intercept
    X_mean = X.mean(axis=0)
    y_mean = y_pm.mean()
    X_c = X - X_mean
    y_c = y_pm - y_mean

    # Precompute Gram matrix and X^T y for full data
    G = X_c.T @ X_c          # [V, V]
    b = X_c.T @ y_c          # [V]

    # 5-fold cross-validation to choose lambda
    indices = np.arange(N)
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    folds = np.array_split(indices, 5)

    lambdas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    best_lambda = 1.0
    best_acc = -1.0

    for lam in lambdas:
        accs = []
        for k in range(5):
            val_idx = folds[k]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != k])
            X_tr = X_c[train_idx]
            y_tr = y_c[train_idx]
            X_val = X_c[val_idx]
            y_val = y_pm[val_idx]

            # Solve ridge on training fold
            G_tr = X_tr.T @ X_tr
            b_tr = X_tr.T @ y_tr
            s_tr = np.linalg.solve(G_tr + lam * np.eye(V), b_tr)
            c_tr = y_mean - X_mean @ s_tr  # use full-data means for simplicity

            # Predict on validation fold
            pred = X_val @ s_tr + c_tr
            pred_label = (pred > 0).astype(np.int64)
            true_label = (y_val > 0).astype(np.int64)
            acc = np.mean(pred_label == true_label)
            accs.append(acc)
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lambda = lam

    # Final fit on all data with best lambda
    s = np.linalg.solve(G + best_lambda * np.eye(V), b)
    c = y_mean - X_mean @ s

    # Zero out all transformer blocks, positional embeddings, and set norms to identity
    for key in list(w.keys()):
        if key.startswith("block0.") or key.startswith("block1."):
            w[key][...] = 0.0
        elif key == "pos.weight":
            w[key][...] = 0.0
        elif key.endswith(".norm1.weight") or key.endswith(".norm2.weight") or key == "norm.weight":
            w[key][...] = 1.0
        elif key.endswith(".norm1.bias") or key.endswith(".norm2.bias") or key == "norm.bias":
            w[key][...] = 0.0

    # Build new embed.weight: column 0 = s, other columns chosen so each row is zero-mean/unit-variance
    embed = np.zeros((V, 32), dtype=np.float32)
    embed[:, 0] = s.astype(np.float32)

    # For each row, we need mean 0 and variance 1 across 32 dims.
    # We have one value s[v] in col 0. We need 31 other values such that sum = -s[v] and sum of squares = 32 - s[v]^2.
    # A simple construction: set columns 1..31 to a constant a, except we need sum of squares.
    # Let the other 31 values be a (same for all). Then 31*a = -s[v] => a = -s[v]/31.
    # Sum of squares = s[v]^2 + 31*a^2 = s[v]^2 + s[v]^2/31 = s[v]^2 * (32/31).
    # We need this to equal 32 (for unit variance). So we need to scale s[v] such that s[v]^2 * (32/31) = 32 => s[v]^2 = 31.
    # But s[v] is fixed. Instead, we can use a different construction: set the other 31 values to a and b alternating? 
    # Simpler: we can just set the other columns to zero, and then adjust the row to have zero mean and unit variance by adding a constant to all columns?
    # Actually, we can set embed[:, 1:] = 0, then row mean = s[v]/32, variance = (s[v]^2 - s[v]^2/32)/32 = s[v]^2 * 31 / (32^2). Not unit.
    # We need to explicitly construct rows with zero mean and unit variance.
    # Let's do: for each row v, we want a vector of length 32 with first element s[v], and the rest chosen so that mean=0, var=1.
    # We can set the other 31 elements to a constant a. Then mean = (s[v] + 31a)/32 = 0 => a = -s[v]/31.
    # Variance = (1/32) * (s[v]^2 + 31*a^2) - mean^2 = (1/32)*(s[v]^2 + s[v]^2/31) = s[v]^2 * (32/31) / 32 = s[v]^2 / 31.
    # We want variance = 1 => s[v]^2 = 31 => s[v] = ±sqrt(31). But s[v] is given.
    # So we cannot achieve both zero mean and unit variance with only one non-zero column and constant others.
    # Instead, we can scale the entire row? But we need column 0 to be s[v] for the classifier.
    # We can add a bias to the head to compensate for the mean, and we don't strictly need unit variance.
    # Actually, LayerNorm normalizes each token's embedding. If we don't make it unit variance, LayerNorm will scale it.
    # But if we zero out the blocks, the final output is LayerNorm(embed). Then pooled = mean_t LayerNorm(embed[ids_t]).
    # If we make each row have zero mean and unit variance, LayerNorm is identity. If not, LayerNorm will scale each row by 1/std.
    # Since we want the classifier to work on BOW, we need the pooled feature to be linear in BOW. If LayerNorm scales each row differently, it's not linear.
    # So we must make each row zero-mean and unit-variance.
    # To do this, we can set the other 31 columns to values that sum to -s[v] and sum of squares to 32 - s[v]^2.
    # We can just set them all to a, but then we need s[v]^2 + 31*a^2 = 32 - s[v]^2? Wait, sum of squares = s[v]^2 + 31*a^2 = 32 (since mean=0, var=1 => sum of squares = 32).
    # And sum = s[v] + 31*a = 0 => a = -s[v]/31.
    # Then sum of squares = s[v]^2 + 31*(s[v]^2/31^2) = s[v]^2 * (1 + 1/31) = s[v]^2 * 32/31.
    # We need this to equal 32 => s[v]^2 = 31. So s[v] must be ±sqrt(31) ≈ ±5.57.
    # But our fitted s[v] can be any value. We can scale s and c together! The classifier score is X @ s + c. If we scale s by alpha and c by alpha, the decision boundary is the same. So we can scale s to have norm such that we can construct the rows.
    # Actually, we can just scale s so that max |s[v]| <= sqrt(31). But we need each row to have exactly zero mean and unit variance. We can do this by setting the other 31 columns to a constant a_v = -s[v]/31, and then we need sum of squares = 32. But sum of squares = s[v]^2 + 31*(s[v]^2/31^2) = s[v]^2 * 32/31. This equals 32 only if s[v]^2 = 31. So for rows where s[v]^2 != 31, we can't use a constant. We can use two different constants: set 15 columns to a and 16 columns to b, etc. But we need to solve for each row individually. Since we have 31 free variables, we can always find a vector of length 31 that sums to -s[v] and has sum of squares = 32 - s[v]^2, provided 32 - s[v]^2 >= (-s[v])^2 / 31 (by Cauchy-Schwarz). That is, 32 - s[v]^2 >= s[v]^2 / 31 => 32 >= s[v]^2 * (1 + 1/31) => s[v]^2 <= 32 * 31 / 32 = 31. So as long as |s[v]| <= sqrt(31) ≈ 5.568, we can construct such a vector.
    # Our fitted s values might be larger. We can scale s and c by a factor alpha to ensure max |s[v]| <= sqrt(31). This does not change the classifier's predictions, only the logit scale.
    # So we can compute alpha = min(1.0, sqrt(31) / max(|s|)). Then s_scaled = s * alpha, c_scaled = c * alpha.
    # Then for each row v, we need to find a vector u of length 31 such that sum(u) = -s_scaled[v] and sum(u^2) = 32 - s_scaled[v]^2.
    # We can just set u = a * ones(31) + b * (some pattern)? Actually, we can use a simple construction: set u[0] = x, u[1..30] = y. Then 2 variables, 2 equations. But we have 31 variables, so we can just set u = (-s_scaled[v]/31) * ones(31) and then adjust one element to fix the sum of squares.
    # Let a = -s_scaled[v]/31. Then sum(u) = -s_scaled[v]. Sum of squares = 31*a^2 = s_scaled[v]^2 / 31. We need total sum of squares = 32 - s_scaled[v]^2. So we need to add extra sum of squares: delta = (32 - s_scaled[v]^2) - s_scaled[v]^2/31 = 32 - s_scaled[v]^2 * (32/31).
    # We can add this delta to one element. Let u[0] = a + d. Then sum(u) = -s_scaled[v] + d. To keep sum = -s_scaled[v], we must subtract d from another element. So set u[0] = a + d, u[1] = a - d, others = a. Then sum is unchanged. Sum of squares = 31*a^2 + 2*d^2. We need 31*a^2 + 2*d^2 = 32 - s_scaled[v]^2. So 2*d^2 = 32 - s_scaled[v]^2 - s_scaled[v]^2/31 = 32 - s_scaled[v]^2 * (32/31). So d = sqrt((32 - s_scaled[v]^2 * (32/31)) / 2). This works as long as the expression under sqrt is >= 0, which is exactly the condition |s_scaled[v]| <= sqrt(31).
    # So we can construct each row.
    # Let's implement this.

    # Scale s and c to ensure |s| <= sqrt(31)
    max_s = np.max(np.abs(s))
    if max_s > 0:
        alpha = min(1.0, np.sqrt(31.0) / max_s)
    else:
        alpha = 1.0
    s_scaled = s * alpha
    c_scaled = c * alpha

    # Build embed.weight
    embed = np.zeros((V, 32), dtype=np.float32)
    embed[:, 0] = s_scaled

    for v in range(V):
        sv = s_scaled[v]
        a = -sv / 31.0
        # base sum of squares = 31 * a^2
        # needed extra = 32 - sv^2 - 31*a^2
        extra = 32.0 - sv*sv - 31.0*a*a
        if extra < 0:
            extra = 0.0  # should not happen if scaled properly
        d = np.sqrt(extra / 2.0)
        # Set columns 1..31
        embed[v, 1] = a + d
        embed[v, 2] = a - d
        embed[v, 3:] = a

    # Verify zero mean and unit variance
    # (optional, can be omitted for speed)

    w["embed.weight"][...] = embed

    # Set head.weight: class 1 reads column 0, class 0 reads negative column 0
    head_w = np.zeros((2, 32), dtype=np.float32)
    head_w[1, 0] = 1.0
    head_w[0, 0] = -1.0
    w["head.weight"][...] = head_w

    # Set head.bias: we want logit difference to be 2*(pooled[0] + c_scaled)
    # logits[1] - logits[0] = (pooled[0] + b1) - (-pooled[0] + b0) = 2*pooled[0] + (b1 - b0)
    # We want this to be 2*(pooled[0] + c_scaled) = 2*pooled[0] + 2*c_scaled
    # So b1 - b0 = 2*c_scaled. Set b1 = c_scaled, b0 = -c_scaled.
    w["head.bias"][...] = np.array([-c_scaled, c_scaled], dtype=np.float32)

    # Diagnostics
    diag = {
        "method": "BOW ridge + LayerNorm-identity embedding",
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_acc),
        "alpha": float(alpha),
        "s_norm": float(np.linalg.norm(s_scaled)),
        "c_scaled": float(c_scaled),
        "embed_row_mean_maxabs": float(np.max(np.abs(embed.mean(axis=1)))),
        "embed_row_var_maxdev": float(np.max(np.abs(embed.var(axis=1) - 1.0))),
    }

    return {"weights": w, "diagnostics": diag}
