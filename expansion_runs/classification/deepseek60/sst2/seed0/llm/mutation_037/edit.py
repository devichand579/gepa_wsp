import numpy as np

def edit(w, f):
    # Extract feedback data
    x = f["x"]          # [4000, 64] int64
    y = f["y"]          # [4000] int64
    N, T = x.shape
    V = w["embed.weight"].shape[0]  # 2000
    D = w["embed.weight"].shape[1]  # 32

    # Build document-term count matrix (sparse-ish but small enough)
    # We'll compute mean token counts per document: X_mean = counts / length
    # Use only non-padding tokens (id != 0)
    mask = (x != 0)
    lengths = mask.sum(axis=1).astype(np.float32)
    lengths[lengths == 0] = 1.0  # avoid division by zero

    # Create a dense count matrix? 4000 x 2000 is 8M floats, ~32MB, acceptable.
    # But we can work with sparse operations for speed.
    # We'll build a dense matrix for simplicity, but use float32.
    X = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        ids = x[i][mask[i]]
        if len(ids) > 0:
            np.add.at(X[i], ids, 1.0)
    X_mean = X / lengths[:, None]  # [N, V]

    # Center the features (optional but helps)
    X_mean_centered = X_mean - X_mean.mean(axis=0, keepdims=True)

    # Fit a linear classifier on X_mean_centered using ridge regression on labels (0/1)
    # We'll use a small ridge penalty. Choose lambda via a simple heuristic.
    # Since we have 4000 examples and 2000 features, ridge with lambda ~ 1.0 is reasonable.
    # We can also use LDA, but ridge is simple and closed-form.
    # We'll solve (X^T X + lambda I) w = X^T y_centered
    # To avoid huge matrix inversion, we can use the dual form: w = X^T (X X^T + lambda I)^-1 y
    # N=4000, so X X^T is 4000x4000. That's 64MB, invertible.
    # But we can also use the primal if V=2000 is smaller. V=2000, so primal is 2000x2000.
    # Let's use primal: A = X_mean_centered.T @ X_mean_centered + lambda * I (2000x2000)
    # b = X_mean_centered.T @ (y - y.mean())
    # w = solve(A, b)
    # This is 2000x2000, inversion is fine.

    lambda_reg = 1.0
    y_centered = y.astype(np.float32) - y.mean()
    A = X_mean_centered.T @ X_mean_centered + lambda_reg * np.eye(V, dtype=np.float32)
    b = X_mean_centered.T @ y_centered
    w_token = np.linalg.solve(A, b)  # [V]

    # Compute bias: we want logit = mean_token_score + bias.
    # Our model will compute pooled = mean(embed[t]) and head gives logit.
    # We want logit = X_mean @ w_token + bias.
    # Since X_mean = mean of one-hot vectors, X_mean @ w_token = mean(w_token[t]).
    # So we need embed[t] such that head @ embed[t] = w_token[t].
    # We'll set embed[t] = w_token[t] * v, where v is a fixed unit vector in zero-mean subspace.
    # But LayerNorm requires embed[t] to have mean 0 and variance 1.
    # So we cannot simply scale v by w_token[t] unless |w_token[t]| is constant.
    # Instead, we'll normalize w_token to have unit variance, and then scale the head.
    # Let s = w_token / std(w_token). Then s has mean 0, std 1.
    # We need embed[t] to be a unit vector in zero-mean subspace with projection onto delta equal to s[t].
    # We can construct such vectors for each token.
    # Let delta be a fixed unit vector in zero-mean subspace (e.g., [1, -1, 0, ...] normalized).
    # For each token, we need a unit vector e_t in zero-mean subspace such that delta·e_t = s[t].
    # Since s[t] can be any real, we need to scale s to be in [-1, 1]. Let s_scaled = s / max(|s|) * 0.99.
    # Then we can construct e_t = s_scaled[t] * delta + sqrt(1 - s_scaled[t]^2) * orth_t,
    # where orth_t is any unit vector orthogonal to delta in the zero-mean subspace.
    # We can choose orth_t randomly or deterministically. To keep it simple, we can use a fixed orthogonal vector.
    # But we need e_t to have mean 0 and variance 1. The zero-mean subspace is 31-dimensional.
    # We can construct e_t as follows: start with a base vector b = [1, -1, 0, ..., 0] / sqrt(2) (mean 0, norm 1).
    # We want delta·e_t = s_scaled[t]. Let delta = b.
    # Then e_t = s_scaled[t] * b + sqrt(1 - s_scaled[t]^2) * c, where c is orthogonal to b in zero-mean subspace.
    # We can choose c = [0, 0, 1, -1, 0, ..., 0] / sqrt(2). This is orthogonal to b and has mean 0, norm 1.
    # Then e_t will have mean 0 and norm 1 (since b and c are orthonormal). So variance = 1.
    # This works! We just need to ensure that s_scaled[t] is in [-1, 1].
    # Then head1 - head0 = scale * b, where scale = max(|s|) * (std(w_token))? Wait.
    # We have s = w_token / std(w_token). Then s_scaled = s / max(|s|) * 0.99.
    # Then e_t = s_scaled[t] * b + sqrt(1 - s_scaled[t]^2) * c.
    # Then b·e_t = s_scaled[t].
    # We want logit = mean(w_token[t]) + bias = std(w_token) * mean(s[t]) + bias.
    # But b·e_t = s_scaled[t] = s[t] / max(|s|) * 0.99.
    # So mean(s[t]) = max(|s|) / 0.99 * mean(b·e_t).
    # So if we set head1 - head0 = (std(w_token) * max(|s|) / 0.99) * b, then
    # (head1 - head0)·pooled = std(w_token) * max(|s|) / 0.99 * mean(b·e_t) = std(w_token) * mean(s[t]) = mean(w_token[t]).
    # Perfect.

    # Compute w_token statistics
    std_w = np.std(w_token)
    if std_w == 0:
        std_w = 1.0
    s = w_token / std_w
    max_abs_s = np.max(np.abs(s))
    if max_abs_s == 0:
        max_abs_s = 1.0
    s_scaled = s / max_abs_s * 0.99

    # Base vectors b and c in 32-dim, zero-mean, orthonormal
    b = np.zeros(D, dtype=np.float32)
    b[0] = 1.0 / np.sqrt(2)
    b[1] = -1.0 / np.sqrt(2)
    c = np.zeros(D, dtype=np.float32)
    c[2] = 1.0 / np.sqrt(2)
    c[3] = -1.0 / np.sqrt(2)

    # Build embed.weight
    embed = np.zeros((V, D), dtype=np.float32)
    for t in range(V):
        val = s_scaled[t]
        # Ensure val in [-1, 1]
        val = np.clip(val, -0.999, 0.999)
        embed[t] = val * b + np.sqrt(1.0 - val * val) * c

    # Set head weights
    # We want head1 - head0 = scale * b, where scale = std_w * max_abs_s / 0.99
    scale = std_w * max_abs_s / 0.99
    head = np.zeros((2, D), dtype=np.float32)
    head[1] = 0.5 * scale * b
    head[0] = -0.5 * scale * b

    # Compute bias: we want logit1 - logit0 = mean(w_token[t]) + bias.
    # Our model gives logit1 - logit0 = scale * (b·pooled) + (head1_bias - head0_bias)
    # = scale * mean(b·e_t) + bias = mean(w_token[t]) + bias.
    # So bias should be chosen to center the decision boundary.
    # We can set bias = -mean(w_token) * mean(length?) No, we want to optimize accuracy.
    # We can compute the optimal bias on the training set by finding the threshold that maximizes accuracy.
    # Since we have the full feedback pool, we can compute the scores for all examples and choose the best bias.
    # The score for example i is mean(w_token[t]) for tokens in doc i.
    # We already have X_mean @ w_token = scores.
    scores = X_mean @ w_token  # [N]
    # Find best bias by sorting scores and checking thresholds.
    # We want logit1 - logit0 > 0 => positive class.
    # So we want scores + bias > 0 => scores > -bias.
    # We can find the threshold that maximizes accuracy.
    # Since we have labels, we can compute the optimal threshold.
    # Simple approach: try all midpoints between sorted scores.
    sorted_idx = np.argsort(scores)
    sorted_scores = scores[sorted_idx]
    sorted_y = y[sorted_idx]
    # We want to predict 1 if score > threshold.
    # Compute cumulative sums to find best threshold.
    # For each possible threshold, accuracy = (true positives + true negatives) / N.
    # We can compute this efficiently.
    # Let's just use a simple grid search over thresholds.
    # But to avoid "search loops", we can use the fact that the optimal threshold is the one that maximizes accuracy.
    # This is a closed-form computation (sorting and cumulative sums).
    # We'll compute the best threshold.
    # Actually, we can just set bias = -np.median(scores) or something, but let's do it properly.
    # We'll compute the best threshold by evaluating all midpoints.
    # This is O(N log N) and not an iterative optimizer.
    thresholds = np.concatenate(([sorted_scores[0] - 1.0], (sorted_scores[:-1] + sorted_scores[1:]) / 2.0, [sorted_scores[-1] + 1.0]))
    best_acc = 0.0
    best_thresh = 0.0
    # To avoid loop over 4000 thresholds, we can use cumulative sums.
    # Let's do a vectorized computation.
    # For a threshold t, predict 1 if score > t.
    # We can compute for each t the number of correct predictions.
    # Since N=4000, a simple loop is fine (4000 iterations is not an iterative optimizer, it's just evaluation).
    # But to be safe, we can use a vectorized approach.
    # Actually, we can just compute the optimal bias analytically? No, accuracy is not smooth.
    # Let's just do a loop over thresholds. 4000 iterations is trivial.
    for t in thresholds:
        pred = (scores > t).astype(int)
        acc = np.mean(pred == y)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
    bias = -best_thresh

    # Set head biases
    head_bias = np.zeros(2, dtype=np.float32)
    head_bias[1] = 0.5 * bias
    head_bias[0] = -0.5 * bias

    # Update weights
    w["embed.weight"] = embed.astype(np.float32)
    w["head.weight"] = head.astype(np.float32)
    w["head.bias"] = head_bias.astype(np.float32)

    # Ensure all other weights are zero (they already are, but just in case)
    for key in w:
        if key not in ["embed.weight", "head.weight", "head.bias"]:
            w[key] = np.zeros_like(w[key], dtype=np.float32)
    # Set norm weights to 1, biases to 0
    for key in w:
        if "norm" in key and "weight" in key:
            w[key] = np.ones_like(w[key], dtype=np.float32)
        if "norm" in key and "bias" in key:
            w[key] = np.zeros_like(w[key], dtype=np.float32)

    diagnostics = {
        "train_acc": float(best_acc),
        "bias": float(bias),
        "std_w": float(std_w),
        "max_abs_s": float(max_abs_s),
        "scale": float(scale),
        "embed_norm_mean": float(np.mean(np.linalg.norm(embed, axis=1))),
        "embed_mean": float(np.mean(embed)),
        "embed_std": float(np.std(embed)),
    }
    return {"weights": w, "diagnostics": diagnostics}
