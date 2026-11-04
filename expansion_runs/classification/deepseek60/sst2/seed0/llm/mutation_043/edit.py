import numpy as np

def edit(w, f):
    # Extract feedback pool
    x = f["x"]          # [4000, 64] int64 token ids
    y = f["y"]          # [4000] int64 labels
    N, L = x.shape
    V = w["embed.weight"].shape[0]  # 2000

    # Build bag-of-words counts (excluding padding id 0)
    # Use binary presence or counts? Counts often work well for linear classifiers.
    # We'll use counts, but also consider length normalization implicitly via mean pooling.
    # Actually the model averages token embeddings, so the feature is mean of token vectors.
    # If we set token vector first coordinate to w_t, the pooled feature is mean(w_t) over non-pad tokens.
    # So the classifier is on the average token weight, which is equivalent to a linear classifier on counts / length.
    # To match this, we should fit a linear classifier on length-normalized counts (i.e., mean of token weights).
    # We'll fit on the average token weight directly.

    # Compute document length (non-pad tokens)
    mask = (x != 0)
    lengths = mask.sum(axis=1).astype(np.float32)
    lengths[lengths == 0] = 1.0  # avoid division by zero

    # Build feature matrix: for each document, the average of one-hot token indicators.
    # This is equivalent to counts / length.
    # We can compute this efficiently using sparse-like accumulation.
    # Since V=2000, N=4000, we can build a dense matrix [N, V] but that's 8M floats, fine.
    X = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        # get non-pad tokens
        tokens = x[i][mask[i]]
        if len(tokens) > 0:
            # accumulate counts
            np.add.at(X[i], tokens, 1.0)
            X[i] /= len(tokens)  # length normalization

    # Now X is [N, V] with each row being the average token indicator.
    # We want to fit a linear classifier: score = X @ w + b, predicting y (0 or 1).
    # Use ridge regression on the labels (0/1) or logistic? Ridge is closed-form.
    # We'll use ridge regression on the labels centered to -1/+1 for better conditioning.
    y_signed = 2.0 * y.astype(np.float32) - 1.0  # -1 for class 0, +1 for class 1

    # Ridge regression: minimize ||X w + b - y_signed||^2 + lambda ||w||^2
    # Center X and y to avoid fitting bias separately.
    X_mean = X.mean(axis=0, keepdims=True)
    y_mean = y_signed.mean()
    Xc = X - X_mean
    yc = y_signed - y_mean

    # Choose lambda via simple cross-validation on a grid.
    # We'll do a quick 5-fold CV to pick lambda.
    best_lambda = 1.0
    best_acc = -1.0
    lambdas = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    # Precompute Xc^T Xc and Xc^T yc for efficiency? For CV we need to refit on folds.
    # Since N=4000, V=2000, we can do full CV. But to save time, we can use a single validation split.
    # Let's do 5-fold CV manually.
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, 5)
    for lam in lambdas:
        accs = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            X_train = Xc[train_idx]
            y_train = yc[train_idx]
            X_val = Xc[val_idx]
            y_val = y_signed[val_idx]
            # Solve (X^T X + lam I) w = X^T y
            # Use dual form if V > N? Here V=2000, N=3200, so primal is fine.
            # But X is [3200, 2000], X^T X is 2000x2000, inversion is O(V^3) ~ 8e9, too slow.
            # Use dual form: w = X^T (X X^T + lam I)^-1 y. X X^T is [3200, 3200], still large.
            # Better: use iterative? No, we can use the fact that X is sparse? But we built dense.
            # Actually we can use the normal equations with Cholesky, but 2000x2000 is okay for a few lambdas.
            # 2000x2000 matrix inversion is ~ 2000^3 = 8e9 flops, might take a few seconds per lambda. 9 lambdas * 5 folds = 45 inversions, too slow.
            # We can instead use a single train/val split to pick lambda, then fit on all data.
            # Let's do a single 80/20 split.
            pass
        # We'll do a simpler approach: use a fixed lambda chosen by a quick heuristic.
        # For bag-of-words with 4000 examples and 2000 features, lambda ~ 1.0 is reasonable.
        # We can also use the closed-form solution on all data with a small lambda.
        # Let's just use lambda = 1.0.
        break

    # Use a single train/val split to pick lambda
    np.random.seed(42)
    perm = np.random.permutation(N)
    train_idx = perm[:3200]
    val_idx = perm[3200:]
    X_train = Xc[train_idx]
    y_train = yc[train_idx]
    X_val = Xc[val_idx]
    y_val = y_signed[val_idx]

    best_lam = 1.0
    best_acc = -1.0
    for lam in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
        # Solve (X^T X + lam I) w = X^T y
        # Use Cholesky or solve. V=2000, so X^T X is 2000x2000.
        # To speed up, we can use the dual form: w = X^T (X X^T + lam I)^-1 y.
        # X X^T is 3200x3200, which is larger. Primal is better.
        # 2000x2000 solve is fine.
        XtX = X_train.T @ X_train
        Xty = X_train.T @ y_train
        # Add regularization
        XtX_reg = XtX + lam * np.eye(V, dtype=np.float32)
        try:
            w_tmp = np.linalg.solve(XtX_reg, Xty)
        except np.linalg.LinAlgError:
            w_tmp = np.linalg.lstsq(XtX_reg, Xty, rcond=None)[0]
        # Predict on val
        scores = X_val @ w_tmp
        preds = (scores > 0).astype(np.int64)
        # y_val is -1/+1, so preds 1 for >0, 0 for <0. Map to 0/1.
        preds = (scores > 0).astype(np.int64)  # 1 if positive
        # y_val is signed, so true label is 1 if y_val > 0 else 0
        true_labels = (y_val > 0).astype(np.int64)
        acc = (preds == true_labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_lam = lam

    # Now fit on all data with best_lam
    XtX = Xc.T @ Xc
    Xty = Xc.T @ yc
    XtX_reg = XtX + best_lam * np.eye(V, dtype=np.float32)
    w_token = np.linalg.solve(XtX_reg, Xty)
    # w_token is the weight for each token's average indicator.
    # The intercept is y_mean - X_mean @ w_token
    b = y_mean - X_mean @ w_token

    # Now we need to embed this into the network.
    # The network computes: pooled = mean(norm(embed(ids))). norm is LayerNorm.
    # We want the first coordinate of pooled to be proportional to X @ w_token + b.
    # If we set embed.weight[t] = w_token[t] * d + c_t * 1, where d is a direction,
    # and we enforce zero mean and unit norm for each token vector,
    # then norm(embed(t)) = embed(t) * sqrt(32) (since var = 1/32).
    # So pooled = sqrt(32) * mean(embed(t)) = sqrt(32) * (mean(w_token) * d + mean(c_t) * 1).
    # The first coordinate of pooled is sqrt(32) * (mean(w_token) * d[0] + mean(c_t) * 1[0]).
    # We want this to be proportional to mean(w_token) (since X is mean of indicators, X @ w_token = mean(w_token) over tokens).
    # Actually X[i] @ w_token = sum_{t in doc} w_token[t] / length = mean(w_token).
    # So we want pooled[0] = alpha * mean(w_token) + beta.
    # We can achieve this by setting d[0] = 1, and ensuring mean(c_t) is constant (e.g., 0).
    # If we set c_t such that sum(embed(t)) = 0 for each token, then mean(c_t) over tokens is not necessarily 0.
    # But we can just set the head to read pooled[0] and pooled[1] to cancel the constant.
    # Simpler: we can set embed.weight[t] to be a vector with first coordinate = w_token[t],
    # and the remaining coordinates chosen so that the vector is zero-mean and unit-norm.
    # Then norm(embed(t)) = embed(t) * sqrt(32).
    # pooled = sqrt(32) * mean(embed(t)).
    # The first coordinate of pooled is sqrt(32) * mean(w_token).
    # The other coordinates of pooled will be sqrt(32) * mean(rest).
    # Since each embed(t) is zero-mean, the sum of all coordinates is 0.
    # So mean(rest) = -mean(w_token) / 31 (if we set the rest equal).
    # Then we can set head.weight[1] to have weight on dim0 = -1.0, and weight on dim1..31 = something to cancel the rest.
    # Actually, we can just set head.weight[1] = [ -1.0, 0, 0, ... ] and head.bias = 0, and then logit1 = -pooled[0].
    # But pooled[0] = sqrt(32) * mean(w_token). So logit1 = -sqrt(32) * mean(w_token).
    # We want logit1 > 0 to predict class 1. Our w_token was fitted such that score = X @ w_token + b > 0 predicts class 1.
    # X @ w_token = mean(w_token). So we want logit1 = mean(w_token) + b > 0.
    # So we can set head.weight[1][0] = -1.0 / sqrt(32), and head.bias[1] = b.
    # Wait: logit1 = head.weight[1] @ pooled + head.bias[1].
    # If head.weight[1] = [ -1/sqrt(32), 0, ... ], then logit1 = -1/sqrt(32) * pooled[0] + b = -1/sqrt(32) * sqrt(32) * mean(w_token) + b = -mean(w_token) + b.
    # We want logit1 = mean(w_token) + b. So we need head.weight[1][0] = -1/sqrt(32) if we want -mean(w_token). But we want +mean(w_token).
    # Actually, our w_token was fitted such that score = X @ w_token + b > 0 predicts class 1.
    # So we want logit1 = X @ w_token + b. Since X @ w_token = mean(w_token), we want logit1 = mean(w_token) + b.
    # So set head.weight[1][0] = 1/sqrt(32), and head.bias[1] = b.
    # Then logit1 = 1/sqrt(32) * pooled[0] + b = mean(w_token) + b. Perfect.
    # For class 0, we can set head.weight[0] = 0, head.bias[0] = 0. Then logit0 = 0.
    # Decision: predict 1 if logit1 > logit0 => mean(w_token) + b > 0.

    # Now construct embed.weight.
    # We need each token vector to be zero-mean and unit-norm.
    # Let v_t be the vector for token t. We want v_t[0] = w_token[t] (or a scaled version).
    # To make it zero-mean and unit-norm, we can set the remaining 31 coordinates to be equal to some value c_t.
    # Then sum(v_t) = w_token[t] + 31 * c_t = 0 => c_t = -w_token[t] / 31.
    # Then norm(v_t)^2 = w_token[t]^2 + 31 * (w_token[t]/31)^2 = w_token[t]^2 * (1 + 1/31) = w_token[t]^2 * 32/31.
    # We want norm = 1, so we need to scale w_token[t] by a factor.
    # Let v_t[0] = a * w_token[t], and v_t[1..31] = -a * w_token[t] / 31.
    # Then norm^2 = a^2 * w_token[t]^2 * (1 + 1/31) = a^2 * w_token[t]^2 * 32/31.
    # Set norm = 1 => a = sqrt(31/32) / |w_token[t]|. But this depends on t, which would make the scaling non-uniform.
    # We want the first coordinate of pooled to be proportional to mean(w_token).
    # If we scale each token differently, the mean will be a weighted sum, not simply mean(w_token).
    # To keep it linear, we need the same scaling for all tokens. But then tokens with different |w_token| will have different norms.
    # LayerNorm will normalize each token individually, so the scaling will be per-token anyway.
    # Actually, LayerNorm normalizes each token vector to zero mean and unit variance.
    # If we set v_t = w_token[t] * d + c_t * 1, where d is a fixed direction, and we choose c_t to make the mean zero,
    # then the variance of v_t is w_token[t]^2 * var(d) + ... It depends on w_token[t].
    # So LayerNorm will scale each token by 1/sqrt(var(v_t)). This is non-linear.
    # To avoid this, we must ensure that all token vectors have the same variance before LayerNorm.
    # That means we need to set v_t such that its variance is constant, regardless of w_token[t].
    # One way: set v_t = w_token[t] * d + c * 1, where c is a global constant, and then we don't enforce zero mean per token.
    # But LayerNorm subtracts the mean of each token vector. The mean of v_t is w_token[t] * mean(d) + c.
    # The variance is w_token[t]^2 * var(d). So LayerNorm scales by 1/(|w_token[t]| * sqrt(var(d))).
    # Then the normalized vector is (w_token[t] * d + c - (w_token[t]*mean(d)+c)) / (|w_token[t]| * sqrt(var(d))) = sign(w_token[t]) * (d - mean(d)) / sqrt(var(d)).
    # This loses the magnitude of w_token[t]! So it becomes just the sign of w_token[t].
    # That's why the earlier attempts failed: they didn't account for the fact that LayerNorm removes the magnitude of the token weight.
    # To preserve magnitude, we must ensure that the variance of v_t does not depend on w_token[t].
    # That means we need to add a component that scales with w_token[t] in a way that keeps the variance constant.
    # Actually, we can set v_t = w_token[t] * d + c_t * 1, and we want var(v_t) to be constant.
    # var(v_t) = w_token[t]^2 * var(d) + var(c_t) + 2 * w_token[t] * cov(d, c_t).
    # If we set c_t to be a function of w_token[t], we can cancel the w_token[t]^2 term.
    # For example, set c_t = -w_token[t] * mean(d) (so mean is zero). Then var(v_t) = w_token[t]^2 * var(d).
    # Still depends on w_token[t]^2. To make it constant, we need to add a component that grows with w_token[t]^2.
    # But we only have 32 dimensions. We can use a second direction e, and set v_t = w_token[t] * d + w_token[t]^2 * e + c_t * 1.
    # Then var(v_t) = w_token[t]^2 * var(d) + w_token[t]^4 * var(e) + 2 * w_token[t]^3 * cov(d,e) + ...
    # This is getting complicated.
    # Alternative: We can bypass the LayerNorm issue by setting the token embeddings to be already normalized.
    # That is, we set v_t such that it is zero-mean and unit-norm. Then LayerNorm just scales by sqrt(32).
    # So the pooled feature is sqrt(32) * mean(v_t). We want mean(v_t) to be proportional to mean(w_token).
    # So we need v_t to be a linear function of w_token[t] plus a constant, but with the constraint that each v_t is zero-mean and unit-norm.
    # This is impossible for all w_token[t] because the norm constraint is non-linear.
    # However, we can choose a different approach: instead of using the mean of token embeddings, we can use the attention mechanism!
    # The attention blocks are currently identity because qkv/proj are zero. We can set them to non-zero to implement a more powerful model.
    # But the problem says "Keep the target architecture fixed: prediction uses only its resulting weight tensors." We can edit any weights.
    # We can set qkv, proj, ff1, ff2 to implement a non-linear function. But we are not allowed to use backprop or iterative training.
    # We can use closed-form fitting on the activations? The activations are given for the current weights. If we change earlier weights, the activations become stale.
    # We could potentially fit a linear classifier on the existing features, but as we saw, they are nearly 1-D.
    # The best we can do without changing the feature extractor is to refit the head on the existing features.
    # But the existing features are already quite good (0.919 accuracy). Can we improve by just refitting the head on feature0?
    # Let's check the current head: it uses feature0 with weight -1.0161 and bias 0.0025. That's a threshold on feature0.
    # We can find the optimal threshold on feature0 using the 4000 examples. That might give a small improvement.
    # But we can also change embed.weight to a better 1-D representation. The current embed.weight was presumably set by some heuristic.
    # We can fit a new 1-D token weight w_token using ridge regression on the bag-of-words, but we must ensure that the forward pass produces mean(w_token) as feature0.
    # To do that, we need to set embed.weight such that after LayerNorm, the first coordinate is exactly w_token[t] (up to a constant).
    # As argued, if we set v_t = w_token[t] * d + c_t * 1, and we enforce zero mean and unit norm, then LayerNorm scales by sqrt(32).
    # But the zero mean and unit norm constraints force c_t and the scaling to depend on w_token[t].
    # However, we can choose d to be a direction that is orthogonal to the all-ones vector, and set c_t = 0.
    # Then v_t = w_token[t] * d. But then v_t is not zero-mean unless mean(d) = 0. If we choose d with mean 0, then v_t is zero-mean.
    # The norm of v_t is |w_token[t]| * ||d||. To make it unit-norm, we would need to scale by 1/|w_token[t]|, which again depends on t.
    # So we cannot have both zero-mean and unit-norm with a simple linear scaling.
    # What if we don't enforce unit-norm? LayerNorm will normalize anyway. The issue is that the normalization factor depends on |w_token[t]|.
    # So the normalized vector becomes sign(w_token[t]) * d / ||d||. The magnitude is lost.
    # To preserve magnitude, we need the variance of v_t to be independent of w_token[t].
    # We can achieve this by adding a constant vector to v_t that dominates the variance.
    # For example, set v_t = w_token[t] * d + C * 1, where C is a large constant.
    # Then var(v_t) ≈ C^2 * var(1) = 0 (since 1 is constant). Actually var of a constant vector is 0.
    # We need a vector with non-zero variance. Let's set v_t = w_token[t] * d + C * e, where e is a fixed vector with non-zero variance.
    # Then var(v_t) = w_token[t]^2 * var(d) + C^2 * var(e) + 2 * w_token[t] * C * cov(d,e).
    # If we choose C large enough, the C^2 term dominates, so var(v_t) ≈ C^2 * var(e), which is constant.
    # Then LayerNorm scales by 1/(C * sqrt(var(e))), which is constant.
    # The normalized vector is (w_token[t] * d + C * e - mean) / (C * sqrt(var(e))) ≈ (w_token[t]/C) * (d - mean(d))/sqrt(var(e)) + (e - mean(e))/sqrt(var(e)).
    # So the first coordinate of the normalized vector is proportional to w_token[t] plus a constant.
    # This works! The key is to make the constant component C * e large compared to the varying component w_token[t] * d.
    # Then the variance is dominated by C, so the scaling is uniform.
    # We can choose C to be large, say C = 10 * max(|w_token|). Then the non-linearity is small.
    # Let's design this.

    # We have w_token from ridge regression. Let's compute its scale.
    w_scale = np.max(np.abs(w_token))
    if w_scale == 0:
        w_scale = 1.0
    C = 10.0 * w_scale  # large constant

    # We need a fixed vector e with non-zero variance. Let's use e = [0, 1, 1, ..., 1]? But we need to control the first coordinate.
    # We want the first coordinate of the normalized vector to be proportional to w_token[t].
    # Let's set d = [1, 0, 0, ..., 0] (so first coordinate is w_token[t]).
    # Let e = [0, 1, 1, ..., 1] (so it adds a constant to other coordinates, but not the first).
    # Then v_t = w_token[t] * d + C * e.
    # v_t[0] = w_token[t]
    # v_t[1..31] = C
    # The mean of v_t is (w_token[t] + 31*C) / 32.
    # The variance is ( (w_token[t] - mean)^2 + 31*(C - mean)^2 ) / 32.
    # Since C is large, mean ≈ C, so variance ≈ ( (w_token[t] - C)^2 + 31*(C - C)^2 ) / 32 ≈ (w_token[t] - C)^2 / 32.
    # Wait, if C is large, mean ≈ C, so w_token[t] - mean ≈ w_token[t] - C ≈ -C. So variance ≈ C^2 / 32, which is constant.
    # Then the normalized vector: (v_t - mean) / sqrt(var) = (v_t - mean) / (C / sqrt(32)) = sqrt(32) * (v_t - mean) / C.
    # v_t - mean = [w_token[t] - mean, C - mean, ..., C - mean].
    # mean = (w_token[t] + 31*C) / 32.
    # w_token[t] - mean = w_token[t] - (w_token[t] + 31*C)/32 = (31*w_token[t] - 31*C)/32 = 31/32 * (w_token[t] - C).
    # C - mean = C - (w_token[t] + 31*C)/32 = (32*C - w_token[t] - 31*C)/32 = (C - w_token[t])/32.
    # So the first coordinate of the normalized vector is sqrt(32) * (31/32 * (w_token[t] - C)) / C ≈ sqrt(32) * 31/32 * (-1) = -31/sqrt(32) ≈ -5.48.
    # That's almost constant! The variation in w_token[t] is scaled by 31/32 / C, which is very small because C is large.
    # So the first coordinate becomes almost constant, losing the signal.
    # We need the varying component to be in the first coordinate, and the constant component to be in other coordinates, but we need the variance to be dominated by the constant component.
    # If we put the constant in other coordinates, the mean will be dominated by the constant, and the first coordinate will be w_token[t] - mean ≈ -C, which is constant.
    # So the first coordinate becomes constant. That's bad.
    # To preserve the signal in the first coordinate, we need the mean to be small, or we need to subtract the mean in a way that doesn't cancel the signal.
    # Actually, LayerNorm subtracts the mean of each token vector. If we want the first coordinate of the normalized vector to be proportional to w_token[t], we need the mean of the first coordinate to be zero or constant.
    # If we set v_t[0] = w_token[t], and v_t[1..31] = c_t, and we choose c_t such that the mean of v_t is zero, then v_t[0] - mean = w_token[t] - 0 = w_token[t] (if mean is 0).
    # But then the variance is (w_token[t]^2 + 31*c_t^2)/32. If c_t is chosen to make mean zero, c_t = -w_token[t]/31. Then variance = w_token[t]^2 * (1 + 1/31)/32 = w_token[t]^2 * 32/31 /32 = w_token[t]^2 /31.
    # So variance depends on w_token[t]^2. LayerNorm scales by 1/|w_token[t]|, so the normalized first coordinate becomes w_token[t] / |w_token[t]| = sign(w_token[t]). Magnitude lost.
    # To keep magnitude, we need the variance to be constant. So we need to add a component that doesn't affect the mean but adds constant variance.
    # For example, set v_t = w_token[t] * d + c_t * 1 + C * e, where e is a vector with zero mean and unit norm, and C is a constant.
    # Then the mean of v_t is w_token[t]*mean(d) + c_t + C*mean(e). If we set mean(d)=0, mean(e)=0, and c_t=0, then mean=0.
    # Then variance = w_token[t]^2 * var(d) + C^2 * var(e) + 2*w_token[t]*C*cov(d,e).
    # If we choose d and e orthogonal, cov=0. Then variance = w_token[t]^2 * var(d) + C^2 * var(e).
    # If C is large, variance ≈ C^2 * var(e), which is constant.
    # Then LayerNorm scales by 1/(C * sqrt(var(e))), which is constant.
    # The normalized vector is (w_token[t] * d + C * e) / (C * sqrt(var(e))) = (w_token[t]/C) * d/sqrt(var(e)) + e/sqrt(var(e)).
    # So the first coordinate (if d[0]=1) is w_token[t]/C * 1/sqrt(var(e)) + e[0]/sqrt(var(e)).
    # Since C is large, the w_token[t] term is small, but it's proportional to w_token[t]. The constant term is e[0]/sqrt(var(e)).
    # We can set e[0] = 0 to avoid a constant offset in the first coordinate. Then the first coordinate is exactly w_token[t]/(C * sqrt(var(e))).
    # So we can recover the signal by setting the head weight on the first coordinate to be C * sqrt(var(e)) / sqrt(32) (to cancel the scaling).
    # Let's design this carefully.

    # Let d be a vector with d[0] = 1, and d[1..31] = 0. So d = [1, 0, ..., 0].
    # Let e be a vector with e[0] = 0, and e[1..31] = 1. So e = [0, 1, 1, ..., 1].
    # We want mean(d) = 1/32, not zero. We can center d and e to have zero mean.
    # Let d_centered = d - mean(d) * 1. mean(d) = 1/32. So d_centered[0] = 1 - 1/32 = 31/32. d_centered[1..31] = -1/32.
    # Let e_centered = e - mean(e) * 1. mean(e) = 31/32. So e_centered[0] = -31/32. e_centered[1..31] = 1 - 31/32 = 1/32.
    # Now d_centered and e_centered both have zero mean.
    # Check orthogonality: dot(d_centered, e_centered) = (31/32)*(-31/32) + 31 * (-1/32)*(1/32) = -961/1024 - 31/1024 = -992/1024 = -31/32. Not orthogonal.
    # We can orthogonalize them. But we don't need them orthogonal if we just want to control variance.
    # Actually, we can just use a single vector d_centered for the signal, and add a constant vector C * e_centered for variance.
    # But we need to ensure the variance is dominated by C. If they are not orthogonal, the cross term might matter.
    # Let's just use a simpler approach: set v_t = w_token[t] * d + C * e, where d and e are fixed vectors, and we don't enforce zero mean per token.
    # LayerNorm will subtract the mean of v_t. The mean of v_t is w_token[t]*mean(d) + C*mean(e).
    # The variance is w_token[t]^2 * var(d) + C^2 * var(e) + 2*w_token[t]*C*cov(d,e).
    # If C is large, variance ≈ C^2 * var(e). So the scaling factor is approximately 1/(C * sqrt(var(e))).
    # The normalized vector is (v_t - mean) / (C * sqrt(var(e))) = (w_token[t] * (d - mean(d)) + C * (e - mean(e))) / (C * sqrt(var(e))) = (w_token[t]/C) * (d - mean(d))/sqrt(var(e)) + (e - mean(e))/sqrt(var(e)).
    # The first coordinate of the normalized vector is (w_token[t]/C) * (d[0] - mean(d))/sqrt(var(e)) + (e[0] - mean(e))/sqrt(var(e)).
    # We can choose d and e such that e[0] = mean(e) to eliminate the constant term in the first coordinate.
    # For example, let e be a vector with all entries equal to 1. Then mean(e)=1, e[0]=1, so e[0]-mean(e)=0. Then the first coordinate is purely proportional to w_token[t].
    # Let's test this: e = [1, 1, ..., 1]. Then var(e) = 0. That's bad because C^2 * var(e) = 0, so variance is not dominated by C.
    # We need e to have non-zero variance. So e cannot be constant.
    # Let e = [0, 1, 1, ..., 1]. mean(e) = 31/32. e[0] = 0. So e[0] - mean(e) = -31/32.
    # Then the first coordinate has a constant term: (e[0]-mean(e))/sqrt(var(e)) = (-31/32)/sqrt(var(e)).
    # We can cancel this constant term by adjusting the head bias. That's fine.
    # The important thing is that the varying part is proportional to w_token[t].
    # So we can set v_t = w_token[t] * d + C * e, where d = [1, 0, ..., 0], e = [0, 1, ..., 1].
    # Then the first coordinate of the normalized vector is approximately (w_token[t]/C) * (1 - 1/32)/sqrt(var(e)) + constant.
    # So it's linear in w_token[t].
    # Let's compute the exact normalized vector for this choice.
    # v_t = [w_token[t], C, C, ..., C] (32 dims: first is w_token[t], rest 31 are C).
    # mean = (w_token[t] + 31*C) / 32.
    # var = ( (w_token[t] - mean)^2 + 31*(C - mean)^2 ) / 32.
    # Let's compute the normalized first coordinate: (v_t[0] - mean) / sqrt(var).
    # v_t[0] - mean = w_token[t] - (w_token[t] + 31*C)/32 = (31*w_token[t] - 31*C)/32 = 31/32 * (w_token[t] - C).
    # C - mean = C - (w_token[t] + 31*C)/32 = (32*C - w_token[t] - 31*C)/32 = (C - w_token[t])/32.
    # So var = ( (31/32)^2 * (w_token[t] - C)^2 + 31 * (1/32)^2 * (C - w_token[t])^2 ) / 32
    #        = (w_token[t] - C)^2 / 32 * ( (31/32)^2 + 31*(1/32)^2 )
    #        = (w_token[t] - C)^2 / 32 * ( 961/1024 + 31/1024 ) = (w_token[t] - C)^2 / 32 * (992/1024) = (w_token[t] - C)^2 / 32 * (31/32).
    # So sqrt(var) = |w_token[t] - C| / sqrt(32) * sqrt(31/32) = |w_token[t] - C| * sqrt(31) / 32.
    # Then normalized first coordinate = (31/32 * (w_token[t] - C)) / (|w_token[t] - C| * sqrt(31) / 32) = 31 * (w_token[t] - C) / (|w_token[t] - C| * sqrt(31)) = sqrt(31) * sign(w_token[t] - C).
    # This is just the sign! The magnitude is lost because the variance is proportional to (w_token[t] - C)^2.
    # So this choice fails. The problem is that the mean is dominated by C, so w_token[t] - mean ≈ -C, and the variance is also dominated by C^2, so the scaling cancels the magnitude.
    # To preserve magnitude, we need the variance to be constant, not proportional to (w_token[t] - C)^2.
    # That means we need to add a component that is independent of w_token[t] and has large variance.
    # But if we add a constant vector C * e, its variance is C^2 * var(e), which is constant. However, the mean also becomes C * mean(e), which is large.
    # Then v_t[0] - mean = w_token[t] - C * mean(e). If mean(e) is large, this is dominated by -C * mean(e), which is constant.
    # So the varying part w_token[t] is small compared to the constant offset. Then the normalized first coordinate is (w_token[t] - C*mean(e)) / (C * sqrt(var(e))) ≈ -mean(e)/sqrt(var(e)) + w_token[t]/(C * sqrt(var(e))).
    # The varying part is w_token[t]/(C * sqrt(var(e))), which is linear in w_token[t]! The constant part can be absorbed by the head bias.
    # So this actually works! The magnitude is preserved, just scaled by 1/C. We can compensate by scaling the head weight.
    # Let's re-evaluate: v_t = w_token[t] * d + C * e, where d = [1, 0, ..., 0], e = [0, 1, 1, ..., 1].
    # mean = (w_token[t] + 31*C) / 32.
    # v_t[0] - mean = w_token[t] - (w_token[t] + 31*C)/32 = (31*w_token[t] - 31*C)/32 = 31/32 * (w_token[t] - C).
    # Wait, this is not dominated by C*mean(e). mean(e) = 31/32. C*mean(e) = 31C/32. So v_t[0] - mean = w_token[t] - 31C/32 - w_token[t]/32 = (31/32)w_token[t] - 31C/32 = 31/32 (w_token[t] - C).
    # So it's proportional to (w_token[t] - C). The varying part is w_token[t], but there's a constant -C. So the signal is w_token[t] - C.
    # The variance: var = ( (31/32)^2 (w_token[t] - C)^2 + 31*(1/32)^2 (C - w_token[t])^2 ) / 32 = (w_token[t] - C)^2 / 32 * (31/32).
    # So sqrt(var) = |w_token[t] - C| * sqrt(31)/32.
    # Then normalized first coordinate = (31/32 (w_token[t] - C)) / (|w_token[t] - C| sqrt(31)/32) = sqrt(31) * sign(w_token[t] - C).
    # This is exactly the sign, no magnitude! Why? Because the variance is proportional to (w_token[t] - C)^2, so the scaling cancels the magnitude.
    # The issue is that the mean is not dominated by C*mean(e) in a way that leaves w_token[t] - C*mean(e) with a large constant part. Actually, mean = (w_token[t] + 31C)/32. So v_t[0] - mean = w_token[t] - (w_token[t] + 31C)/32 = (31/32)w_token[t] - (31/32)C = (31/32)(w_token[t] - C). So it's exactly proportional to (w_token[t] - C). The variance is also proportional to (w_token[t] - C)^2. So the normalization cancels the magnitude completely.
    # To avoid this, we need the variance to have a component that is independent of w_token[t]. That means we need to add a vector that has variance but whose mean is zero, so it doesn't affect the mean.
    # Let's add a vector C * e, where e has mean 0. Then mean of v_t = w_token[t] * mean(d) + C * 0 = w_token[t] * mean(d).
    # If we set d to have mean 0 as well, then mean = 0.
    # Then v_t[0] - mean = w_token[t] * d[0] + C * e[0].
    # Variance = w_token[t]^2 * var(d) + C^2 * var(e) + 2*w_token[t]*C*cov(d,e).
    # If C is large, variance ≈ C^2 * var(e), which is constant.
    # Then normalized first coordinate = (w_token[t] * d[0] + C * e[0]) / (C * sqrt(var(e))) = (w_token[t]/C) * d[0]/sqrt(var(e)) + e[0]/sqrt(var(e)).
    # This is linear in w_token[t]! The constant term e[0]/sqrt(var(e)) can be absorbed by the head bias.
    # So we need d and e to have zero mean. Let's construct them.
    # We want d[0] = 1, and mean(d) = 0. So d = [1, -1/31, -1/31, ..., -1/31] (32 dims: first 1, rest -1/31).
    # Check mean: (1 + 31*(-1/31))/32 = (1 - 1)/32 = 0. Good.
    # We want e to have zero mean, and e[0] = 0 (to avoid constant term in first coordinate, but we can allow it). Let's set e = [0, 1, -1/30, -1/30, ..., -1/30]? Wait, we need mean(e)=0. If e[0]=0, and we have 31 other entries, we can set them to 1 and -1/30? Actually, if we have 31 entries, we can set 30 of them to 1 and one to -30 to make mean 0. But simpler: let e be a vector with zero mean and non-zero variance. We can just use e = [0, 1, 1, ..., 1] and then subtract its mean. But we need to do this in code.
    # Let's just define d and e as zero-mean vectors.
    # d = np.zeros(32); d[0] = 1.0; d = d - d.mean()
    # e = np.zeros(32); e[1:] = 1.0; e = e - e.mean()
    # Then v_t = w_token[t] * d + C * e.
    # Then mean(v_t) = w_token[t]*0 + C*0 = 0.
    # Variance = w_token[t]^2 * var(d) + C^2 * var(e) + 2*w_token[t]*C*cov(d,e).
    # We can orthogonalize d and e to remove cov. Let's do that.
    # d = d - (d @ e) / (e @ e) * e. Then d and e are orthogonal.
    # Then var(v_t) = w_token[t]^2 * var(d) + C^2 * var(e).
    # If C is large, var(v_t) ≈ C^2 * var(e).
    # Then normalized vector = (v_t - 0) / sqrt(var(v_t)) ≈ (w_token[t] * d + C * e) / (C * sqrt(var(e))) = (w_token[t]/C) * d/sqrt(var(e)) + e/sqrt(var(e)).
    # The first coordinate is (w_token[t]/C) * d[0]/sqrt(var(e)) + e[0]/sqrt(var(e)).
    # So it's linear in w_token[t] with slope d[0]/(C * sqrt(var(e))).
    # We can set the head weight on the first coordinate to be C * sqrt(var(e)) / (sqrt(32) * d[0]) to get the desired coefficient.
    # Wait, the final pooled feature is sqrt(32) * mean(normalized vector). Since mean is over tokens, and normalized vector is as above, the first coordinate of pooled is sqrt(32) * ( (mean(w_token)/C) * d[0]/sqrt(var(e)) + e[0]/sqrt(var(e)) ).
    # We want this to be proportional to mean(w_token). So we can set head.weight[1][0] = 1.0, and then logit1 = pooled[0] + bias.
    # But pooled[0] = sqrt(32) * (mean(w_token)/C * d[0]/sqrt(var(e)) + e[0]/sqrt(var(e))).
    # So logit1 = sqrt(32) * d[0]/(C * sqrt(var(e))) * mean(w_token) + constant.
    # We want logit1 = mean(w_token) + b. So we need to scale the head weight appropriately.
    # Actually, we can just set head.weight[1][0] = 1.0, and then the effective coefficient on mean(w_token) is sqrt(32) * d[0]/(C * sqrt(var(e))).
    # We can absorb this into w_token by scaling w_token. But we already fitted w_token. We can just scale the head weight to match.
    # Let's compute the exact scaling factor and set head.weight[1][0] = 1.0 / (sqrt(32) * d[0]/(C * sqrt(var(e)))) = C * sqrt(var(e)) / (sqrt(32) * d[0]).
    # Then logit1 = mean(w_token) + constant. Then set head.bias[1] = b - constant.
    # This is doable.

    # Let's implement this construction.
    # First, compute w_token and b from ridge regression on all data.
    # We already have w_token and b from the CV step? We need to fit on all data with best_lam.
    # Let's do that now.

    # Recompute X and y_signed (we already have X and y_signed from earlier? We need to keep them.)
    # We'll just recompute everything cleanly.

    # Build X: [N, V] average token indicators.
    X = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        tokens = x[i][mask[i]]
        if len(tokens) > 0:
            np.add.at(X[i], tokens, 1.0)
            X[i] /= len(tokens)

    y_signed = 2.0 * y.astype(np.float32) - 1.0
    X_mean = X.mean(axis=0, keepdims=True)
    y_mean = y_signed.mean()
    Xc = X - X_mean
    yc = y_signed - y_mean

    # Use a fixed lambda based on CV. We'll do a quick CV to pick lambda.
    # To save time, we can use a single validation split.
    np.random.seed(42)
    perm = np.random.permutation(N)
    train_idx = perm[:3200]
    val_idx = perm[3200:]
    X_train = Xc[train_idx]
    y_train = yc[train_idx]
    X_val = Xc[val_idx]
    y_val = y_signed[val_idx]

    best_lam = 1.0
    best_acc = -1.0
    for lam in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
        XtX = X_train.T @ X_train
        Xty = X_train.T @ y_train
        XtX_reg = XtX + lam * np.eye(V, dtype=np.float32)
        try:
            w_tmp = np.linalg.solve(XtX_reg, Xty)
        except np.linalg.LinAlgError:
            w_tmp = np.linalg.lstsq(XtX_reg, Xty, rcond=None)[0]
        scores = X_val @ w_tmp
        preds = (scores > 0).astype(np.int64)
        true_labels = (y_val > 0).astype(np.int64)
        acc = (preds == true_labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_lam = lam

    # Fit on all data
    XtX = Xc.T @ Xc
    Xty = Xc.T @ yc
    XtX_reg = XtX + best_lam * np.eye(V, dtype=np.float32)
    w_token = np.linalg.solve(XtX_reg, Xty)
    b = y_mean - X_mean @ w_token

    # Now construct embed.weight.
    # We want v_t = w_token[t] * d + C * e, where d and e are zero-mean, orthogonal, and e has non-zero variance.
    # Let's choose d and e.
    d = np.zeros(32, dtype=np.float32)
    d[0] = 1.0
    d = d - d.mean()

    e = np.zeros(32, dtype=np.float32)
    e[1:] = 1.0
    e = e - e.mean()

    # Orthogonalize d with respect to e
    d = d - (d @ e) / (e @ e) * e

    # Now d and e are orthogonal and zero-mean.
    var_d = (d @ d) / 32.0
    var_e = (e @ e) / 32.0

    # Choose C large enough to dominate variance.
    w_scale = np.max(np.abs(w_token))
    if w_scale == 0:
        w_scale = 1.0
    C = 10.0 * w_scale

    # Construct embed.weight: for each token t, v_t = w_token[t] * d + C * e.
    # Shape: [V, 32]
    embed_new = np.zeros((V, 32), dtype=np.float32)
    for t in range(V):
        embed_new[t] = w_token[t] * d + C * e

    # Now compute the scaling factor for the head.
    # The normalized vector for token t is approximately (w_token[t] * d + C * e) / (C * sqrt(var_e)) because var(v_t) ≈ C^2 * var_e.
    # Actually, let's compute the exact normalization for a given w_token[t].
    # But since C is large, the approximation is good. We can just use the exact formula if we want, but it's per-token.
    # However, the head is linear, so we can just set the head weight to read the first coordinate.
    # The first coordinate of the normalized vector is (w_token[t] * d[0] + C * e[0]) / sqrt(var(v_t)).
    # var(v_t) = w_token[t]^2 * var_d + C^2 * var_e (since orthogonal).
    # So normalized first coord = (w_token[t] * d[0] + C * e[0]) / sqrt(w_token[t]^2 * var_d + C^2 * var_e).
    # For large C, this is approximately (w_token[t] * d[0] + C * e[0]) / (C * sqrt(var_e)) = (w_token[t]/C) * d[0]/sqrt(var_e) + e[0]/sqrt(var_e).
    # So it's linear in w_token[t] with slope d[0]/(C * sqrt(var_e)).
    # The pooled feature first coordinate is sqrt(32) * mean(normalized first coord).
    # So pooled[0] = sqrt(32) * ( mean(w_token)/C * d[0]/sqrt(var_e) + e[0]/sqrt(var_e) ).
    # We want logit1 = mean(w_token) + b.
    # So we set head.weight[1][0] = 1.0 / (sqrt(32) * d[0]/(C * sqrt(var_e))) = C * sqrt(var_e) / (sqrt(32) * d[0]).
    # And head.bias[1] = b - sqrt(32) * e[0]/sqrt(var_e) * head.weight[1][0].
    # Let's compute these.

    # But wait, the approximation ignores the w_token[t]^2 term in variance. For tokens with large |w_token|, the variance might be larger, so the normalization might differ.
    # However, since C = 10 * max|w_token|, the term w_token[t]^2 * var_d is at most (max|w|)^2 * var_d, while C^2 * var_e = 100 * (max|w|)^2 * var_e.
    # So the w_token^2 term is at most 1% of the C^2 term if var_d and var_e are comparable. So the approximation is very good.
    # We can also just use the exact normalization by computing the normalized vector for each token and then setting the head to read the first coordinate.
    # But the head is linear, so if the normalized first coordinate is not exactly linear in w_token, the linear head will only capture the linear part.
    # Since the non-linearity is small, it should be fine.

    # Let's compute the exact scaling factor by fitting a linear regression from w_token to the normalized first coordinate.
    # Actually, we can just compute the normalized first coordinate for all tokens, then fit a linear model to predict it from w_token.
    # But we don't need to; we can just set the head weight to the slope of the linear approximation.
    # Let's compute the exact normalized first coordinate for each token using the formula.
    # v_t = w_token[t] * d + C * e
    # mean_v = w_token[t] * d.mean() + C * e.mean() = 0 (since both zero-mean)
    # var_v = w_token[t]^2 * var_d + C^2 * var_e (since orthogonal)
    # norm_first = (w_token[t] * d[0] + C * e[0]) / sqrt(var_v + 1e-5)
    # We want to set head.weight[1][0] such that head.weight[1][0] * sqrt(32) * mean(norm_first) ≈ mean(w_token).
    # So head.weight[1][0] = 1.0 / (sqrt(32) * slope), where slope is the linear coefficient of norm_first vs w_token.
    # We can compute slope by linear regression: slope = cov(w_token, norm_first) / var(w_token).
    # Let's do that.

    # Compute norm_first for all tokens
    norm_first = np.zeros(V, dtype=np.float32)
    for t in range(V):
        var_v = w_token[t]**2 * var_d + C**2 * var_e
        norm_first[t] = (w_token[t] * d[0] + C * e[0]) / np.sqrt(var_v + 1e-5)

    # Fit linear model: norm_first ≈ slope * w_token + intercept
    # We can use least squares.
    A = np.vstack([w_token, np.ones(V)]).T
    slope, intercept = np.linalg.lstsq(A, norm_first, rcond=None)[0]

    # Now set head weight and bias.
    # We want logit1 = head.weight[1][0] * pooled[0] + head.bias[1]
    # pooled[0] = sqrt(32) * mean(norm_first) ≈ sqrt(32) * (slope * mean(w_token) + intercept)
    # We want logit1 = mean(w_token) + b.
    # So head.weight[1][0] * sqrt(32) * slope = 1  => head.weight[1][0] = 1.0 / (sqrt(32) * slope)
    # head.weight[1][0] * sqrt(32) * intercept + head.bias[1] = b => head.bias[1] = b - head.weight[1][0] * sqrt(32) * intercept

    head_w = np.zeros(32, dtype=np.float32)
    head_w[0] = 1.0 / (np.sqrt(32.0) * slope)
    head_b = b - head_w[0] * np.sqrt(32.0) * intercept

    # Set head.weight[1] = head_w, head.bias[1] = head_b
    # Set head.weight[0] = 0, head.bias[0] = 0 (so logit0 = 0)
    # Update w
    w["embed.weight"] = embed_new.astype(np.float32)
    w["head.weight"][0] = 0.0
    w["head.weight"][1] = head_w
    w["head.bias"][0] = 0.0
    w["head.bias"][1] = head_b

    # Also ensure all other weights remain as they are (they are already zero/identity).
    # We don't need to change them.

    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lam),
        "val_acc": float(best_acc),
        "w_token_norm": float(np.linalg.norm(w_token)),
        "C": float(C),
        "slope": float(slope),
        "intercept": float(intercept),
        "head_w0": float(head_w[0]),
        "head_b": float(head_b),
        "embed_norm_mean": float(np.mean(np.linalg.norm(embed_new, axis=1))),
        "embed_mean": float(np.mean(embed_new)),
    }

    return {"weights": w, "diagnostics": diagnostics}
