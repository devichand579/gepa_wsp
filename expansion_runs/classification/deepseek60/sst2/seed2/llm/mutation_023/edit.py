import numpy as np

def edit(w, f):
    # Extract feedback arrays
    x = f["x"]                     # (4000, 64) int64
    y = f["y"].astype(np.float32)  # (4000,) int64 -> float32
    N, L = x.shape
    V = w["embed.weight"].shape[0] # 2000
    D = w["embed.weight"].shape[1] # 32

    # Build BOW matrix (4000, 2000)
    BOW = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        np.add.at(BOW[i], x[i], 1.0)
    # Remove padding token 0 from BOW? The model uses ids!=0 for pooling, but embed[0] is zero.
    # BOW includes token 0, but embed[0] is zero, so it doesn't affect the pooled feature.
    # We can leave it.

    # Center BOW for PCA (optional, but good for PCA)
    BOW_mean = BOW.mean(axis=0, keepdims=True)
    BOW_centered = BOW - BOW_mean

    # 1. Ridge classifier on BOW
    # We want to predict y in {0,1}. Use targets t = 2*y - 1.
    t = 2.0 * y - 1.0  # (4000,)
    # Add intercept by augmenting BOW with a column of ones
    BOW_aug = np.concatenate([BOW, np.ones((N, 1), dtype=np.float32)], axis=1)
    # Ridge regression: w_ridge = (X^T X + lambda I)^-1 X^T t
    # Choose lambda by 5-fold CV on the full pool.
    # We'll do a simple CV over a small set of lambdas.
    lambdas = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_lambda = 0.1
    best_cv_acc = -1.0
    # Precompute X^T X and X^T t for the full data? For CV we need to split.
    # Since N=4000, V=2000, X^T X is 2001x2001, which is large but manageable.
    # We can do CV by computing the full X^T X and then subtracting the fold's contribution.
    # This is efficient.
    XtX_full = BOW_aug.T @ BOW_aug  # (2001, 2001)
    Xty_full = BOW_aug.T @ t        # (2001,)
    # 5-fold CV
    fold_size = N // 5
    indices = np.arange(N)
    # We'll use a fixed random permutation for reproducibility
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    for lam in lambdas:
        cv_acc = 0.0
        for fold in range(5):
            val_idx = indices[fold*fold_size:(fold+1)*fold_size]
            train_idx = np.setdiff1d(indices, val_idx)
            # Compute XtX_train and Xty_train by subtracting val contributions
            X_val = BOW_aug[val_idx]
            XtX_val = X_val.T @ X_val
            Xty_val = X_val.T @ t[val_idx]
            XtX_train = XtX_full - XtX_val
            Xty_train = Xty_full - Xty_val
            # Solve ridge
            reg = lam * np.eye(XtX_train.shape[0], dtype=np.float32)
            w_ridge = np.linalg.solve(XtX_train + reg, Xty_train)
            # Predict on val
            scores = X_val @ w_ridge
            preds = (scores > 0).astype(np.float32)
            acc = np.mean(preds == y[val_idx])
            cv_acc += acc
        cv_acc /= 5.0
        if cv_acc > best_cv_acc:
            best_cv_acc = cv_acc
            best_lambda = lam

    # Fit final ridge on full data
    reg = best_lambda * np.eye(XtX_full.shape[0], dtype=np.float32)
    w_ridge_full = np.linalg.solve(XtX_full + reg, Xty_full)
    # w_ridge_full has length V+1: first V are token weights, last is bias.
    s = w_ridge_full[:V].astype(np.float32)  # (2000,)
    b_ridge = w_ridge_full[V]

    # 2. PCA on BOW_centered
    # We want top 31 principal components. We can compute the covariance matrix (2000x2000) and its eigenvectors.
    # But 2000x2000 is large. Instead, we can use the fact that N=4000 < V=2000? Actually N=4000 > V=2000.
    # We can compute the SVD of BOW_centered (4000x2000). The right singular vectors are the principal components.
    # To save time, we can compute the covariance matrix: C = BOW_centered.T @ BOW_centered / N. (2000x2000)
    # Then compute top 31 eigenvectors. This is 2000x2000, which is 4M floats, manageable.
    # But we can also use the Gram matrix (4000x4000) which is larger. Let's use the covariance.
    # Actually, we only need the top 31 eigenvectors. We can use np.linalg.eigh on the covariance matrix.
    # To speed up, we can use the fact that BOW_centered is sparse? But it's dense here.
    # Let's just compute the covariance and use eigh. It might take a few seconds but is fine.
    # Alternatively, we can use randomized SVD, but that's iterative. We'll use eigh.
    # To avoid memory issues, we can compute C = BOW_centered.T @ BOW_centered / N.
    # This is 2000x2000.
    C = (BOW_centered.T @ BOW_centered) / N
    # Compute top 31 eigenvectors
    # eigh returns eigenvalues in ascending order.
    evals, evecs = np.linalg.eigh(C)
    # Take the top 31 (largest eigenvalues)
    top_indices = np.argsort(evals)[-31:][::-1]
    V_pca = evecs[:, top_indices].astype(np.float32)  # (2000, 31)

    # 3. Construct embed.weight
    # We want column 0 to be s, columns 1..31 to be V_pca.
    # But we need to ensure that the final LayerNorm doesn't destroy the information.
    # The final LayerNorm is applied per token. So we can just set embed.weight to be the concatenation.
    # However, the scale of s and V_pca might be very different. We should normalize each column to have similar scale.
    # We can standardize each column to have mean 0 and std 1 across tokens? Or we can just let LayerNorm handle it.
    # Actually, LayerNorm normalizes each token's vector to zero mean and unit variance. So the relative scale of columns matters.
    # If one column has much larger variance across tokens, it will dominate the token vector after LayerNorm? No, LayerNorm normalizes the vector, so it's the pattern across dimensions that matters.
    # To make the columns comparable, we can normalize each column to have zero mean and unit variance across tokens.
    # Let's do that.
    embed_new = np.zeros((V, D), dtype=np.float32)
    # Column 0: s
    col0 = s.copy()
    # Normalize col0 to zero mean, unit std
    col0 = col0 - col0.mean()
    std0 = col0.std()
    if std0 > 1e-8:
        col0 = col0 / std0
    embed_new[:, 0] = col0
    # Columns 1..31: V_pca
    for j in range(31):
        col = V_pca[:, j].copy()
        col = col - col.mean()
        std = col.std()
        if std > 1e-8:
            col = col / std
        embed_new[:, j+1] = col

    # Set the weights
    w["embed.weight"] = embed_new.astype(np.float32)
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)

    # Zero out all transformer blocks
    for block in ["block0", "block1"]:
        w[f"{block}.qkv.weight"] = np.zeros_like(w[f"{block}.qkv.weight"], dtype=np.float32)
        w[f"{block}.qkv.bias"] = np.zeros_like(w[f"{block}.qkv.bias"], dtype=np.float32)
        w[f"{block}.proj.weight"] = np.zeros_like(w[f"{block}.proj.weight"], dtype=np.float32)
        w[f"{block}.proj.bias"] = np.zeros_like(w[f"{block}.proj.bias"], dtype=np.float32)
        w[f"{block}.ff1.weight"] = np.zeros_like(w[f"{block}.ff1.weight"], dtype=np.float32)
        w[f"{block}.ff1.bias"] = np.zeros_like(w[f"{block}.ff1.bias"], dtype=np.float32)
        w[f"{block}.ff2.weight"] = np.zeros_like(w[f"{block}.ff2.weight"], dtype=np.float32)
        w[f"{block}.ff2.bias"] = np.zeros_like(w[f"{block}.ff2.bias"], dtype=np.float32)
        # Set norm weights to 1, biases to 0
        w[f"{block}.norm1.weight"] = np.ones_like(w[f"{block}.norm1.weight"], dtype=np.float32)
        w[f"{block}.norm1.bias"] = np.zeros_like(w[f"{block}.norm1.bias"], dtype=np.float32)
        w[f"{block}.norm2.weight"] = np.ones_like(w[f"{block}.norm2.weight"], dtype=np.float32)
        w[f"{block}.norm2.bias"] = np.zeros_like(w[f"{block}.norm2.bias"], dtype=np.float32)

    # Final norm
    w["norm.weight"] = np.ones_like(w["norm.weight"], dtype=np.float32)
    w["norm.bias"] = np.zeros_like(w["norm.bias"], dtype=np.float32)

    # 4. Compute the new features for the full pool
    # We need to simulate the forward pass up to the pooled feature.
    # Since blocks are zero, h = embed(ids). Then final norm: LN(h). Then pooled = mean over non-pad.
    # We can compute this efficiently.
    # For each example, get the token ids, look up embed_new, apply LayerNorm per token, then average over non-pad.
    # LayerNorm: (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta. Here gamma=1, beta=0, eps=1e-5.
    # We can compute this for all tokens in the batch.
    # Let's compute features for all 4000 examples.
    # We'll process in chunks to save memory.
    features = np.zeros((N, D), dtype=np.float32)
    eps = 1e-5
    for i in range(N):
        ids = x[i]
        mask = ids != 0
        if not np.any(mask):
            continue
        tokens = embed_new[ids[mask]]  # (n_tokens, D)
        # LayerNorm per token
        mean = tokens.mean(axis=1, keepdims=True)
        var = tokens.var(axis=1, keepdims=True)
        tokens_norm = (tokens - mean) / np.sqrt(var + eps)
        # Pooled mean
        features[i] = tokens_norm.mean(axis=0)

    # 5. Fit a new head on these features
    # We'll use ridge regression on ±1 targets, with lambda chosen by 5-fold CV.
    # Add intercept
    features_aug = np.concatenate([features, np.ones((N, 1), dtype=np.float32)], axis=1)
    XtX_head = features_aug.T @ features_aug
    Xty_head = features_aug.T @ t
    lambdas_head = [0.001, 0.01, 0.1, 1.0, 10.0]
    best_lambda_head = 0.1
    best_cv_acc_head = -1.0
    fold_size = N // 5
    indices = np.arange(N)
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    for lam in lambdas_head:
        cv_acc = 0.0
        for fold in range(5):
            val_idx = indices[fold*fold_size:(fold+1)*fold_size]
            train_idx = np.setdiff1d(indices, val_idx)
            X_val = features_aug[val_idx]
            XtX_val = X_val.T @ X_val
            Xty_val = X_val.T @ t[val_idx]
            XtX_train = XtX_head - XtX_val
            Xty_train = Xty_head - Xty_val
            reg = lam * np.eye(XtX_train.shape[0], dtype=np.float32)
            w_head = np.linalg.solve(XtX_train + reg, Xty_train)
            scores = X_val @ w_head
            preds = (scores > 0).astype(np.float32)
            acc = np.mean(preds == y[val_idx])
            cv_acc += acc
        cv_acc /= 5.0
        if cv_acc > best_cv_acc_head:
            best_cv_acc_head = cv_acc
            best_lambda_head = lam

    # Fit final head
    reg = best_lambda_head * np.eye(XtX_head.shape[0], dtype=np.float32)
    w_head_full = np.linalg.solve(XtX_head + reg, Xty_head)
    head_w = w_head_full[:D].astype(np.float32)  # (32,)
    head_b = w_head_full[D]

    # 6. Calibrate logits with LDA-style scale and bias
    # Compute scores on full pool
    scores = features @ head_w + head_b
    # For each class, compute mean and variance
    scores_0 = scores[y == 0]
    scores_1 = scores[y == 1]
    mu0, mu1 = scores_0.mean(), scores_1.mean()
    var0, var1 = scores_0.var(), scores_1.var()
    # Pooled variance
    var_pool = (var0 + var1) / 2.0
    if var_pool < 1e-8:
        var_pool = 1.0
    # LDA scale: we want logits such that logit = a * score + c, with decision boundary at 0.
    # The optimal a and c for logistic loss can be approximated by matching the class-conditional densities.
    # For equal variances, the LDA direction is score, and the log-odds is (score - (mu0+mu1)/2) / var_pool * (mu1 - mu0).
    # So we can set a = (mu1 - mu0) / var_pool, and c = -a * (mu0 + mu1) / 2.
    a = (mu1 - mu0) / var_pool
    c = -a * (mu0 + mu1) / 2.0
    # We want head.weight to be [ -a*head_w, a*head_w ] and head.bias to be [ -c, c ]? Wait, logits are 2D.
    # The model outputs logits for class 0 and class 1. We want logit_1 - logit_0 = a * score + c.
    # So we can set head.weight[1] = a * head_w, head.weight[0] = -a * head_w, and head.bias[1] = c, head.bias[0] = -c.
    # But the original head has shape (2, 32). We can set:
    w["head.weight"] = np.zeros((2, D), dtype=np.float32)
    w["head.weight"][1] = a * head_w
    w["head.weight"][0] = -a * head_w
    w["head.bias"] = np.zeros(2, dtype=np.float32)
    w["head.bias"][1] = c
    w["head.bias"][0] = -c

    # Diagnostics
    diagnostics = {
        "best_lambda_ridge": float(best_lambda),
        "best_lambda_head": float(best_lambda_head),
        "cv_acc_ridge": float(best_cv_acc),
        "cv_acc_head": float(best_cv_acc_head),
        "train_acc_head": float(np.mean((scores > 0) == y)),
        "embed_col0_std": float(embed_new[:, 0].std()),
        "embed_col1_std": float(embed_new[:, 1].std()),
        "head_scale": float(a),
        "head_bias": float(c),
    }

    return {"weights": w, "diagnostics": diagnostics}
