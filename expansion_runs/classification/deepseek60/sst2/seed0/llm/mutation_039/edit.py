import numpy as np
import math

def edit(w, f):
    # Extract training data
    x = f["x"].astype(np.int64)          # [4000, 64]
    y = f["y"].astype(np.float32)        # [4000]
    N, L = x.shape
    V = w["embed.weight"].shape[0]       # 2000
    D = w["embed.weight"].shape[1]       # 32

    # Build count matrix (ignore padding token 0)
    # We'll use a dense matrix for simplicity; 4000x2000 is small.
    X_counts = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        row = x[i]
        # count non-pad tokens
        valid = row[row != 0]
        if valid.size > 0:
            np.add.at(X_counts[i], valid, 1.0)

    # Compute document lengths (non-pad tokens)
    doc_len = X_counts.sum(axis=1, keepdims=True)
    doc_len[doc_len == 0] = 1.0

    # TF-IDF features
    # Term frequency: count / doc_len
    tf = X_counts / doc_len
    # Document frequency: number of documents containing token
    df = (X_counts > 0).sum(axis=0).astype(np.float32)
    # IDF: log(N / (1 + df)) + 1  (smooth)
    idf = np.log((N + 1.0) / (df + 1.0)) + 1.0
    # TF-IDF
    X = tf * idf

    # Add bias column
    X_aug = np.concatenate([X, np.ones((N, 1), dtype=np.float32)], axis=1)
    y_target = y.astype(np.float32)

    # Ridge regression with cross-validation to select lambda
    # We'll use 5-fold CV on a small grid.
    lambdas = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    best_lambda = 1.0
    best_acc = -1.0

    # Shuffle indices for CV
    rng = np.random.RandomState(42)
    indices = rng.permutation(N)
    n_folds = 5
    fold_size = N // n_folds

    for lam in lambdas:
        accs = []
        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else N
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

            X_train = X_aug[train_idx]
            y_train = y_target[train_idx]
            X_val = X_aug[val_idx]
            y_val = y_target[val_idx]

            # Solve ridge: (X^T X + lam I) w = X^T y
            # Use normal equations. X_train is (3200, 2001). X^T X is 2001x2001.
            XtX = X_train.T @ X_train
            Xty = X_train.T @ y_train
            reg = lam * np.eye(XtX.shape[0], dtype=np.float32)
            # Don't regularize bias? Usually we do, but can skip. We'll regularize all for simplicity.
            w_aug = np.linalg.solve(XtX + reg, Xty)

            # Predict on validation
            scores = X_val @ w_aug
            preds = (scores > 0.5).astype(np.float32)
            acc = (preds == y_val).mean()
            accs.append(acc)
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lambda = lam

    # Train on full data with best lambda
    XtX = X_aug.T @ X_aug
    Xty = X_aug.T @ y_target
    reg = best_lambda * np.eye(XtX.shape[0], dtype=np.float32)
    w_aug = np.linalg.solve(XtX + reg, Xty)

    w_vec = w_aug[:-1]   # [2000]
    b = w_aug[-1]        # scalar

    # Now we need to embed this linear classifier into the network.
    # We want logit difference = mean(v_token · delta) + b = w·(c/n) + b.
    # So we need v_token · delta = w_token.
    # We'll choose delta as a fixed unit vector in the zero-sum subspace.
    # Let delta = (1, -1, 0, ..., 0) normalized? But must be zero-sum.
    # We can use a random zero-sum unit vector.
    rng2 = np.random.RandomState(123)
    delta = rng2.randn(D).astype(np.float32)
    delta -= delta.mean()
    delta /= np.linalg.norm(delta)

    # We need v_token such that:
    #   v_token · delta = w_token
    #   sum(v_token) = 0
    #   ||v_token||^2 = D  (so LayerNorm is identity)
    # We can set v_token = (w_token / ||delta||^2) * delta + u_token
    # where u_token is orthogonal to delta and to all-ones, and ||u_token||^2 = D - w_token^2 / ||delta||^2.
    # Since ||delta||=1, we need ||u_token||^2 = D - w_token^2.
    # We can choose u_token = sqrt(D - w_token^2) * u0, where u0 is a fixed unit vector orthogonal to delta and all-ones.
    # First, find u0.
    # Start with a random vector, project out delta and all-ones.
    u0 = rng2.randn(D).astype(np.float32)
    u0 -= u0.mean()  # orthogonal to all-ones
    u0 -= (u0 @ delta) * delta
    u0 /= np.linalg.norm(u0)

    # Now build embed.weight
    embed = np.zeros((V, D), dtype=np.float32)
    for i in range(V):
        w_i = w_vec[i]
        # Clip to avoid sqrt of negative due to numerical issues
        rem = D - w_i * w_i
        if rem < 0:
            rem = 0.0
        scale = math.sqrt(rem)
        embed[i] = w_i * delta + scale * u0

    # Ensure zero mean and unit variance for each row (LayerNorm identity)
    # The construction above ensures sum=0 and ||v||^2 = D, so variance = D/D = 1.
    # But due to floating point, we can renormalize.
    embed -= embed.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(embed, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embed = embed / norms * math.sqrt(D)

    # Set head weights: we want head1 - head0 = delta, head1_bias - head0_bias = b.
    # We can set head.weight[0] = -delta/2, head.weight[1] = delta/2.
    # head.bias[0] = -b/2, head.bias[1] = b/2.
    head_w = np.zeros((2, D), dtype=np.float32)
    head_w[0] = -delta / 2.0
    head_w[1] = delta / 2.0
    head_b = np.array([-b / 2.0, b / 2.0], dtype=np.float32)

    # Update weights
    w["embed.weight"] = embed.astype(np.float32)
    w["head.weight"] = head_w.astype(np.float32)
    w["head.bias"] = head_b.astype(np.float32)

    # Ensure all other weights are zero (identity blocks)
    # We can set all block weights and biases to zero, and norm weights to 1, biases to 0.
    for key in w:
        if key.startswith("block0.") or key.startswith("block1."):
            if "weight" in key:
                w[key] = np.zeros_like(w[key], dtype=np.float32)
            elif "bias" in key:
                w[key] = np.zeros_like(w[key], dtype=np.float32)
    # Set norm weights to 1, biases to 0
    for key in ["norm.weight", "norm.bias"]:
        if key in w:
            if "weight" in key:
                w[key] = np.ones_like(w[key], dtype=np.float32)
            else:
                w[key] = np.zeros_like(w[key], dtype=np.float32)
    # pos.weight is already zero, but ensure
    if "pos.weight" in w:
        w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)

    # Diagnostics
    diag = {
        "best_lambda": float(best_lambda),
        "cv_acc": float(best_acc),
        "w_norm": float(np.linalg.norm(w_vec)),
        "bias": float(b),
        "embed_mean": float(embed.mean()),
        "embed_std": float(embed.std()),
        "delta_norm": float(np.linalg.norm(delta)),
    }

    return {"weights": w, "diagnostics": diag}
