import numpy as np

def edit(w, f):
    # Extract feedback arrays
    x = f["x"]          # [4000, 64] int64
    y = f["y"]          # [4000] int64
    N, L = x.shape
    V = w["embed.weight"].shape[0]  # 2000
    D = w["embed.weight"].shape[1]  # 32

    # Build BOW matrix (binary)
    X = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        for j in range(L):
            tok = int(x[i, j])
            if tok != 0:
                X[i, tok] = 1.0

    # Target: +1 for positive (1), -1 for negative (0)
    y_pm = np.where(y == 1, 1.0, -1.0).astype(np.float32)

    # --- Fit ridge classifier on BOW ---
    # Use cross-validation to pick lambda
    lambdas = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    best_lambda = 1.0
    best_cv_acc = -1.0
    # 5-fold CV
    n_folds = 5
    fold_size = N // n_folds
    indices = np.arange(N)
    # Simple deterministic split
    for lam in lambdas:
        accs = []
        for fold in range(n_folds):
            val_idx = indices[fold*fold_size:(fold+1)*fold_size]
            train_idx = np.setdiff1d(indices, val_idx)
            X_train = X[train_idx]
            y_train = y_pm[train_idx]
            X_val = X[val_idx]
            y_val = y_pm[val_idx]
            # Solve (X^T X + lam I) w = X^T y
            A = X_train.T @ X_train + lam * np.eye(V, dtype=np.float32)
            b = X_train.T @ y_train
            w_lin = np.linalg.solve(A, b)
            pred = X_val @ w_lin
            pred_label = (pred > 0).astype(np.int64)
            true_label = (y_val > 0).astype(np.int64)
            acc = np.mean(pred_label == true_label)
            accs.append(acc)
        mean_acc = np.mean(accs)
        if mean_acc > best_cv_acc:
            best_cv_acc = mean_acc
            best_lambda = lam

    # Final fit on all data
    A = X.T @ X + best_lambda * np.eye(V, dtype=np.float32)
    b = X.T @ y_pm
    w_lin = np.linalg.solve(A, b)  # [V]

    # --- PCA on BOW (top 31 components) ---
    # Center X
    X_mean = X.mean(axis=0, keepdims=True)
    X_centered = X - X_mean
    # Compute covariance (V x V) is too large (2000x2000) but we can use SVD on X_centered (4000x2000)
    # Actually we need top 31 right singular vectors. Use randomized SVD or just compute X_centered.T @ X_centered? 
    # 2000x2000 is 4M floats, ~16MB, manageable.
    cov = X_centered.T @ X_centered / (N - 1)
    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    # Take top 31
    pca_components = eigvecs[:, :D-1].T  # [31, V]

    # --- Construct embed.weight ---
    # Column 0: ridge classifier weights (scaled)
    # Columns 1..31: PCA components
    embed_new = np.zeros((V, D), dtype=np.float32)
    embed_new[:, 0] = w_lin
    embed_new[:, 1:] = pca_components.T

    # Normalize each row to have zero mean and unit variance? 
    # Actually, we want LayerNorm(embed[v]) to be roughly linear in w_lin and PCA.
    # If we just set embed_new as above, LayerNorm will normalize each row.
    # To make LayerNorm approximately identity, we can center each row and scale to unit variance.
    # But centering will remove the mean of w_lin, which is fine because we can adjust head bias.
    # Let's center each row (across 32 dims) and scale to unit variance.
    row_mean = embed_new.mean(axis=1, keepdims=True)
    row_std = embed_new.std(axis=1, keepdims=True) + 1e-8
    embed_new = (embed_new - row_mean) / row_std

    # Assign to w
    w["embed.weight"][:] = embed_new.astype(np.float32)

    # Zero out pos and all transformer blocks
    for key in w:
        if key.startswith("pos."):
            w[key][:] = 0.0
        if key.startswith("block0.") or key.startswith("block1."):
            w[key][:] = 0.0
        if key.startswith("norm."):
            if key.endswith("weight"):
                w[key][:] = 1.0
            else:
                w[key][:] = 0.0

    # --- Compute pooled features for all examples ---
    # Forward pass: h = embed[ids]; final norm = LayerNorm(h); pooled = mean over non-pad tokens
    # We need to compute LayerNorm(embed[token]) for each token, then average.
    # Since embed is fixed, we can compute the normalized embedding for each token.
    # LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias
    # Here weight=1, bias=0, eps=1e-5
    eps = 1e-5
    # Compute normalized embedding for each token v
    # embed_norm[v] = (embed[v] - mean(embed[v])) / sqrt(var(embed[v]) + eps)
    # But we already centered and scaled rows to zero mean unit variance, so embed_norm[v] ≈ embed[v]
    # Actually, we centered and scaled, so mean=0, std=1. So LayerNorm is just embed[v] / sqrt(1 + eps) ≈ embed[v].
    # So pooled feature = mean over tokens of embed[token].
    # Compute features for all examples
    features = np.zeros((N, D), dtype=np.float32)
    for i in range(N):
        tokens = x[i]
        mask = tokens != 0
        if mask.sum() == 0:
            continue
        # Get embeddings for tokens
        embs = w["embed.weight"][tokens[mask]]  # [num_tokens, D]
        # Apply LayerNorm per token (already centered/scaled, but do it properly)
        mean = embs.mean(axis=1, keepdims=True)
        var = embs.var(axis=1, keepdims=True)
        embs_norm = (embs - mean) / np.sqrt(var + eps)
        # Pooled mean
        features[i] = embs_norm.mean(axis=0)

    # --- Fit head on features ---
    # Use ridge regression on ±1 targets
    # Add bias term by augmenting features with 1
    F_aug = np.hstack([features, np.ones((N, 1), dtype=np.float32)])
    # Fit ridge for each class? Actually we want logits for 2 classes.
    # We can fit a linear model to predict y_pm, then set head weights accordingly.
    # Head is [2, 32] + bias [2]. We want logit_1 - logit_0 = w_head · features + b.
    # So we can set head.weight[1] = w_head, head.weight[0] = -w_head, head.bias[1] = b/2, head.bias[0] = -b/2.
    # Fit ridge on F_aug to predict y_pm
    lambdas_head = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    best_lambda_head = 1.0
    best_cv_acc_head = -1.0
    for lam in lambdas_head:
        accs = []
        for fold in range(n_folds):
            val_idx = indices[fold*fold_size:(fold+1)*fold_size]
            train_idx = np.setdiff1d(indices, val_idx)
            F_train = F_aug[train_idx]
            y_train = y_pm[train_idx]
            F_val = F_aug[val_idx]
            y_val = y_pm[val_idx]
            # Solve (F^T F + lam I) w = F^T y
            D_aug = F_train.shape[1]
            A = F_train.T @ F_train + lam * np.eye(D_aug, dtype=np.float32)
            b = F_train.T @ y_train
            w_head = np.linalg.solve(A, b)
            pred = F_val @ w_head
            pred_label = (pred > 0).astype(np.int64)
            true_label = (y_val > 0).astype(np.int64)
            acc = np.mean(pred_label == true_label)
            accs.append(acc)
        mean_acc = np.mean(accs)
        if mean_acc > best_cv_acc_head:
            best_cv_acc_head = mean_acc
            best_lambda_head = lam

    # Final head fit
    D_aug = F_aug.shape[1]
    A = F_aug.T @ F_aug + best_lambda_head * np.eye(D_aug, dtype=np.float32)
    b = F_aug.T @ y_pm
    w_head_aug = np.linalg.solve(A, b)  # [D+1]
    w_head = w_head_aug[:D]
    b_head = w_head_aug[D]

    # Set head weights
    # head.weight[1] = w_head, head.weight[0] = -w_head
    # head.bias[1] = b_head/2, head.bias[0] = -b_head/2
    w["head.weight"][1, :] = w_head.astype(np.float32)
    w["head.weight"][0, :] = -w_head.astype(np.float32)
    w["head.bias"][1] = b_head / 2.0
    w["head.bias"][0] = -b_head / 2.0

    # Ensure all values are finite and within bounds
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)

    diagnostics = {
        "best_lambda_bow": float(best_lambda),
        "best_lambda_head": float(best_lambda_head),
        "cv_acc_bow": float(best_cv_acc),
        "cv_acc_head": float(best_cv_acc_head),
        "train_acc_head": float(np.mean((F_aug @ w_head_aug > 0) == (y_pm > 0))),
        "embed_col0_std": float(np.std(w["embed.weight"][:, 0])),
        "head_scale": float(np.linalg.norm(w_head)),
        "head_bias": float(b_head),
    }
    return {"weights": w, "diagnostics": diagnostics}
