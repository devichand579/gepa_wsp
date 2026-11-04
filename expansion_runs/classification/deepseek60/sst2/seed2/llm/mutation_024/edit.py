import numpy as np

def edit(w, f):
    # Extract feedback data
    x = f["x"]          # (4000, 64) int64
    y = f["y"]          # (4000,) int64
    N, L = x.shape
    V = w["embed.weight"].shape[0]  # 2000

    # Build BOW matrix (4000, V)
    BOW = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        for t in x[i]:
            if t > 0:  # ignore padding (0)
                BOW[i, t] += 1.0

    # Normalize by sequence length (non-pad tokens)
    lengths = np.sum(x > 0, axis=1).astype(np.float32)
    lengths[lengths == 0] = 1.0
    BOW_norm = BOW / lengths[:, None]

    # Ridge regression on BOW to get a linear classifier
    # Target: +1 for positive (1), -1 for negative (0)
    y_signed = 2.0 * y.astype(np.float32) - 1.0

    # Center BOW and target
    BOW_mean = BOW_norm.mean(axis=0, keepdims=True)
    BOW_centered = BOW_norm - BOW_mean
    y_mean = y_signed.mean()
    y_centered = y_signed - y_mean

    # Choose lambda by 5-fold CV on the full pool
    best_lambda = None
    best_cv_acc = -1.0
    lambdas = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    # Simple 5-fold split
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, 5)

    for lam in lambdas:
        cv_correct = 0
        for k in range(5):
            val_idx = folds[k]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != k])
            X_tr = BOW_centered[train_idx]
            y_tr = y_centered[train_idx]
            X_val = BOW_centered[val_idx]
            y_val = y_signed[val_idx]
            # Closed-form ridge: w = (X^T X + lam I)^-1 X^T y
            # Use dual form for efficiency: w = X^T (X X^T + lam I)^-1 y
            # Here N=4000, V=2000. X is 4000x2000. X X^T is 4000x4000, too big.
            # Instead use primal: X^T X is 2000x2000, manageable.
            XtX = X_tr.T @ X_tr
            Xty = X_tr.T @ y_tr
            w_ridge = np.linalg.solve(XtX + lam * np.eye(V, dtype=np.float32), Xty)
            # Predict on validation
            pred = X_val @ w_ridge
            pred_sign = np.sign(pred)
            # Accuracy
            acc = np.mean(pred_sign == y_val)
            cv_correct += acc * len(val_idx)
        cv_acc = cv_correct / N
        if cv_acc > best_cv_acc:
            best_cv_acc = cv_acc
            best_lambda = lam

    # Fit final ridge on all data with best_lambda
    XtX = BOW_centered.T @ BOW_centered
    Xty = BOW_centered.T @ y_centered
    w_bow = np.linalg.solve(XtX + best_lambda * np.eye(V, dtype=np.float32), Xty)
    b_bow = y_mean - BOW_mean @ w_bow  # intercept

    # Compute top 31 principal components of BOW_centered
    # Use SVD on a subset or covariance matrix. V=2000, N=4000.
    # Covariance = BOW_centered^T BOW_centered / N
    cov = XtX / N
    # Eigen decomposition (symmetric)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    # Take top 31 (excluding the first, which is the mean direction? Actually BOW_centered already centered, so first PC is the main variance)
    # We want 31 components. Let's take the top 31 eigenvectors.
    top_pcs = eigvecs[:, :31]  # (V, 31)

    # Scale w_bow and top_pcs to have reasonable magnitude for embedding
    # The final LayerNorm will normalize each token embedding. To make LN identity, we need each row to be zero-mean and unit-variance.
    # But we can also just let LN normalize. It's fine.
    # We'll set embed.weight columns: col0 = w_bow, cols1..31 = top_pcs
    # But w_bow and top_pcs have different scales. We can normalize each column to unit norm.
    def normalize_col(v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    w_bow_norm = normalize_col(w_bow)
    top_pcs_norm = top_pcs / np.linalg.norm(top_pcs, axis=0, keepdims=True)

    # Build new embed.weight
    embed_new = np.zeros((V, 32), dtype=np.float32)
    embed_new[:, 0] = w_bow_norm
    embed_new[:, 1:32] = top_pcs_norm

    # Make each row zero-mean and unit-variance across the 32 dims to make final LayerNorm identity
    # Actually, final LayerNorm is applied per token. If we make each row zero-mean/unit-variance, LN is identity.
    row_mean = embed_new.mean(axis=1, keepdims=True)
    row_std = embed_new.std(axis=1, keepdims=True)
    row_std[row_std == 0] = 1.0
    embed_new = (embed_new - row_mean) / row_std

    # Update w
    w["embed.weight"] = embed_new.astype(np.float32)

    # Set pos to zero
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)

    # Set all transformer blocks to zero (they are already zero, but ensure)
    for block in ["block0", "block1"]:
        for name in ["norm1.weight", "norm1.bias", "qkv.weight", "qkv.bias",
                     "proj.weight", "proj.bias", "norm2.weight", "norm2.bias",
                     "ff1.weight", "ff1.bias", "ff2.weight", "ff2.bias"]:
            key = f"{block}.{name}"
            if key in w:
                if "weight" in name and "norm" in name:
                    w[key] = np.ones_like(w[key], dtype=np.float32)
                elif "bias" in name and "norm" in name:
                    w[key] = np.zeros_like(w[key], dtype=np.float32)
                else:
                    w[key] = np.zeros_like(w[key], dtype=np.float32)

    # Set final norm to identity (weight=1, bias=0)
    w["norm.weight"] = np.ones_like(w["norm.weight"], dtype=np.float32)
    w["norm.bias"] = np.zeros_like(w["norm.bias"], dtype=np.float32)

    # Now compute the pooled features for the full pool using the new embed
    # Since blocks are zero and pos zero, h = embed[token]. Final norm is identity because rows are zero-mean/unit-variance.
    # Pooled feature = mean of embed[token] over non-pad tokens.
    # We already have BOW_norm = BOW / lengths. So pooled = BOW_norm @ embed_new.
    pooled = BOW_norm @ embed_new  # (4000, 32)

    # Fit a new head on pooled features using ridge regression on ±1 targets
    # Center pooled
    pooled_mean = pooled.mean(axis=0, keepdims=True)
    pooled_centered = pooled - pooled_mean
    # y_centered already computed

    # Choose lambda for head by 5-fold CV
    best_lambda_head = None
    best_cv_acc_head = -1.0
    for lam in lambdas:
        cv_correct = 0
        for k in range(5):
            val_idx = folds[k]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != k])
            X_tr = pooled_centered[train_idx]
            y_tr = y_centered[train_idx]
            X_val = pooled_centered[val_idx]
            y_val = y_signed[val_idx]
            # Ridge on 32 dims
            XtX = X_tr.T @ X_tr
            Xty = X_tr.T @ y_tr
            w_head = np.linalg.solve(XtX + lam * np.eye(32, dtype=np.float32), Xty)
            pred = X_val @ w_head
            pred_sign = np.sign(pred)
            acc = np.mean(pred_sign == y_val)
            cv_correct += acc * len(val_idx)
        cv_acc = cv_correct / N
        if cv_acc > best_cv_acc_head:
            best_cv_acc_head = cv_acc
            best_lambda_head = lam

    # Fit final head
    XtX = pooled_centered.T @ pooled_centered
    Xty = pooled_centered.T @ y_centered
    w_head = np.linalg.solve(XtX + best_lambda_head * np.eye(32, dtype=np.float32), Xty)
    b_head = y_mean - pooled_mean @ w_head

    # The head output is 2 classes. We need to set head.weight and head.bias.
    # The logit for class 1 (positive) should be w_head · pooled + b_head.
    # The logit for class 0 (negative) should be - (w_head · pooled + b_head).
    # So head.weight[1] = w_head, head.bias[1] = b_head
    # head.weight[0] = -w_head, head.bias[0] = -b_head
    w["head.weight"] = np.zeros((2, 32), dtype=np.float32)
    w["head.weight"][1] = w_head
    w["head.weight"][0] = -w_head
    w["head.bias"] = np.zeros(2, dtype=np.float32)
    w["head.bias"][1] = b_head
    w["head.bias"][0] = -b_head

    # Diagnostics
    diagnostics = {
        "best_lambda_bow": float(best_lambda),
        "best_lambda_head": float(best_lambda_head),
        "cv_acc_bow": float(best_cv_acc),
        "cv_acc_head": float(best_cv_acc_head),
        "train_acc_head": float(np.mean(np.sign(pooled @ w_head + b_head) == y_signed)),
        "embed_col0_std": float(embed_new[:, 0].std()),
        "head_scale": float(np.linalg.norm(w_head)),
        "head_bias": float(b_head)
    }

    return {"weights": w, "diagnostics": diagnostics}
