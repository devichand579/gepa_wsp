import numpy as np
import math

def edit(w, f):
    # Extract training data
    x = f["x"]          # [N, 64] int64 token ids
    y = f["y"]          # [N] int64 labels
    N, L = x.shape
    V = w["embed.weight"].shape[0]  # 2000

    # Build design matrix: X[i, v] = count of token v in example i / number of non-pad tokens
    # We use all tokens including padding? Padding is id 0. The forward pass excludes padding from mean.
    # So we should exclude token id 0 from the mean. The mask is x != 0.
    mask = (x != 0)  # [N, L]
    lengths = mask.sum(axis=1).astype(np.float32)  # [N]
    lengths = np.maximum(lengths, 1.0)  # avoid division by zero

    # Build sparse count matrix efficiently
    # We'll create a dense matrix X of shape [N, V] but V=2000, N=4000 -> 8M floats, fine.
    X = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        # Get token ids for this example, excluding padding
        toks = x[i][mask[i]]
        if len(toks) > 0:
            # Count occurrences
            counts = np.bincount(toks, minlength=V)
            X[i] = counts / lengths[i]

    # Ridge regression: minimize ||X w + b - y||^2 + lambda ||w||^2
    # We can augment X with a column of ones for bias.
    # But we want to regularize only w, not bias. So we center y and X? Simpler: use standard ridge with bias unregularized.
    # Let's use the closed-form solution for ridge with intercept.
    # We'll try a few lambdas via cross-validation on the training set.
    # To avoid iterative optimizer, we just compute closed-form for each lambda.
    # We'll use 5-fold cross-validation.

    lambdas = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    best_lambda = None
    best_cv_acc = -1.0
    best_w = None
    best_b = None

    # Precompute XTX and XTy for full data? For CV we need to split.
    # We'll do a simple 5-fold split.
    indices = np.arange(N)
    np.random.seed(42)  # for reproducibility
    np.random.shuffle(indices)
    folds = np.array_split(indices, 5)

    for lam in lambdas:
        cv_accs = []
        for k in range(5):
            val_idx = folds[k]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != k])
            X_train = X[train_idx]
            y_train = y[train_idx].astype(np.float32)
            X_val = X[val_idx]
            y_val = y[val_idx]

            # Center X and y for intercept
            X_mean = X_train.mean(axis=0)
            y_mean = y_train.mean()
            Xc = X_train - X_mean
            yc = y_train - y_mean

            # Ridge solution: w = (Xc^T Xc + lam I)^-1 Xc^T yc
            # Use np.linalg.solve
            A = Xc.T @ Xc + lam * np.eye(V, dtype=np.float32)
            b_vec = Xc.T @ yc
            w_ridge = np.linalg.solve(A, b_vec)
            b_ridge = y_mean - X_mean @ w_ridge

            # Predict on validation
            pred = X_val @ w_ridge + b_ridge
            pred_label = (pred > 0.5).astype(np.int64)  # since y in {0,1}
            acc = (pred_label == y_val).mean()
            cv_accs.append(acc)
        mean_cv = np.mean(cv_accs)
        if mean_cv > best_cv_acc:
            best_cv_acc = mean_cv
            best_lambda = lam

    # Fit final model on all data with best_lambda
    X_mean = X.mean(axis=0)
    y_mean = y.mean()
    Xc = X - X_mean
    yc = y - y_mean
    A = Xc.T @ Xc + best_lambda * np.eye(V, dtype=np.float32)
    b_vec = Xc.T @ yc
    w_final = np.linalg.solve(A, b_vec)
    b_final = y_mean - X_mean @ w_final

    # Now construct new embed.weight
    # We want e_v in R^32 such that:
    #   mean(e_v) = 0
    #   ||e_v|| = R (constant)
    #   delta . e_v = w_final[v]  (for some fixed delta)
    # We choose delta = [1, -1, 0, ..., 0] (norm^2 = 2, zero mean)
    # and v_perp = [1, 1, -2, 0, ..., 0] (norm^2 = 6, zero mean, orthogonal to delta)
    # Then e_v = a_v * delta + c_v * v_perp
    # delta . e_v = a_v * ||delta||^2 = 2 a_v = w_final[v] => a_v = w_final[v] / 2
    # ||e_v||^2 = a_v^2 * 2 + c_v^2 * 6 = w_final[v]^2 / 2 + 6 c_v^2 = R^2
    # => c_v^2 = (R^2 - w_final[v]^2 / 2) / 6
    # We need R^2 >= max_v (w_final[v]^2 / 2). Choose R with a margin.
    max_w2 = np.max(w_final**2)
    R2 = max_w2 / 2.0 + 1.0  # add margin to keep c_v real
    R = math.sqrt(R2)

    delta = np.zeros(32, dtype=np.float32)
    delta[0] = 1.0
    delta[1] = -1.0

    v_perp = np.zeros(32, dtype=np.float32)
    v_perp[0] = 1.0
    v_perp[1] = 1.0
    v_perp[2] = -2.0

    # Build new embed weight
    new_embed = np.zeros((V, 32), dtype=np.float32)
    for v in range(V):
        a_v = w_final[v] / 2.0
        c_v_sq = (R2 - w_final[v]**2 / 2.0) / 6.0
        c_v = math.sqrt(max(0.0, c_v_sq))
        # We can choose sign of c_v arbitrarily; use positive.
        new_embed[v] = a_v * delta + c_v * v_perp

    # Now set head to read delta.
    # The forward pass: pooled = mean of LayerNorm(e_token) over non-pad tokens.
    # LayerNorm(e) = e / sqrt(||e||^2/32 + eps) since mean=0.
    # ||e|| = R, so LayerNorm(e) = e * sqrt(32) / sqrt(R^2 + 32*eps) ≈ e * sqrt(32)/R.
    # Let alpha = sqrt(32) / R.
    # pooled = alpha * (1/L) sum e_token.
    # logit1 = head.weight[1] . pooled + head.bias[1]
    # We want logit1 = alpha * (X_i w_final + b_final) = alpha * ( (1/L) sum w_final[token] + b_final )
    # So set head.weight[1] = delta, head.bias[1] = alpha * b_final.
    # head.weight[0] = 0, head.bias[0] = 0.
    alpha = math.sqrt(32.0) / R

    new_head_weight = np.zeros((2, 32), dtype=np.float32)
    new_head_weight[1] = delta
    new_head_bias = np.zeros(2, dtype=np.float32)
    new_head_bias[1] = alpha * b_final

    # Update weights
    w["embed.weight"] = new_embed.astype(np.float32)
    w["head.weight"] = new_head_weight
    w["head.bias"] = new_head_bias

    # Also ensure pos.weight is zero (it already is)
    w["pos.weight"] = np.zeros_like(w["pos.weight"])

    # Ensure all other weights are zero or identity as before? They are already zero.
    # But we should not change them.

    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_acc": float(best_cv_acc),
        "bias": float(b_final),
        "alpha": float(alpha),
        "R": float(R),
        "w_norm": float(np.linalg.norm(w_final)),
        "embed_norm_mean": float(np.mean(np.linalg.norm(new_embed, axis=1))),
        "embed_mean": float(np.mean(new_embed)),
        "head_weight_norm": float(np.linalg.norm(new_head_weight)),
    }

    return {"weights": w, "diagnostics": diagnostics}
