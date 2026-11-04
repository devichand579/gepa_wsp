import numpy as np
import math

def edit(w, f):
    # Extract feedback arrays
    x = f["x"]          # [4000, 64] int64 token ids
    y = f["y"]          # [4000] int64 labels (0/1)
    N, L = x.shape
    V = w["embed.weight"].shape[0]  # 2000

    # Build term-frequency features: count of each token / number of non-pad tokens
    # Mask for non-padding tokens (id != 0)
    mask = (x != 0)  # [N, 64]
    lengths = mask.sum(axis=1).astype(np.float32)  # [N]
    lengths[lengths == 0] = 1.0  # avoid division by zero

    # Count occurrences of each token per example
    # We can use np.add.at or bincount per example, but vectorized is faster.
    # Create a one-hot-like count matrix: [N, V]
    # Since V=2000 and N=4000, we can do:
    X = np.zeros((N, V), dtype=np.float32)
    # For each example, count tokens
    for i in range(N):
        # Get non-pad tokens
        tokens = x[i][mask[i]]
        if len(tokens) > 0:
            counts = np.bincount(tokens, minlength=V)
            X[i] = counts / lengths[i]

    # Fit ridge regression on centered labels
    y_float = y.astype(np.float32)
    y_mean = y_float.mean()
    y_centered = y_float - y_mean

    # Ridge regression: w = (X^T X + lambda I)^-1 X^T y_centered
    # Use a small lambda for numerical stability. We can try a few lambdas and pick best.
    # Since we have the full pool, we can do a quick grid search using closed-form solutions.
    best_acc = -1.0
    best_w = None
    best_b = None
    best_lambda = None

    # Precompute X^T X and X^T y_centered
    XtX = X.T @ X
    Xty = X.T @ y_centered

    for lam in [0.01, 0.1, 1.0, 10.0, 100.0]:
        # Solve (XtX + lam * I) w = Xty
        A = XtX + lam * np.eye(V, dtype=np.float32)
        try:
            w_vec = np.linalg.solve(A, Xty)
        except np.linalg.LinAlgError:
            continue
        # Bias: since we centered y, the intercept is y_mean.
        # But we need to account for the fact that X is not centered.
        # The model is: y_pred = X w + b, where b = y_mean - (X_mean @ w).
        # Actually, if we fit on centered y, the optimal b is y_mean - mean(X) @ w.
        # Let's compute mean of X over examples.
        X_mean = X.mean(axis=0)  # [V]
        b = y_mean - X_mean @ w_vec
        # Evaluate accuracy on the feedback pool
        scores = X @ w_vec + b
        preds = (scores > 0.5).astype(np.int64)  # since y is 0/1, threshold at 0.5
        acc = (preds == y).mean()
        if acc > best_acc:
            best_acc = acc
            best_w = w_vec
            best_b = b
            best_lambda = lam

    if best_w is None:
        # Fallback: use lambda=1.0
        lam = 1.0
        A = XtX + lam * np.eye(V, dtype=np.float32)
        best_w = np.linalg.solve(A, Xty)
        X_mean = X.mean(axis=0)
        best_b = y_mean - X_mean @ best_w
        best_lambda = lam

    # Scale w and b so that max |w_t| <= 5.5 (to satisfy the normalization constraint)
    max_abs = np.max(np.abs(best_w))
    if max_abs > 5.5:
        scale = 5.5 / max_abs
        best_w = best_w * scale
        best_b = best_b * scale

    # Now construct the token embeddings v_t (32-dim) such that:
    # v_t[0] = best_w[t]
    # mean(v_t) = 0, var(v_t) = 1
    # We have 32 dims. Let v_t[0] = w_t. We need to choose v_t[1..31] (31 dims) such that:
    # sum(v_t[1..31]) = -w_t
    # sum(v_t[1..31]^2) = 32 - w_t^2
    # We can set v_t[1..31] = c * 1 + d, where c = -w_t/31, and d is orthogonal to 1.
    # Then sum(d^2) = 32 - w_t^2 - 31*c^2 = 32 - w_t^2 - w_t^2/31 = 32 - w_t^2 * 32/31.
    # We need this >= 0, which holds since |w_t| <= 5.5 < sqrt(31) ≈ 5.568.
    # We can set d to have only two non-zero entries: d[0] = sqrt(needed/2), d[1] = -sqrt(needed/2).
    # Then v_t[1] = c + d[0], v_t[2] = c + d[1], v_t[3..31] = c.

    embed_weight = w["embed.weight"]  # [2000, 32]
    # We'll overwrite embed_weight for all tokens.
    # First, ensure pos.weight is zero.
    w["pos.weight"][:] = 0.0

    # Zero out proj and ff2 in both blocks to make blocks identity.
    for block in ["block0", "block1"]:
        w[f"{block}.proj.weight"][:] = 0.0
        w[f"{block}.proj.bias"][:] = 0.0
        w[f"{block}.ff2.weight"][:] = 0.0
        w[f"{block}.ff2.bias"][:] = 0.0
        # Also zero qkv and ff1 to be safe (not strictly needed since proj/ff2 are zero)
        w[f"{block}.qkv.weight"][:] = 0.0
        w[f"{block}.qkv.bias"][:] = 0.0
        w[f"{block}.ff1.weight"][:] = 0.0
        w[f"{block}.ff1.bias"][:] = 0.0

    # Set norm weights to 1 and biases to 0 (they already are, but ensure)
    for name in ["block0.norm1.weight", "block0.norm1.bias",
                 "block0.norm2.weight", "block0.norm2.bias",
                 "block1.norm1.weight", "block1.norm1.bias",
                 "block1.norm2.weight", "block1.norm2.bias",
                 "norm.weight", "norm.bias"]:
        if "weight" in name:
            w[name][:] = 1.0
        else:
            w[name][:] = 0.0

    # Construct v_t for each token
    for t in range(V):
        wt = best_w[t]
        # c = -wt / 31
        c = -wt / 31.0
        needed_norm2 = 32.0 - wt * wt * (32.0 / 31.0)
        if needed_norm2 < 0:
            needed_norm2 = 0.0
        # d has 31 elements, we set d[0] = sqrt(needed/2), d[1] = -sqrt(needed/2), rest 0.
        d0 = math.sqrt(needed_norm2 / 2.0) if needed_norm2 > 0 else 0.0
        # v_t[0] = wt
        embed_weight[t, 0] = wt
        # v_t[1] = c + d0
        embed_weight[t, 1] = c + d0
        # v_t[2] = c - d0
        embed_weight[t, 2] = c - d0
        # v_t[3..31] = c
        embed_weight[t, 3:] = c

    # Set head to read only the first coordinate.
    # logit_0 = pooled[0] + b, logit_1 = -pooled[0] - b
    # So head.weight[0,0] = 1, head.weight[1,0] = -1, rest 0.
    # head.bias = [b, -b]
    head_weight = w["head.weight"]  # [2, 32]
    head_bias = w["head.bias"]      # [2]
    head_weight[:] = 0.0
    head_weight[0, 0] = 1.0
    head_weight[1, 0] = -1.0
    head_bias[0] = best_b
    head_bias[1] = -best_b

    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lambda),
        "train_accuracy": float(best_acc),
        "max_abs_token_weight": float(np.max(np.abs(best_w))),
        "bias": float(best_b),
        "embed_norm_mean": float(np.mean(np.linalg.norm(embed_weight, axis=1))),
    }

    return {"weights": w, "diagnostics": diagnostics}
