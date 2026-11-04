import numpy as np
import math

def edit(w, f):
    # Extract feedback data
    x = f["x"]          # [4000, 64] int64
    y = f["y"]          # [4000] int64
    N, L = x.shape
    V = 2000            # vocabulary size

    # Build normalized bag-of-words features: X[i, t] = count(t in example i) / n_i
    # where n_i is number of non-pad tokens (id != 0)
    X = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        tokens = x[i]
        valid = tokens[tokens != 0]
        n = len(valid)
        if n > 0:
            counts = np.bincount(valid, minlength=V)
            X[i] = counts / n
        else:
            X[i] = 0.0

    # Signed labels: -1 for class 0, +1 for class 1
    y_signed = 2.0 * y.astype(np.float32) - 1.0

    # Augment with bias column
    X_aug = np.hstack([X, np.ones((N, 1), dtype=np.float32)])  # [N, V+1]

    # Ridge regression: (X^T X + lambda I) w = X^T y
    lambda_reg = 1.0
    A = X_aug.T @ X_aug
    A += lambda_reg * np.eye(V + 1, dtype=np.float32)
    b_vec = X_aug.T @ y_signed
    w_aug = np.linalg.solve(A, b_vec)   # [V+1]

    w_tok = w_aug[:V].astype(np.float32)   # per-token weights
    b_bias = float(w_aug[V])

    # Choose head direction v (zero-mean, length 32)
    # We'll use v = [c, -c, 0, ..., 0] with c chosen large enough
    max_w = np.max(np.abs(w_tok))
    c = max_w / 8.0 + 1.0
    v = np.zeros(32, dtype=np.float32)
    v[0] = c
    v[1] = -c
    v_norm_sq = 2.0 * c * c

    # Fixed unit vector u in zero-mean subspace orthogonal to v
    u = np.zeros(32, dtype=np.float32)
    u[2] = 1.0 / math.sqrt(2.0)
    u[3] = -1.0 / math.sqrt(2.0)

    # Build token embeddings e_t such that:
    #   sum(e_t) = 0, ||e_t||^2 = 32, v @ e_t = w_t
    embed_new = np.zeros((V, 32), dtype=np.float32)
    for t in range(V):
        wt = w_tok[t]
        # e_t = (wt / ||v||^2) * v + sqrt(32 - wt^2/||v||^2) * u
        scale_v = wt / v_norm_sq
        rem_sq = 32.0 - (wt * wt) / v_norm_sq
        if rem_sq < 0.0:
            rem_sq = 0.0
        scale_u = math.sqrt(rem_sq)
        embed_new[t] = scale_v * v + scale_u * u

    # Set head weights: head[1] - head[0] = v, bias[1] - bias[0] = b_bias
    head_new = np.zeros((2, 32), dtype=np.float32)
    head_new[0] = -0.5 * v
    head_new[1] =  0.5 * v
    head_bias_new = np.array([-0.5 * b_bias, 0.5 * b_bias], dtype=np.float32)

    # Write back into w
    w["embed.weight"] = embed_new
    w["head.weight"] = head_new
    w["head.bias"] = head_bias_new

    # Optionally zero out positional and block weights to keep them inert
    # (they are already zero, but we ensure they stay zero)
    for key in ["pos.weight"]:
        if key in w:
            w[key] = np.zeros_like(w[key])
    for block in ["block0", "block1"]:
        for sub in ["norm1.weight", "norm1.bias", "qkv.weight", "qkv.bias",
                    "proj.weight", "proj.bias", "norm2.weight", "norm2.bias",
                    "ff1.weight", "ff1.bias", "ff2.weight", "ff2.bias"]:
            key = f"{block}.{sub}"
            if key in w:
                w[key] = np.zeros_like(w[key])
    for key in ["norm.weight", "norm.bias"]:
        if key in w:
            if "weight" in key:
                w[key] = np.ones_like(w[key])
            else:
                w[key] = np.zeros_like(w[key])

    # Diagnostics
    diag = {
        "train_accuracy": float(np.mean((X @ w_tok + b_bias > 0) == (y == 1))),
        "max_abs_w_tok": float(max_w),
        "b_bias": b_bias,
        "c": float(c),
        "v_norm_sq": float(v_norm_sq),
        "n_tokens": V,
        "n_samples": N,
    }
    return {"weights": w, "diagnostics": diag}
