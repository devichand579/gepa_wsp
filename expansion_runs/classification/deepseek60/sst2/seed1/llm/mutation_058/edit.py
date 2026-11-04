import numpy as np

def edit(w, f):
    # Extract training data
    x = f["x"]          # [4000, 64] int64
    y = f["y"]          # [4000] int64
    N, L = x.shape
    V = w["embed.weight"].shape[0]  # 2000
    D = w["embed.weight"].shape[1]  # 32

    # Build bag-of-words counts (excluding padding token 0)
    # X[i, t] = count of token t in document i
    X = np.zeros((N, V), dtype=np.float64)
    for i in range(N):
        tokens = x[i]
        valid = tokens != 0
        if valid.sum() > 0:
            # Count occurrences
            counts = np.bincount(tokens[valid], minlength=V)
            X[i] = counts / valid.sum()  # normalize by length

    # Ridge regression on ±1 targets
    # Target: +1 for positive (y=1), -1 for negative (y=0)
    t = np.where(y == 1, 1.0, -1.0)
    # Add bias column
    X_aug = np.hstack([X, np.ones((N, 1))])
    # Ridge with small lambda
    lam = 1.0
    A = X_aug.T @ X_aug + lam * np.eye(V + 1)
    b_vec = X_aug.T @ t
    # Solve normal equations
    coef = np.linalg.solve(A, b_vec)
    s = coef[:V]        # per-token weights
    b = coef[V]         # bias

    # Clip s to avoid exceeding sqrt(32) in construction
    max_s = np.sqrt(32.0) - 1e-3
    s = np.clip(s, -max_s, max_s)

    # Construct orthonormal mean-zero vectors u and w
    # u: mean-zero unit vector
    u = np.ones(D, dtype=np.float64)
    u -= u.mean()
    u /= np.linalg.norm(u)
    # w: mean-zero unit vector orthogonal to u
    w = np.random.RandomState(42).randn(D)
    w -= w.mean()
    w -= (w @ u) * u
    w /= np.linalg.norm(w)

    # Build v_t for each token
    v = np.zeros((V, D), dtype=np.float64)
    for t_idx in range(V):
        st = s[t_idx]
        # v_t = st * u + sqrt(32 - st^2) * w
        v[t_idx] = st * u + np.sqrt(32.0 - st * st) * w

    # Set embed.weight to v (since LN(v) = v because mean=0, var=1)
    w["embed.weight"][:] = v.astype(np.float32)

    # Set head to read u direction
    # logit difference = u @ pooled + b
    # head[1] - head[0] = u, head.bias[1] - head.bias[0] = b
    head_w = np.zeros((2, D), dtype=np.float64)
    head_w[1] = u / 2.0
    head_w[0] = -u / 2.0
    w["head.weight"][:] = head_w.astype(np.float32)

    head_b = np.zeros(2, dtype=np.float64)
    head_b[1] = b / 2.0
    head_b[0] = -b / 2.0
    w["head.bias"][:] = head_b.astype(np.float32)

    # Zero out pos and all block weights to keep degenerate structure
    for key in w:
        if key.startswith("pos.") or key.startswith("block0.") or key.startswith("block1."):
            w[key][:] = 0.0
        elif key.endswith("norm1.weight") or key.endswith("norm2.weight") or key == "norm.weight":
            w[key][:] = 1.0
        elif key.endswith("norm1.bias") or key.endswith("norm2.bias") or key == "norm.bias":
            w[key][:] = 0.0

    # Diagnostics
    diagnostics = {
        "method": "closed_form_ridge_bag_of_words",
        "lambda": lam,
        "s_absmax": float(np.max(np.abs(s))),
        "bias": float(b),
        "head_w_norm": float(np.linalg.norm(head_w[1] - head_w[0])),
        "embed_row_std": float(np.std(w["embed.weight"])),
        "embed_row_mean": float(np.mean(w["embed.weight"])),
    }

    return {"weights": w, "diagnostics": diagnostics}
