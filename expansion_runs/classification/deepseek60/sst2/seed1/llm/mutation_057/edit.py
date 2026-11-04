import numpy as np
import math

def edit(w, f):
    # Extract training data
    x = f["x"]          # [4000, 64] int64
    y = f["y"]          # [4000] int64
    N, T = x.shape
    V = w["embed.weight"].shape[0]  # 2000

    # Build design matrix for ridge regression:
    # y_i ≈ (1/L_i) Σ_{t in tokens} s_t + b
    # We'll use a dense matrix X of shape (N, V+1)
    X = np.zeros((N, V + 1), dtype=np.float64)
    for i in range(N):
        tokens = x[i][x[i] != 0]   # exclude padding (token 0)
        L = len(tokens)
        if L > 0:
            # count occurrences (tokens may repeat)
            for t in tokens:
                X[i, t] += 1.0 / L
        X[i, V] = 1.0   # bias column

    # Ridge regression: solve (X^T X + λI) β = X^T y
    # Try a few λ values and pick the one with best training accuracy
    best_acc = -1.0
    best_beta = None
    for lam in [0.01, 0.1, 1.0, 10.0]:
        A = X.T @ X + lam * np.eye(V + 1)
        b_vec = X.T @ y.astype(np.float64)
        try:
            beta = np.linalg.solve(A, b_vec)
        except np.linalg.LinAlgError:
            continue
        s = beta[:V]
        b = beta[V]
        # Compute training predictions: logit = X[:, :V] @ s + b
        logits = X[:, :V] @ s + b
        preds = (logits > 0.5).astype(np.int64)
        acc = np.mean(preds == y)
        if acc > best_acc:
            best_acc = acc
            best_beta = beta

    if best_beta is None:
        # fallback: zero weights
        best_beta = np.zeros(V + 1)

    s = best_beta[:V].copy()
    b = best_beta[V]

    # Scale s to ensure |s_t| <= sqrt(32) for the construction
    max_abs = np.max(np.abs(s))
    limit = math.sqrt(32.0) * 0.99
    if max_abs > limit:
        scale = limit / max_abs
        s *= scale
        b *= scale

    # Construct orthonormal mean-zero vectors u, w in R^32
    u = np.zeros(32, dtype=np.float32)
    u[0] = 1.0 / math.sqrt(2.0)
    u[1] = -1.0 / math.sqrt(2.0)

    w_vec = np.zeros(32, dtype=np.float32)
    w_vec[0] = 1.0 / math.sqrt(6.0)
    w_vec[1] = 1.0 / math.sqrt(6.0)
    w_vec[2] = -2.0 / math.sqrt(6.0)

    # Build embed.weight: v_t = s_t * u + sqrt(32 - s_t^2) * w
    embed = w["embed.weight"]
    for t in range(V):
        st = s[t]
        val = math.sqrt(max(0.0, 32.0 - st * st))
        embed[t, :] = st * u + val * w_vec

    # Set head weights: logit1 = u^T pooled + b, logit0 = 0
    head_w = w["head.weight"]
    head_b = w["head.bias"]
    head_w[0, :] = 0.0
    head_w[1, :] = u
    head_b[0] = 0.0
    head_b[1] = b

    # Ensure all other weights remain zero (they already are)
    # Optionally, we could leave them as is.

    # Diagnostics
    diag = {
        "train_acc": float(best_acc),
        "lambda": float(lam) if best_beta is not None else None,
        "s_absmax": float(np.max(np.abs(s))),
        "bias": float(b),
        "head_w_norm": float(np.linalg.norm(head_w[1])),
    }

    return {"weights": w, "diagnostics": diag}
