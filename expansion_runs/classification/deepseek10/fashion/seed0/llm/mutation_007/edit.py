import numpy as np

def edit(w, f):
    # Extract training features and labels
    X = f["features"].astype(np.float64)   # [2000, 1568]
    y = f["y"].astype(np.int64)            # [2000]
    n, d = X.shape
    classes = np.unique(y)
    C = len(classes)                       # 10

    # Global mean
    mu_global = X.mean(axis=0)

    # Class means and counts
    mus = np.zeros((C, d), dtype=np.float64)
    counts = np.zeros(C, dtype=np.float64)
    for idx, c in enumerate(classes):
        mask = (y == c)
        counts[idx] = mask.sum()
        if counts[idx] > 0:
            mus[idx] = X[mask].mean(axis=0)

    # Pooled within-class scatter matrix Sw
    # Sw = sum_c sum_{i in c} (x_i - mu_c)(x_i - mu_c)^T
    # Compute efficiently: Sw = X^T X - sum_c n_c mu_c mu_c^T
    XtX = X.T @ X
    Sw = XtX.copy()
    for idx in range(C):
        if counts[idx] > 0:
            Sw -= counts[idx] * np.outer(mus[idx], mus[idx])

    # Regularization: ridge term. Choose a small value relative to trace.
    # We use a fixed small ridge because cross-validation would be expensive.
    # The trace of Sw is roughly n * average feature variance.
    # A ridge of 1e-3 * trace(Sw)/d is a mild shrinkage.
    trace_Sw = np.trace(Sw)
    lam = 1e-3 * trace_Sw / d
    if lam <= 0:
        lam = 1e-6
    Sw_reg = Sw + lam * np.eye(d, dtype=np.float64)

    # Solve for LDA weights: W = Sw_reg^{-1} (mu_c - mu_global)
    # We can solve linear system instead of explicit inverse.
    # RHS: [d, C]
    RHS = (mus - mu_global).T   # [d, C]
    # Solve Sw_reg * W = RHS
    W = np.linalg.solve(Sw_reg, RHS)   # [d, C]

    # Bias: b_c = - W_c^T mu_global
    b = - (W.T @ mu_global)            # [C]

    # Set head weights and bias. head.weight is [10, 1568], head.bias is [10]
    # W is [1568, 10], so head.weight = W.T
    w["head.weight"] = W.T.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

    # Ensure finite and within bounds
    for k in w:
        v = w[k]
        if not np.isfinite(v).all():
            v = np.nan_to_num(v, nan=0.0, posinf=1000.0, neginf=-1000.0)
        v = np.clip(v, -1000.0, 1000.0)
        w[k] = v.astype(np.float32)

    diagnostics = {
        "ridge_lambda": float(lam),
        "head_weight_max_abs": float(np.max(np.abs(w["head.weight"]))),
        "head_bias_max_abs": float(np.max(np.abs(w["head.bias"]))),
        "trace_Sw": float(trace_Sw),
    }
    return {"weights": w, "diagnostics": diagnostics}
