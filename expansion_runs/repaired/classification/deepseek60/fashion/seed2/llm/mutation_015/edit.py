import numpy as np

def edit(w, f):
    # Read current head and features
    X = f["features"].astype(np.float64)          # (2000, 1568)
    y = f["y"].astype(np.int64)                   # (2000,)
    n, d = X.shape
    C = 10

    # One-hot targets
    Y = np.zeros((n, C), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    # Center features and targets
    X_mean = X.mean(axis=0, keepdims=True)
    Y_mean = Y.mean(axis=0, keepdims=True)
    Xc = X - X_mean
    Yc = Y - Y_mean

    # Ridge regression: W = (Xc^T Xc + lambda I)^-1 Xc^T Yc
    # Choose lambda via Ledoit-Wolf style shrinkage on the covariance.
    # Sample covariance of centered features
    S = (Xc.T @ Xc) / n
    # Ledoit-Wolf shrinkage intensity (simplified, assuming zero mean)
    # We use a small fixed lambda based on the trace to avoid singularity.
    # The features have many zero columns; trace(S) is the sum of variances.
    trace_S = np.trace(S)
    # A reasonable small regularization: lambda = 1e-4 * trace(S) / d
    # This is a closed-form choice, not an iterative search.
    lam = 1e-4 * trace_S / d

    # Solve for W (C x d)
    # Use Cholesky or solve directly. d=1568, C=10.
    # We solve (Xc^T Xc + n*lam I) W^T = Xc^T Yc
    # Actually we want W (C x d) such that Xc @ W.T + Y_mean ≈ Yc
    # Normal equations: (Xc^T Xc + n*lam I) W.T = Xc^T Yc
    A = Xc.T @ Xc + n * lam * np.eye(d)
    B = Xc.T @ Yc          # (d, C)
    W_T = np.linalg.solve(A, B)   # (d, C)
    W_new = W_T.T          # (C, d)
    b_new = (Y_mean - X_mean @ W_new.T).ravel()  # (C,)

    # Evaluate current head on full pool
    W_cur = w["head.weight"].astype(np.float64)
    b_cur = w["head.bias"].astype(np.float64)
    logits_cur = X @ W_cur.T + b_cur
    probs_cur = np.exp(logits_cur - logits_cur.max(axis=1, keepdims=True))
    probs_cur /= probs_cur.sum(axis=1, keepdims=True)
    ce_cur = -np.log(probs_cur[np.arange(n), y] + 1e-12).mean()
    acc_cur = (logits_cur.argmax(axis=1) == y).mean()

    # Evaluate new head
    logits_new = X @ W_new.T + b_new
    probs_new = np.exp(logits_new - logits_new.max(axis=1, keepdims=True))
    probs_new /= probs_new.sum(axis=1, keepdims=True)
    ce_new = -np.log(probs_new[np.arange(n), y] + 1e-12).mean()
    acc_new = (logits_new.argmax(axis=1) == y).mean()

    # Choose the better head (lower CE, tie-break by accuracy)
    if ce_new < ce_cur or (np.isclose(ce_new, ce_cur) and acc_new > acc_cur):
        W_best = W_new
        b_best = b_new
        chosen = "ridge"
    else:
        W_best = W_cur
        b_best = b_cur
        chosen = "current"

    # Rescale to max absolute value 1000
    max_abs = max(np.max(np.abs(W_best)), np.max(np.abs(b_best)))
    if max_abs > 0:
        scale = 1000.0 / max_abs
        W_best = W_best * scale
        b_best = b_best * scale

    # Write back as float32
    w["head.weight"] = W_best.astype(np.float32)
    w["head.bias"] = b_best.astype(np.float32)

    diagnostics = {
        "chosen": chosen,
        "ce_current": float(ce_cur),
        "ce_new": float(ce_new),
        "acc_current": float(acc_cur),
        "acc_new": float(acc_new),
        "max_abs_head_weight": float(np.max(np.abs(w["head.weight"]))),
        "max_abs_head_bias": float(np.max(np.abs(w["head.bias"]))),
        "lambda": float(lam),
    }
    return {"weights": w, "diagnostics": diagnostics}
