def edit(w, f):
    """
    Closed‑form ridge‑regression update of the final head weights
    using the entire feedback pool.
    """
    import numpy as np

    # ----- data from feedback -----
    X = f["features"].astype(np.float32)                 # [4000, 32]
    y = f["y"].astype(np.int64)                         # [4000]

    # one‑hot labels
    Y = np.zeros((X.shape[0], 2), dtype=np.float32)
    Y[np.arange(X.shape[0]), y] = 1.0

    # ridge parameter
    lam = np.float32(1e-3)

    # ----- closed‑form solution for head weights -----
    # XᵀX + λI   (32×32)
    XtX = X.T @ X
    A   = XtX + lam * np.eye(32, dtype=np.float32)

    # XᵀY        (32×2)
    XtY = X.T @ Y

    # solve A * W = XtY  →  W is [32, 2]
    W32_2 = np.linalg.solve(A, XtY).astype(np.float32)   # [32,2]

    # head.weight is [2,32] → transpose
    new_head_weight = W32_2.T.astype(np.float32)          # [2,32]

    # ----- head bias -----
    # bias = mean( Y - X @ W )  (per class)
    resid = Y - (X @ W32_2)                             # [4000,2]
    new_head_bias = np.mean(resid, axis=0).astype(np.float32)  # [2]

    # ----- apply updates -----
    w["head.weight"] = new_head_weight
    w["head.bias"]   = new_head_bias

    # ----- diagnostics -----
    logits = X @ w["head.weight"].T + w["head.bias"]       # [4000,2]
    # numerically stable softmax
    logits_max = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    pred = probs.argmax(axis=1)
    accuracy = (pred == y).mean()
    # cross‑entropy loss (average over samples)
    ce = -np.mean(np.log(probs[np.arange(X.shape[0]), y] + 1e-12))

    diag = {
        "head_weight_norm": float(np.linalg.norm(new_head_weight)),
        "head_bias":      new_head_bias.tolist(),
        "feedback_accuracy": float(accuracy),
        "feedback_cross_entropy": float(ce),
        "n_feedback": X.shape[0],
    }

    return {"weights": w, "diagnostics": diag}
