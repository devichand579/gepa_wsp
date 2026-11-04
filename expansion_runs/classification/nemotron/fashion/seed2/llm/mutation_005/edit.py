def edit(w, f):
    """
    Edit the network head via a closed‑form ridge regression on the full
    permitted feedback pool.  Only NumPy operations are used; no autograd,
    back‑propagation, or iterative optimisers.
    """
    import numpy as np

    # --- feedback data -------------------------------------------------
    X = f["features"]                 # shape (2000, 1568)
    y = f["y"]                        # shape (2000,), int64 labels 0‑9
    N = X.shape[0]

    # one‑hot encode labels
    Y = np.eye(10, dtype=np.float32)[y]   # (2000, 10)

    # --- ridge‑regression (closed‑form) -------------------------------
    lam = 1e-3                     # small ridge coefficient for stability
    I = np.eye(X.shape[1], dtype=np.float32)

    # normal equation: (XᵀX + λI) w = Xᵀ Y
    w_head = np.linalg.solve(X.T @ X + lam * I, X.T @ Y)   # (10, 1568)

    # bias: mean residual after applying the weight matrix
    b_head = np.mean(Y - X @ w_head, axis=0)               # (10,)

    # --- replace head parameters ---------------------------------------
    w["head.weight"] = w_head.astype(np.float32)           # keep shape (10,1568)
    w["head.bias"]   = b_head.astype(np.float32)           # keep shape (10,)

    # --- diagnostics ---------------------------------------------------
    logits = X @ w_head.T + b_head[np.newaxis, :]          # (2000, 10)
    # numerically stable softmax
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    pred = np.argmax(probs, axis=1)
    acc = float(np.mean(pred == y))

    ce = float(-np.mean(np.log(probs[np.arange(N), y])))

    diagnostics = {
        "head_weight_norm": float(np.linalg.norm(w_head)),
        "head_bias_mean":   float(np.mean(b_head)),
        "train_accuracy":   acc,
        "train_cross_entropy": ce
    }

    return {"weights": w, "diagnostics": diagnostics}
