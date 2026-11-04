def edit(w, f):
    """
    Closed‑form ridge‑regression update of the final head weights (and bias)
    using the entire permitted feedback pool.
    
    Parameters
    ----------
    w : dict
        Full current float32 parameter dictionary (mutable).
    f : dict
        Read‑only feedback arrays with the exact keys/shapes listed in the prompt.
    
    Returns
    -------
    {"weights": w, "diagnostics": diag}
        Updated weight dictionary and a small JSON‑compatible statistics dict.
    """
    import numpy as np

    # ----- data -------------------------------------------------
    X = f["features"]                     # shape (4000, 32), float32
    y = f["y"]                            # shape (4000,), int64

    # one‑hot encode labels → (4000, 2)
    Y = np.zeros((X.shape[0], 2), dtype=np.float32)
    Y[np.arange(X.shape[0]), y] = 1.0

    # ----- ridge regression ------------------------------------
    lam = 1e-3                            # ridge penalty (λ)
    d = X.shape[1]                        # feature dimension = 32

    XtX = X.T @ X                         # (32, 32)
    XtX_reg = XtX + lam * np.eye(d)       # regularised Gram matrix

    XtY = X.T @ Y                         # (32, 2)

    # solve normal equations (W = (XᵀX + λI)⁻¹ XᵀY)
    W = np.linalg.solve(XtX_reg, XtY).astype(np.float32)   # (32, 2)

    # ----- bias -------------------------------------------------
    # residual = Y - X @ W, then average over the pool
    residual = Y - X @ W                  # (4000, 2)
    bias = np.mean(residual, axis=0).astype(np.float32)    # (2,)

    # ----- update weights ---------------------------------------
    w["head.weight"] = W.T.astype(np.float32)   # head.weight shape (2, 32)
    w["head.bias"]   = bias                     # head.bias shape (2,)

    # ----- diagnostics -----------------------------------------
    # new logits / probabilities on the same feedback features
    logits = X @ w["head.weight"].T + w["head.bias"]   # (4000, 2)
    # numerically stable softmax
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    pred = np.argmax(probs, axis=1)            # (4000,)
    acc  = (pred == y).mean()
    # cross‑entropy loss (mean over samples)
    ce   = -np.mean(np.log(probs[np.arange(X.shape[0]), y] + 1e-12))

    diag = {
        "ridge_lambda": float(lam),
        "head_weight_norm": float(np.linalg.norm(w["head.weight"])),
        "head_bias": w["head.bias"].tolist(),
        "feedback_accuracy": float(acc),
        "feedback_cross_entropy": float(ce),
        "n_feedback": int(X.shape[0]),
    }

    return {"weights": w, "diagnostics": diag}
