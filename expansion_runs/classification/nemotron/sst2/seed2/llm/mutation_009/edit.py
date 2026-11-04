def edit(w, f):
    """
    Edit the final head weights (and bias) of the network using a
    closed‑form ridge‑regression on the entire feedback pool.
    No autograd, back‑prop, or iterative optimisers are used.
    """
    import numpy as np

    # ----- data -----
    X = f["features"].astype(np.float32)          # shape (4000, 32)
    y = f["y"].astype(np.int64)                   # shape (4000,), 0=neg, 1=pos
    lam = 1e-3                                    # ridge coefficient

    # one‑hot labels (4000, 2)
    Y = np.eye(2, dtype=np.float32)[y]

    # ----- ridge solution for head.weight -----
    XtX = X.T @ X                                 # (32, 32)
    A = XtX + lam * np.eye(32, dtype=np.float32) # regularised Gram matrix
    XtY = X.T @ Y                                 # (32, 2)
    W = np.linalg.solve(A, XtY)                  # (32, 2)  ->  rows = input, cols = output
    new_weight = W.T.astype(np.float32)           # (2, 32) matches head.weight shape

    # ----- bias from residuals -----
    residuals = Y - X @ new_weight.T             # (4000, 2)
    new_bias = np.mean(residuals, axis=0).astype(np.float32)   # (2,)

    # ----- apply edits -----
    w["head.weight"] = new_weight
    w["head.bias"] = new_bias

    # ----- diagnostics (computed on the feedback pool) -----
    logits = X @ new_weight.T + new_bias         # (4000, 2)
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    acc = (probs.argmax(axis=1) == y).mean()
    ce = -np.mean(np.log(probs[np.arange(4000), y]))

    diag = {
        "ridge_lambda": float(lam),
        "head_weight_norm": float(np.linalg.norm(new_weight)),
        "head_bias": new_bias.tolist(),
        "feedback_accuracy": float(acc),
        "feedback_cross_entropy": float(ce),
        "n_feedback": int(X.shape[0])
    }

    return {"weights": w, "diagnostics": diag}
