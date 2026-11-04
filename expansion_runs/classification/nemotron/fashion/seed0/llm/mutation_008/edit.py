def edit(w, f):
    """
    Closed‑form ridge‑regression training of the `head` module on the full
    feedback pool.  Conv1 and Conv2 weights are left unchanged.
    """
    import numpy as np

    # ----- data from feedback arrays -----
    X = f["features"]                     # shape (2000, 1568), float32
    y = f["y"]                            # shape (2000,), int64

    # one‑hot labels (2000, 10)
    Y = np.eye(10, dtype=np.float32)[y]

    # ----- ridge regularisation -----
    lam = 1e-3                            # small ridge term

    # augment features with a bias column (ones)
    X_aug = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float32)])  # (2000, 1569)

    # normal equations with ridge: (XᵀX + λI) β = XᵀY
    Gram = X_aug.T @ X_aug + lam * np.eye(1569, dtype=np.float32)       # (1569, 1569)
    rhs  = X_aug.T @ Y                                              # (1569, 10)

    # solve for augmented coefficients [W, b]  (1569, 10)
    beta = np.linalg.solve(Gram, rhs)                               # (1569, 10)

    # split into weight matrix and bias
    # beta[:1568, :]  -> head.weight  (1568, 10); we need shape (10, 1568)
    new_head_weight = beta[:1568, :].T                               # (10, 1568)
    new_head_bias   = beta[1568, :]                                   # (10,)

    # ----- edit the weight dictionary in‑place -----
    w["head.weight"] = new_head_weight.astype(np.float32)
    w["head.bias"]   = new_head_bias.astype(np.float32)

    # ----- diagnostics -----
    # compute fitted logits / probabilities on the feedback pool
    logits = X @ new_head_weight.T + new_head_bias[np.newaxis, :]      # (2000, 10)
    probs  = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)  # softmax
    pred   = np.argmax(probs, axis=1)
    acc    = np.mean(pred == y)

    diag = {
        "head_weight_mean": float(new_head_weight.mean()),
        "head_weight_std":  float(new_head_weight.std()),
        "head_bias_mean":   float(new_head_bias.mean()),
        "feedback_accuracy": float(acc),
        "feedback_cross_entropy": float(-np.mean(np.log(probs[np.arange(2000), y])))
    }

    return {"weights": w, "diagnostics": diag}
