def edit(w, f):
    """
    Closed‑form ridge‑regression update of the classification head
    using the entire permitted feedback pool (features + labels).
    No autograd, back‑prop, or iterative optimisers are used.
    """
    import numpy as np

    # ------- data from feedback dict -------
    X = f["features"].astype(np.float32)          # (4000, 32)
    y = f["y"].astype(np.int64)                   # (4000,)
    # one‑hot encode labels into [0,1] log‑space targets
    Y = np.eye(2, dtype=np.float32)[y]            # (4000, 2)

    # ------- ridge regularisation -------
    lam = 1e-3                                    # small Tikhonov term
    I32 = np.eye(32, dtype=np.float32) * lam
    XtX = X.T @ X                                 # (32,32)
    XtY = X.T @ Y                                 # (32,2)

    # solve (XᵀX + λI) W = XᵀY  in closed form
    coeff = np.linalg.solve(XtX + I32, XtY)       # (32,2)  ->  Wᵀ
    W = coeff.T                                   # (2,32) matches head.weight shape

    # bias = mean residual after linear projection
    pred_logits = X @ coeff                       # (4000,2)
    b = np.mean(Y - pred_logits, axis=0)          # (2,)

    # ------- write updated head parameters -------
    w["head.weight"] = W.astype(np.float32)       # shape [2,32]
    w["head.bias"]   = b.astype(np.float32)       # shape [2]

    # ------- diagnostics (metrics on the feedback pool) -------
    logits = X @ W.T + b                         # (4000,2)
    # numerically stable softmax
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    pred = np.argmax(probs, axis=1)
    acc = float(np.mean(pred == y))
    # cross‑entropy loss
    ce = -float(np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12)))

    diag = {
        "head_weight_mean":  float(np.mean(W)),
        "head_weight_std":   float(np.std(W)),
        "head_bias_mean":    float(np.mean(b)),
        "accuracy":          acc,
        "cross_entropy":     ce,
        "ridge_lambda":      lam
    }

    return {"weights": w, "diagnostics": diag}
