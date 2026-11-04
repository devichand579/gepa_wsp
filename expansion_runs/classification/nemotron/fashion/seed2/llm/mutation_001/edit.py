def edit(w, f):
    """
    Closed‑form ridge‑fit of the network head to the full training feedback.
    Parameters
    ----------
    w : dict
        Current full parameter dictionary (mutable).
    f : dict
        Feedback arrays as supplied by the environment.
    Returns
    -------
    {"weights": w, "diagnostics": {...}}
    """
    import numpy as np

    # ----- data -----
    X = f["features"]                     # (2000, 1568) float32
    y = f["y"]                            # (2000,) int64

    # one‑hot labels (2000, 10)
    Y = np.eye(10, dtype=np.float32)[y]

    # ----- ridge regularisation -----
    lam = np.float32(1e-3)                # small Tikhonov term
    D = X.shape[1]                        # 1568

    Gram = X.T @ X + lam * np.eye(D, dtype=np.float32)   # (1568,1568)

    # closed‑form solution: (XᵀX + λI)⁻¹ XᵀY  →  (1568,10)
    coeff = np.linalg.solve(Gram, X.T @ Y)   # (1568,10)

    # head weight = coeffᵀ  →  (10,1568)  matching the original layout
    new_weight = coeff.T.astype(np.float32)

    # bias: average residual (one‑hot – logits) over the training set
    logits = X @ new_weight.T                # (2000,10)
    new_bias = np.mean(Y.astype(np.float32) - logits, axis=0).astype(np.float32)

    # ----- edit the weights -----
    w["head.weight"] = new_weight
    w["head.bias"] = new_bias

    # ----- diagnostics (training‑set metrics) -----
    preds = np.argmax(logits, axis=1)
    train_acc = float(np.mean(preds == y))

    # numerically stable softmax + cross‑entropy
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    ce = -float(np.mean(np.log(probs[np.arange(2000), y] + 1e-12)))

    diag = {
        "train_accuracy": train_acc,
        "train_cross_entropy": ce,
        "head_weight_norm": float(np.linalg.norm(new_weight)),
        "head_bias": new_bias.tolist(),
        "ridge_lambda": float(lam)
    }

    return {"weights": w, "diagnostics": diag}
