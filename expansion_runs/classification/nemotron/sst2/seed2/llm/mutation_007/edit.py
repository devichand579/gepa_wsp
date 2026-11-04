def edit(w, f):
    """
    Edit the network's head weights via a closed‑form ridge regression
    on the provided feedback pool.  Only head.weight and head.bias are
    changed; all other parameters stay exactly as they were.
    """
    import numpy as np  # np is already available in the outer scope

    # ----- feedback data -------------------------------------------------
    features = f["features"]                 # shape (4000, 32), float32
    y        = f["y"]                        # shape (4000,), int64
    # one‑hot encode labels
    Y = np.eye(2, dtype=np.float32)[y]      # shape (4000, 2)

    # ----- ridge‑regression parameters ----------------------------------
    lam = np.float32(1e-3)                   # small ridge penalty

    # ----- compute ridge solution for head.weight -----------------------
    # Gram matrix (32×32) + λI
    Gram = features.T @ features + lam * np.eye(32, dtype=np.float32)
    # cross‑covariance (32×2)
    XTY = features.T @ Y
    # solve for weight matrix shaped (32,2); we will transpose to (2,32)
    W_ridge = np.linalg.solve(Gram, XTY)    # shape (32,2)

    # keep a copy of the original head parameters for diagnostics
    old_weight = w["head.weight"].copy()
    old_bias   = w["head.bias"].copy()
    old_weight_norm = float(np.linalg.norm(old_weight))
    old_bias_norm   = float(np.linalg.norm(old_bias))

    # ----- apply new head.weight -----------------------------------------
    # desired shape (2,32)
    new_weight = W_ridge.T.astype(np.float32)
    w["head.weight"] = new_weight
    new_weight_norm = float(np.linalg.norm(new_weight))

    # ----- compute and apply new head.bias ---------------------------------
    # residual Y - X @ W^T
    resid = Y - features @ new_weight.T   # shape (4000,2)
    b_ridge = np.mean(resid, axis=0).astype(np.float32)   # shape (2,)
    w["head.bias"] = (old_bias + b_ridge).astype(np.float32)
    new_bias_norm = float(np.linalg.norm(w["head.bias"]))

    # ----- diagnostics ----------------------------------------------------
    # accuracy before any change, using the supplied probabilities
    pred_before = f["probs"].argmax(axis=1)
    acc_before  = float((pred_before == y).mean())

    diag = {
        "ridge_lambda": float(lam),
        "old_head_weight_norm": old_weight_norm,
        "new_head_weight_norm": new_weight_norm,
        "old_head_bias": old_bias.tolist(),
        "new_head_bias": w["head.bias"].tolist(),
        "feedback_accuracy_before": acc_before,
    }

    # return modified weights and a small JSON‑compatible diagnostics dict
    return {"weights": w, "diagnostics": diag}
