def edit(w, f):
    """
    Closed‑form ridge‑regression fit of the network head
    using the entire permitted feedback pool (f["features"], f["y"]).
    All other parameters are left unchanged.
    """
    import numpy as np

    # ------- data ----------
    X = f["features"].astype(np.float32)                # (2000, 1568)
    y = f["y"].astype(np.int64)                         # (2000,)
    # one‑hot targets
    Y = np.eye(10, dtype=np.float32)[y]                 # (2000, 10)

    # ------- ridge hyper‑parameter ----------
    lam = np.float32(1e-3)                              # small ridge

    # ------- centred / normal equations ----------
    # A = Y^T @ X   (10, 1568)
    A = Y.T @ X

    # Gram = X^T @ X + λ I   (1568, 1568)
    Gram = X.T @ X + lam * np.eye(1568, dtype=np.float32)

    # inverse (Gram is symmetric positive‑definite after ridge)
    Gram_inv = np.linalg.inv(Gram)

    # new head weight: (10, 1568)  – matches original shape
    new_W = (A @ Gram_inv).astype(np.float32)

    # new head bias: b = mean(Y) – W @ mean(X)
    mean_Y = Y.mean(axis=0, keepdims=True)            # (1, 10)
    mean_X = X.mean(axis=0, keepdims=True)            # (1, 1568)
    new_b = (mean_Y - new_W @ mean_X.T).T.astype(np.float32)  # (10,)

    # ------- apply edits directly to w ----------
    w["head.weight"] = new_W
    w["head.bias"]   = new_b

    # ------- optional diagnostics ----------
    # compute new logits and cross‑entropy on the full pool
    logits = X @ new_W.T + new_b[np.newaxis, :]        # (2000, 10)
    # numerically stable softmax
    logits_max = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    # cross‑entropy (average over samples)
    ce = -np.mean(np.sum(Y * np.log(probs + 1e-30), axis=1))

    diag = {
        "ridge_lambda": float(lam),
        "head_weight_norm": float(np.linalg.norm(new_W)),
        "head_bias": new_b.tolist(),
        "train_cross_entropy": float(ce),
        "train_accuracy_top1": float(np.mean(np.argmax(probs, axis=1) == y))
    }

    # return the edited weight dict together with a tiny diagnostics block
    return {"weights": w, "diagnostics": diag}
