def edit(w, f):
    """
    One‑shot feedback‑assisted construction.
    Updates the network's head (weight + bias) via ridge regression on
    the full feedback pool (features + labels).  No autograd, no loops,
    only linear algebra.
    """
    import numpy as np

    # ----- data -------------------------------------------------
    X = f["features"].astype(np.float32)            # shape (4000, 32)
    y = f["y"].astype(np.int64)                     # shape (4000,)

    # one‑hot targets for the two classes
    y_onehot = np.eye(2, dtype=np.float32)[y]       # shape (4000, 2)

    # ----- ridge regression (closed‑form) ----------------------
    lam = 1e-3                                      # small Tikhonov regulariser
    A = X.T @ X + lam * np.eye(32, dtype=np.float32)   # (32,32)
    B = X.T @ y_onehot                                 # (32,2)

    # solve A * W.T = B  =>  W.T = A^{-1} B
    Wt = np.linalg.solve(A, B)          # (32,2)
    head_weight = Wt.T.astype(np.float32)   # (2,32)  to match original shape

    # bias: mean residual after the optimal linear projection
    residuals = y_onehot - X @ head_weight.T   # (4000,2)
    head_bias = np.mean(residuals, axis=0).astype(np.float32)   # (2,)

    # ----- apply updates, preserve shapes -----------------------
    w["head.weight"][:] = head_weight
    w["head.bias"][:]   = head_bias

    # ----- diagnostics -------------------------------------------
    # current (pre‑edit) accuracy on the feedback pool
    cur_logits = f["logits"]                       # (4000,2) already available
    cur_pred   = np.argmax(cur_logits, axis=1)
    before_acc  = (cur_pred == y).mean()

    # forward‑pass with the new head on the same features
    new_logits = X @ w["head.weight"].T + w["head.bias"]   # (4000,2)
    # numerically stable soft‑max
    exp_logits = np.exp(new_logits - new_logits.max(axis=1, keepdims=True))
    new_probs  = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    new_pred   = np.argmax(new_probs, axis=1)
    after_acc   = (new_pred == y).mean()

    diag = {
        "before_accuracy": float(before_acc),
        "after_accuracy":  float(after_acc),
        "weight_delta_norm": float(np.linalg.norm(head_weight - w["head.weight"][:] if False else head_weight - w["head.weight"][:] if False else 0.0)),  # placeholder; will be filled by caller
        "lambda": float(lam)
    }

    # return the (now modified) weight dict together with a tiny stats block
    return {"weights": w, "diagnostics": diag}
