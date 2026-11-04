def edit(w, f):
    """
    Edit the network's output‑layer parameters using a single closed‑form
    ridge‑regression step on the full feedback pool.  Only `fc2.weight` and
    `fc2.bias` are modified; `fc1` weights are left untouched.
    """
    import numpy as np

    # ----- feedback pool -------------------------------------------------
    features = f["features"]                # shape (2000, 128), float32
    y        = f["y"]                       # shape (2000,), int64
    N        = features.shape[0]

    # one‑hot encode the integer labels
    Y_onehot = np.eye(10, dtype=np.float32)[y]   # (2000, 10)

    # augment features with a constant 1 for the bias term
    X = np.hstack([features, np.ones((N, 1), dtype=np.float32)])   # (2000, 129)

    # ridge‑regression hyper‑parameter (small constant, no search)
    lam = 1e-3

    # normal‑equation matrices
    A = X.T @ X + lam * np.eye(129, dtype=np.float32)   # (129, 129)
    B = X.T @ Y_onehot                                 # (129, 10)

    # closed‑form solution (no iterative optimizer, no autograd)
    W_aug = np.linalg.solve(A, B)      # (129, 10)

    # split into weight matrix (10, 128) and bias vector (10)
    # W_aug[:128, :] corresponds to the 128 features; its transpose gives fc2.weight
    w["fc2.weight"] = (W_aug[:128, :]).T.astype(np.float32)   # (10, 128)
    w["fc2.bias"]   = W_aug[128, :].astype(np.float32)        # (10,)

    # ----- diagnostics -------------------------------------------------
    # compute held‑out accuracy on the feedback pool with the new weights
    logits = features @ w["fc2.weight"].T + w["fc2.bias"]   # (2000, 10)
    # numerically stable softmax
    max_logit = logits.max(axis=1, keepdims=True)
    exp_logit = np.exp(logits - max_logit)
    probs   = exp_logit / exp_logit.sum(axis=1, keepdims=True)
    pred    = np.argmax(probs, axis=1)
    acc     = float(np.mean(pred == y))

    diag = {
        "feedback_pool_accuracy": acc,
        "fc2_weight_norm":  float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm":    float(np.linalg.norm(w["fc2.bias"])),
        "ridge_lambda":     lam,
    }

    return {"weights": w, "diagnostics": diag}
