def edit(w, f):
    """
    Closed‑form ridge‑regression training of the `head` module using the
    full permitted feedback pool.  All other tensors (conv1, conv2) are left
    unchanged.
    """
    import numpy as np

    # ----- data from feedback pool -----
    X = f["features"]                # shape (2000, 1568)
    y = f["y"]                       # shape (2000,), int labels 0‑9

    # one‑hot encode labels
    y_onehot = np.eye(10, dtype=np.float32)[y]   # (2000, 10)

    # ridge regularisation (small constant)
    lam = 1e-3

    # augment features with a constant 1 for the bias term
    X_aug = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float32)])  # (2000, 1569)

    # normal equations: (XᵀX + λI) w = Xᵀ Y
    A = X_aug.T @ X_aug + lam * np.eye(X_aug.shape[1], dtype=np.float32)   # (1569, 1569)
    b = X_aug.T @ y_onehot                                                    # (1569, 10)

    # closed‑form solution (pseudo‑inverse for stability)
    w_head = np.linalg.pinv(A) @ b   # (1569, 10)

    # split into weight matrix and bias vector
    # w_head columns = [weight_vector(1568), bias]
    new_weight = w_head[:1568].T.astype(np.float32)   # (10, 1568)
    new_bias   = w_head[1568].astype(np.float32)     # (10,)

    # ----- edit the network weights -----
    w["head.weight"] = new_weight
    w["head.bias"]   = new_bias

    # ----- diagnostics -----
    # predictions on the feedback pool
    logits = X @ new_weight.T + new_bias          # (2000, 10)
    # numerically stable softmax
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    pred = np.argmax(probs, axis=1)
    accuracy = float(np.mean(pred == y))
    ce = float(-np.mean(np.log(probs[np.arange(2000), y] + 1e-12)))

    diagnostics = {
        "head_weight_mean": float(np.mean(new_weight)),
        "head_weight_std":  float(np.std(new_weight)),
        "head_bias_mean":   float(np.mean(new_bias)),
        "feedback_accuracy": accuracy,
        "feedback_cross_entropy": ce,
        "full_pool_accuracy_before": 0.9995   # from context
    }

    return {"weights": w, "diagnostics": diagnostics}
