def edit(w, f):
    """
    Closed‑form ridge‑regression update of the head weights using the full feedback pool.
    Only head.weight and head.bias are modified; everything else stays exactly as supplied.
    """
    import numpy as np

    # ----- feedback data -----
    X = f["features"].astype(np.float32)          # shape (4000, 32)
    y = f["y"].astype(np.int32)                   # shape (4000,)
    # binary labels → +1 (positive) / -1 (negative)
    t = (2.0 * y - 1.0).astype(np.float32)       # shape (4000,)

    # ridge hyper‑parameter
    lam = 1e-3

    # augment features with a constant 1 column for the bias term
    X_aug = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float32)])  # (4000, 33)

    # normal equations: (XᵀX + λI) w_aug = Xᵀ t
    I = np.eye(X_aug.shape[1], dtype=np.float32)
    A = X_aug.T @ X_aug + lam * I
    rhs = X_aug.T @ t
    w_aug = np.linalg.solve(A, rhs)             # shape (33,)  [w_part; b_part]

    w_part = w_aug[:32].astype(np.float32)      # 32‑dim weight vector
    b_part = w_aug[32].astype(np.float32)       # scalar bias

    # ----- set head parameters -----
    # head.weight shape (2,32); we give both output units opposite signs so that
    # logits are (w·x + b) and -(w·x + b), producing a proper soft‑max probability.
    head_weight = np.zeros((2, 32), dtype=np.float32)
    head_weight[0] = w_part
    head_weight[1] = -w_part

    head_bias = np.array([b_part, -b_part], dtype=np.float32)

    # copy into the weight dictionary (preserve names & shapes)
    w["head.weight"] = head_weight
    w["head.bias"] = head_bias

    # ----- diagnostics (optional, for later reflection) -----
    # compute logits and accuracy on the feedback pool
    logits_feedback = X @ head_weight.T + head_bias[np.newaxis, :]   # (4000,2)
    pred = np.argmax(logits_feedback, axis=1)                      # 0 or 1
    acc = float(np.mean(pred == y))

    diag = {
        "ridge_lambda": float(lam),
        "head_weight_norm": float(np.linalg.norm(w_part)),
        "head_bias": [float(b_part), float(-b_part)],
        "feedback_accuracy": acc,
        "n_feedback": int(X.shape[0])
    }

    return {"weights": w, "diagnostics": diag}
