def edit(w, f):
    """
    Closed‑form ridge‑regression fit of fc2 to the full feedback pool.
    No autograd, no iterative optimiser – a single normal‑equation solve.
    """
    # ----- data -----
    features = f["features"]                     # shape (2000, 128), float32
    y        = f["y"]                            # shape (2000,), int64

    # one‑hot labels (2000, 10)
    Y = np.eye(10, dtype=np.float32)[y]

    # ridge regularisation (small constant)
    lam = np.float32(1e-3)
    D   = features.shape[1]                     # 128

    # Gram matrix with ridge term
    G = features.T @ features + lam * np.eye(D, dtype=np.float32)   # (128,128)

    # features.T @ Y  -> (128,10)
    WT = features.T @ Y

    # solve G * W_coef = WT   →  W_coef shape (128,10)
    W_coef = np.linalg.solve(G, WT).astype(np.float32)   # (128,10)

    # fc2 weight expects shape (10,128); transpose
    W2 = W_coef.T   # (10,128)

    # bias: mean of (Y - features @ W2.T) over the batch
    residual = Y - features @ W2.T          # (2000,10)
    b2 = np.mean(residual, axis=0).astype(np.float32)   # (10,)

    # ----- update weights -----
    w["fc2.weight"] = W2
    w["fc2.bias"]   = b2

    # ----- diagnostics on the feedback pool -----
    logits_new = features @ W2.T + b2          # (2000,10)
    pred_new   = np.argmax(logits_new, axis=1)
    acc_new    = float(np.mean(pred_new == y))

    # numerically stable cross‑entropy
    exp_logits = np.exp(logits_new - np.max(logits_new, axis=1, keepdims=True))
    probs_new  = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    ce_new     = float(-np.mean(np.log(probs_new[np.arange(2000), y])))

    diag = {
        "fc2_weight_norm": float(np.linalg.norm(W2)),
        "fc2_bias":       b2.tolist(),
        "feedback_accuracy": acc_new,
        "feedback_cross_entropy": ce_new,
        "current_accuracy":  float(np.mean(np.argmax(f["logits"], axis=1) == y)),
        "current_cross_entropy": float(-np.mean(np.log(f["probs"][np.arange(2000), y]))),
    }

    return {"weights": w, "diagnostics": diag}
