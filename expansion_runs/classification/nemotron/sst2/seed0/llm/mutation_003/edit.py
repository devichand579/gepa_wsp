def edit(w, f):
    """
    Closed‑form ridge‑regression fit of the network head to the full feedback pool.
    Returns the modified weight dictionary and a small diagnostics dict.
    """
    import numpy as np

    # ----- feedback data -------------------------------------------------
    F = f["features"].astype(np.float32)          # shape (4000, 32)
    y = f["y"].astype(np.int64)                  # shape (4000,), values 0/1

    # one‑hot encode labels → (4000, 2)
    Y = np.zeros((len(y), 2), dtype=np.float32)
    Y[np.arange(len(y)), y] = 1.0

    # ridge regularisation (small constant)
    lam = np.float32(1e-3)

    # ----- solve for head weight (32,2) and bias (2,) --------------------
    # A = FᵀF + λI   (32×32)
    A = F.T @ F + lam * np.eye(32, dtype=np.float32)
    # B = FᵀY        (32×2)
    B = F.T @ Y

    # W maps features → logits: logits = F @ W + bias
    W = np.linalg.solve(A, B)          # shape (32, 2)

    # bias = mean residual after the linear part
    bias = np.mean(Y - F @ W, axis=0)  # shape (2,)

    # ----- write updated head parameters ---------------------------------
    # current head.weight shape (2,32) = W.T
    w["head.weight"] = W.T.astype(np.float32)
    # head.bias shape (2,)
    w["head.bias"] = bias.astype(np.float32)

    # ----- diagnostics ---------------------------------------------------
    new_logits = F @ W + bias               # (4000, 2)
    # numerically stable softmax
    exp_logits = np.exp(new_logits - np.max(new_logits, axis=1, keepdims=True))
    new_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    pred = np.argmax(new_logits, axis=1)
    accuracy = float(np.mean(pred == y))

    # cross‑entropy loss
    ce = -float(np.mean(np.log(new_probs[np.arange(len(y)), y])))

    diag = {
        "accuracy": accuracy,
        "cross_entropy": ce,
        "head_weight_norm": float(np.linalg.norm(W.T)),
        "head_bias": bias.tolist(),
        "ridge_lambda": float(lam)
    }

    return {"weights": w, "diagnostics": diag}
