def edit(w, f):
    """
    Edit the network's head weights via a closed‑form ridge regression
    from the provided frozen features to the training labels.
    Returns the modified weight dict and a small diagnostics dict.
    """
    # ----- data from feedback arrays -----
    X = f["features"]                # (2000, 1568)  pre‑edit hidden activations
    y = f["y"]                      # (2000,)       integer labels

    # one‑hot encode labels → (2000, 10)
    Y = np.eye(10, dtype=np.float32)[y]

    # ridge regularisation (small λ stabilises the solve)
    lam = 1e-3
    I = np.eye(X.shape[1], dtype=np.float32)

    # normal equation: (XᵀX + λI) w = Xᵀ Y
    A = X.T @ X + lam * I            # (1568, 1568)
    b = X.T @ Y                      # (1568, 10)

    # solve in one step (no iterative optimiser)
    W_sol = np.linalg.solve(A, b)    # (1568, 10)
    new_weight = W_sol.T.astype(np.float32)   # (10, 1568)  → head.weight shape
    new_bias   = np.mean(Y, axis=0).astype(np.float32)   # (10,) → head.bias shape

    # ----- replace head parameters -----
    w["head.weight"] = new_weight
    w["head.bias"]   = new_bias

    # ----- diagnostics (computed on the full feedback pool) -----
    new_logits = X @ new_weight.T + new_bias          # (2000, 10)
    # numerically stable softmax
    exp_logits = np.exp(new_logits - np.max(new_logits, axis=1, keepdims=True))
    new_probs  = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    pred          = np.argmax(new_probs, axis=1)
    train_acc     = float(np.mean(pred == y))

    # cross‑entropy loss
    ce = -float(np.mean(np.log(new_probs[np.arange(2000), y] + 1e-12)))

    diag = {
        "train_accuracy": train_acc,
        "train_cross_entropy": ce,
        "head_weight_norm": float(np.linalg.norm(new_weight)),
        "head_bias": new_bias.tolist()
    }

    # return per spec: {"weights": w, "diagnostics": diag}
    return {"weights": w, "diagnostics": diag}
