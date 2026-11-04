def edit(w, f):
    """
    Closed‑form ridge‑regression update of the network head using the entire
    permitted feedback pool.  Only the head.weight and head.bias are modified;
    all other parameters stay unchanged.
    """
    # ----- feedback data -----
    X = f["features"]                     # shape (4000, 32), float32
    y = f["y"]                            # shape (4000,), int labels {0,1}
    # one‑hot targets
    Y = np.eye(2, dtype=np.float32)[y]    # shape (4000, 2)

    # ----- ridge regularisation -----
    lam = 1e-3                            # small ridge term
    Gram = X.T @ X + lam * np.eye(32, dtype=np.float32)   # (32,32)
    target = X.T @ Y                      # (32, 2)

    # ----- solve for head weight (32 -> 2) -----
    w_head_raw = np.linalg.solve(Gram, target)   # (32,2)
    head_weight = w_head_raw.T.astype(np.float32)  # (2,32)  matches w["head.weight"]

    # ----- bias from mean residual -----
    residuals = Y - X @ head_weight.T          # (4000,2)
    head_bias = np.mean(residuals, axis=0).astype(np.float32)  # (2,)

    # ----- write back into the weight dict -----
    w["head.weight"] = head_weight
    w["head.bias"]   = head_bias

    # ----- diagnostics (optional, JSON‑compatible) -----
    pred_logits = X @ head_weight.T + head_bias          # (4000,2)
    # numerically stable softmax
    max_logit = pred_logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(pred_logits - max_logit)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    acc = np.mean(np.argmax(probs, axis=1) == y)
    # cross‑entropy (using the correct‑class probability)
    ce = -np.mean(np.log(probs[np.arange(X.shape[0]), y]))
    diagnostics = {
        "accuracy": float(acc),
        "cross_entropy": float(ce)
    }

    return {"weights": w, "diagnostics": diagnostics}
