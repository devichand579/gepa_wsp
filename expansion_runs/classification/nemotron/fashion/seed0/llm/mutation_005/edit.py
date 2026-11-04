def edit(w, f):
    """
    Edit the network's head layer with a closed‑form ridge‑regression fit
    on the full permitted feedback pool.  All other weights remain unchanged.
    """
    # ----- data from feedback arrays -----
    X = f["features"]                # shape (2000, 1568), float32
    y = f["y"]                       # shape (2000,), int64

    # ----- ridge regularisation -----
    lam = 0.001                      # small L2 penalty
    I = np.eye(1568, dtype=np.float32) * lam

    # ----- one‑hot encode labels -----
    Y = np.zeros((2000, 10), dtype=np.float32)
    Y[np.arange(2000), y] = 1.0

    # ----- ridge solution for head.weight (10,1568) -----
    # Solve (XᵀX + λI) w = XᵀY  for wᵀ (1568,10), then transpose.
    XtX = X.T @ X + I                # (1568,1568)
    XtY = X.T @ Y                    # (1568,10)
    head_weight_t = np.linalg.solve(XtX, XtY)   # (1568,10)
    new_head_weight = head_weight_t.T           # (10,1568), float32

    # ----- bias: mean residual per class -----
    # preds = X @ new_head_weight.T  gives logits without bias
    residual = Y - X @ new_head_weight.T       # (2000,10)
    new_head_bias = np.mean(residual, axis=0)  # (10,), float32

    # ----- update weights in the dictionary -----
    w["head.weight"] = new_head_weight.astype(np.float32)
    w["head.bias"]   = new_head_bias.astype(np.float32)

    # ----- diagnostics -----
    logits = X @ new_head_weight.T + new_head_bias          # (2000,10)
    pred_labels = np.argmax(logits, axis=1)
    accuracy = float(np.mean(pred_labels == y))

    # cross‑entropy (numerically stable)
    probs = np.exp(logits - np.max(logits, axis, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    ce = -np.mean(np.log(probs[np.arange(2000), y] + 1e-12))

    diag = {
        "lambda": float(lam),
        "head_weight_mean": float(new_head_weight.mean()),
        "head_weight_std":  float(new_head_weight.std()),
        "head_bias_mean":   float(new_head_bias.mean()),
        "train_accuracy":   accuracy,
        "cross_entropy":    float(ce),
    }

    return {"weights": w, "diagnostics": diag}
