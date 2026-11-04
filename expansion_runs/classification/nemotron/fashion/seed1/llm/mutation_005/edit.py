def edit(w, f):
    """
    Edit the network weights using a single closed‑form ridge‑regression step
    on the provided feedback arrays.  Only the head (linear classifier) is
    modified; all other parameters are left unchanged.
    """
    # ----- feedback data ----------------------------------------------------
    X = f["features"]                 # (2000, 1568) float32
    y = f["y"]                        # (2000,) int64
    N, D = X.shape                     # D = 1568
    C = 10                            # number of classes

    # one‑hot targets (N, C)
    Y = np.zeros((N, C), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    # ----- ridge‑regularised linear solver -------------------------------
    lam = 1e-3                         # small ridge coefficient
    # Dual form: W = Y @ X.T @ inv(X @ X.T + λ I)
    XtX = X @ X.T + lam * np.eye(D, dtype=np.float32)   # (D, D)
    XtX_inv = np.linalg.inv(XtX)                       # (D, D)
    W = Y @ X.T @ XtX_inv                              # (C, D) -> head.weight shape

    # bias = mean residual after the linear map
    logits_raw = X @ W.T                               # (N, C)
    bias = np.mean(Y - logits_raw, axis=0).astype(np.float32)   # (C,)

    # clip to allowed range
    W = np.clip(W, -1000, 1000).astype(np.float32)
    bias = np.clip(bias, -1000, 1000).astype(np.float32)

    # ----- apply the new head weights ------------------------------------
    w["head.weight"] = W
    w["head.bias"] = bias

    # ----- diagnostics ----------------------------------------------------
    # softmax for probabilities
    logits_new = X @ W.T + bias                        # (N, C)
    # numerically stable softmax
    max_logit = logits_new.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits_new - max_logit)
    probs_new = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    acc = float(np.mean(np.argmax(probs_new, axis=1) == y))
    # cross‑entropy loss
    ce = -float(np.mean(np.sum(Y * np.log(probs_new + 1e-12), axis=1)))

    diag = {
        "train_accuracy": acc,
        "train_cross_entropy": ce,
        "head_weight_norm": float(np.linalg.norm(W.astype(np.float64))),
        "head_bias_norm": float(np.linalg.norm(bias.astype(np.float64))),
        "lambda_used": float(lam)
    }

    return {"weights": w, "diagnostics": diag}
