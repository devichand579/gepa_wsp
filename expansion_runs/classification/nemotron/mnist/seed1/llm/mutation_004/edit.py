def edit(w, f):
    """
    Closed‑form ridge‑regression fit of fc2 weights using the entire
    permitted feedback pool.  No autograd, gradients, or iterative optimisers
    are used – only NumPy linear algebra.
    """
    # ----- feedback arrays -------------------------------------------------
    features = f["features"]                 # shape (2000, 128)
    y        = f["y"]                        # shape (2000,)   int64

    N, D_in  = features.shape                # 2000, 128
    C        = 10                            # number of classes

    # one‑hot encode the targets
    Y = np.eye(C)[y]                         # (2000, 10)

    # ----- ridge‑regression solution ---------------------------------------
    lam = 0.1                                # small ridge constant
    FT_F = features.T @ features             # (128, 128)
    FT_Y = features.T @ Y                    # (128, 10)

    # solve (FᵀF + λI) A = FᵀY  for A (128,10)
    A = np.linalg.solve(FT_F + lam * np.eye(D_in), FT_Y)

    # desired fc2 weight matrix shape (10, 128) → transpose A
    new_fc2_weight = A.T.astype(np.float32)   # (10, 128)

    # ----- edit the network weights ----------------------------------------
    w["fc2.weight"] = new_fc2_weight

    # keep the original fc2 bias (no change needed)
    # optional: recompute logits on the feedback pool for diagnostics
    new_logits = features @ new_fc2_weight.T + w["fc2.bias"]   # (2000,10)
    pred       = np.argmax(new_logits, axis=1)
    accuracy   = float(np.mean(pred == y))

    # cross‑entropy (stable softmax)
    exp_logits = np.exp(new_logits)
    probs      = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    ce         = float(-np.log(probs[np.arange(N), y]).mean())

    # ----- diagnostics -------------------------------------------------------
    diag = {
        "fc2_weight_norm": float(np.linalg.norm(new_fc2_weight)),
        "train_accuracy":  accuracy,
        "train_cross_entropy": ce,
        "lambda_used": lam
    }

    return {"weights": w, "diagnostics": diag}
