def edit(w, f):
    """
    Edit the neural‑network weights using a closed‑form ridge‑regression fit
    on the provided feedback arrays. Only fc2.weight and fc2.bias are updated;
    all other parameters are left unchanged.
    """
    # --- extract feedback arrays ------------------------------------------------
    features = f["features"]                # shape (2000, 128)
    y        = f["y"]                       # shape (2000,), int64

    # --- one‑hot encode labels -------------------------------------------------
    N = features.shape[0]
    Y = np.eye(10, dtype=np.float32)[y]    # (N, 10)

    # --- ridge‑regression solution for fc2 ------------------------------------
    lam = 1e-3                              # small ridge parameter
    Gram = features.T @ features            # (128, 128)
    Gram_reg = Gram + lam * np.eye(128, dtype=np.float32)

    # solve (Gram_reg) @ W_raw = features.T @ Y  for W_raw (128, 10)
    W_raw = np.linalg.solve(Gram_reg, features.T @ Y)   # (128, 10)

    # fc2.weight expects shape (10, 128); transpose accordingly
    fc2_weight = W_raw.T.astype(np.float32)            # (10, 128)

    # fc2.bias: mean residual across samples, shape (10,)
    residuals = Y - (features @ W_raw)                 # (N, 10)
    fc2_bias = residuals.mean(axis=0).astype(np.float32)  # (10,)

    # --- update weights ---------------------------------------------------------
    w["fc2.weight"] = fc2_weight
    w["fc2.bias"]    = fc2_bias

    # --- diagnostics (train‑set fit quality) -----------------------------------
    logits = features @ W_raw + fc2_bias               # (N, 10)
    # softmax for probabilities
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)
    train_acc = float(np.mean(preds == y))
    train_ce  = float(-np.mean(np.log(probs[np.arange(N), y])))

    diagnostics = {
        "ridge_lambda": float(lam),
        "train_accuracy": train_acc,
        "train_cross_entropy": train_ce,
        "fc2_weight_mean": float(fc2_weight.mean()),
        "fc2_bias_mean":   float(fc2_bias.mean())
    }

    return {"weights": w, "diagnostics": diagnostics}
