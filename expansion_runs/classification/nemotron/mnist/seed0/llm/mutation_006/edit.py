def edit(w, f):
    # --- extract feedback pool -------------------------------------------------
    H = f["features"]          # shape [2000, 128], post‑fc1 ReLU activations
    y = f["y"]                 # shape [2000], integer labels 0‑9
    N = H.shape[0]

    # --- one‑hot encode labels -------------------------------------------------
    Y = np.eye(10)[y]          # shape [2000, 10]

    # --- augment features with a constant term for bias learning -------------
    H_aug = np.hstack([H, np.ones((N, 1))])   # shape [2000, 129]

    # --- ridge‑regression hyper‑parameter --------------------------------------
    lam = 0.1   # small L2 regularisation; can be tuned but kept fixed here

    # --- normal equations -------------------------------------------------------
    A = H_aug.T @ H_aug + lam * np.eye(129)    # [129, 129]
    B = H_aug.T @ Y                            # [129, 10]
    W_aug = np.linalg.solve(A, B)              # [129, 10]

    # --- split into weight matrix and bias --------------------------------------
    # W_aug[:128, :] corresponds to fc2.weight transposed (128, 10)
    weight_part = W_aug[:128, :]                # shape [128, 10]
    new_fc2_weight = weight_part.T              # shape [10, 128]  (PyTorch convention)
    new_fc2_bias   = W_aug[128, :]              # shape [10]

    # --- apply the new parameters ------------------------------------------------
    w["fc2.weight"][:] = new_fc2_weight
    w["fc2.bias"][:]   = new_fc2_bias

    # --- diagnostics ------------------------------------------------------------
    logits = H @ new_fc2_weight.T + new_fc2_bias   # [2000, 10]
    pred   = np.argmax(logits, axis=1)
    acc    = float(np.mean(pred == y))

    diag = {
        "fc2_weight_norm": float(np.linalg.norm(new_fc2_weight)),
        "fc2_bias_norm":   float(np.linalg.norm(new_fc2_bias)),
        "feedback_pool_accuracy": acc,
        "ridge_lambda": lam
    }

    return {"weights": w, "diagnostics": diag}
