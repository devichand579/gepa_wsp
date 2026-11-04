def edit(w, f):
    """
    Closed‑form ridge‑regression fit of fc2 on the full 2000‑example feedback pool.
    fc1 weights are left untouched; only fc2.weight and fc2.bias are updated.
    Returns the modified weight dict and a small diagnostics dict.
    """
    # ------------------------------------------------------------------
    # 1. Pull current fc2 parameters (float32)
    # ------------------------------------------------------------------
    W2 = w["fc2.weight"].astype(np.float32)          # shape (10, 128)
    b2 = w["fc2.bias"].astype(np.float32)          # shape (10,)

    # ------------------------------------------------------------------
    # 2. Pull the full feedback pool: features and labels
    # ------------------------------------------------------------------
    X = f["features"].astype(np.float32)           # shape (2000, 128)
    y = f["y"].astype(np.int64)                    # shape (2000,)

    # ------------------------------------------------------------------
    # 3. One‑hot encode the labels -> Y (2000, 10)
    # ------------------------------------------------------------------
    Y = np.eye(10, dtype=np.float32)[y]            # (2000, 10)

    # ------------------------------------------------------------------
    # 4. Augment features with a constant 1 so bias is learned jointly
    # ------------------------------------------------------------------
    ones = np.ones((X.shape[0], 1), dtype=np.float32)
    Xaug = np.hstack([X, ones])                    # (2000, 129)

    # ------------------------------------------------------------------
    # 5. Ridge regularisation (small lambda; no iterative search)
    # ------------------------------------------------------------------
    lam = 1e-3
    Gram = Xaug.T @ Xaug + lam * np.eye(129, dtype=np.float32)

    # ------------------------------------------------------------------
    # 6. Closed‑form solution for augmented weights [W2 | b2]
    # ------------------------------------------------------------------
    #   W2_aug = (Y.T @ Xaug) @ inv(Gram)   -> shape (10, 129)
    # ------------------------------------------------------------------
    W2_aug = (Y.T @ Xaug) @ np.linalg.inv(Gram)      # (10, 129)

    # ------------------------------------------------------------------
    # 7. Split back into weight matrix and bias vector
    # ------------------------------------------------------------------
    new_W2 = W2_aug[:, :128].astype(np.float32)     # (10, 128)
    new_b2 = W2_aug[:, 128].astype(np.float32)     # (10,)

    # ------------------------------------------------------------------
    # 8. Update the weight dictionary (fc1 stays unchanged)
    # ------------------------------------------------------------------
    w["fc2.weight"] = new_W2
    w["fc2.bias"]   = new_b2

    # ------------------------------------------------------------------
    # 9. Compute diagnostics on the full pool (no training loop)
    # ------------------------------------------------------------------
    logits = X @ new_W2.T + new_b2                 # (2000, 10)
    pred   = np.argmax(logits, axis=1)
    acc    = float(np.mean(pred == y))
    wnorm  = float(np.linalg.norm(new_W2))

    diagnostics = {
        "train_accuracy": acc,
        "fc2_weight_norm": wnorm,
        "fc2_bias": new_b2.tolist()
    }

    # ------------------------------------------------------------------
    # 10. Return the modified weights together with diagnostics
    # ------------------------------------------------------------------
    return {"weights": w, "diagnostics": diagnostics}
