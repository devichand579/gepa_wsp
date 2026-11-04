def edit(w, f):
    """
    Closed‑form ridge‑regression update of the network's head.
    Only the head weights and bias are modified; conv1/conv2 stay unchanged.
    """
    # ------------------------------------------------------------------
    # 1. Pull the full training pool from feedback arrays
    # ------------------------------------------------------------------
    X = f["features"].astype(np.float32)       # shape (2000, 1568)
    y = f["y"].astype(np.int64)                # shape (2000,)

    # ------------------------------------------------------------------
    # 2. One‑hot encode the labels (10 classes)
    # ------------------------------------------------------------------
    N, D = X.shape
    K = 10
    T = np.eye(K, dtype=np.float32)[y]        # (2000, 10)

    # ------------------------------------------------------------------
    # 3. Center features and targets (helps the closed‑form solution)
    # ------------------------------------------------------------------
    f_mean = X.mean(axis=0, keepdims=True)    # (1, 1568)
    t_mean = T.mean(axis=0, keepdims=True)    # (1, 10)
    Xc = X - f_mean
    Tc = T - t_mean

    # ------------------------------------------------------------------
    # 4. Ridge‑regression normal equation (lambda = 1e‑3)
    # ------------------------------------------------------------------
    lam = 1e-3
    Gram = Xc.T @ Xc + lam * np.eye(D, dtype=np.float32)   # (1568, 1568)
    # Solve Xc @ Wc ≈ Tc  →  Wc = (XcᵀXc + λI)⁻¹ Xcᵀ Tc   shape (1568, 10)
    Wc = np.linalg.solve(Gram, Xc.T @ Tc)                # (1568, 10)

    # ------------------------------------------------------------------
    # 5. Map back to the network's weight convention
    #    head.weight [10,1568]  →  Wc.T
    #    head.bias  = t_mean - f_mean @ Wc
    # ------------------------------------------------------------------
    head_weight_new = Wc.T.astype(np.float32)            # (10, 1568)
    head_bias_new   = (t_mean - f_mean @ Wc).ravel().astype(np.float32)  # (10,)

    # ------------------------------------------------------------------
    # 6. Apply the updates in‑place (other layers stay untouched)
    # ------------------------------------------------------------------
    w["head.weight"] = head_weight_new
    w["head.bias"]   = head_bias_new

    # ------------------------------------------------------------------
    # 7. Diagnostics: new logits on the training set + accuracy
    # ------------------------------------------------------------------
    logits_new = X @ head_weight_new.T + head_bias_new[np.newaxis, :]   # (2000, 10)
    pred       = np.argmax(logits_new, axis=1)
    train_acc  = float(np.mean(pred == y))

    diagnostics = {
        "head_weight_mean":  float(np.mean(head_weight_new)),
        "head_weight_std":   float(np.std(head_weight_new)),
        "head_bias_mean":    float(np.mean(head_bias_new)),
        "train_accuracy":    train_acc,
        "ridge_lambda":      float(lam)
    }

    # Return the full weight dict together with a tiny JSON‑compatible stats block
    return {"weights": w, "diagnostics": diagnostics}
