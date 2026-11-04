def edit(w, f):
    """
    Edit the neural network weights using a closed‑form ridge regression
    on the fc2 output layer. fc1 weights and biases are left unchanged.
    """
    import numpy as np

    # ------------------------------------------------------------------
    # 1. Pull current fc2 parameters and the full feedback pool.
    # ------------------------------------------------------------------
    fc2_weight = w["fc2.weight"].astype(np.float32)       # shape (10, 128)
    fc2_bias   = w["fc2.bias"].astype(np.float32)        # shape (10,)

    features = f["features"]      # shape (2000, 128)  – activations before edit
    y        = f["y"].astype(np.int64)               # shape (2000,)

    # ------------------------------------------------------------------
    # 2. Build one‑hot label matrix Y (2000, 10).
    # ------------------------------------------------------------------
    Y = np.eye(10, dtype=np.float32)[y]   # each row is a one‑hot vector

    # ------------------------------------------------------------------
    # 3. Ridge‑regression solution:  W = (XᵀX + λI)⁻¹ Xᵀ Y
    #    λ = 0.1  (chosen from the recent proposal history)
    # ------------------------------------------------------------------
    lam = np.float32(0.1)
    XtX = features.T @ features + lam * np.eye(128, dtype=np.float32)   # (128,128)
    XtY = features.T @ Y                     # (128,10)

    # Solve the linear system for all 10 output units at once.
    # np.linalg.solve returns a (128,10) matrix where column c is the
    # weight vector for class c.
    W_raw = np.linalg.solve(XtX, XtY)        # (128,10)

    # Transpose to the expected (10,128) shape.
    new_fc2_weight = W_raw.T.astype(np.float32)

    # ------------------------------------------------------------------
    # 4. Set the new fc2 bias to the empirical class frequencies
    #    (mean of the one‑hot targets).  This is a simple closed‑form
    #    choice; previous proposals kept the old bias, but updating it
    #    helps align the decision boundaries.
    # ------------------------------------------------------------------
    new_fc2_bias = Y.mean(axis=0).astype(np.float32)   # shape (10,)

    # ------------------------------------------------------------------
    # 5. Write the updated parameters back into the weight dict.
    # ------------------------------------------------------------------
    w["fc2.weight"] = new_fc2_weight
    w["fc2.bias"]   = new_fc2_bias

    # ------------------------------------------------------------------
    # 6. Diagnostics – useful statistics for later reflection.
    # ------------------------------------------------------------------
    # Predict logits and probabilities on the feedback pool.
    logits = features @ new_fc2_weight.T + new_fc2_bias   # (2000,10)
    # Numerically stable softmax.
    maxlog = logits.max(axis=1, keepdims=True)
    expE   = np.exp(logits - maxlog)
    probs  = expE / expE.sum(axis=1, keepdims=True)       # (2000,10)

    accuracy = np.mean(np.argmax(probs, axis=1) == y)
    # Cross‑entropy (using the predicted probabilities).
    # Clip probs to avoid log(0).
    eps = np.finfo(np.float32).eps
    neg_ll = -np.log(np.clip(probs[np.arange(2000), y], eps, 1.0))
    cross_entropy = neg_ll.mean()

    diagnostics = {
        "fc2_weight_mean": float(new_fc2_weight.mean()),
        "fc2_weight_std":  float(new_fc2_weight.std()),
        "fc2_bias_mean":   float(new_fc2_bias.mean()),
        "fc2_bias_std":    float(new_fc2_bias.std()),
        "train_accuracy":  float(accuracy),
        "train_cross_entropy": float(cross_entropy),
    }

    # Return the modified weight dict together with a small diagnostics
    # JSON‑compatible block.  The caller may keep only the weights or
    # both, according to the interface spec.
    return {"weights": w, "diagnostics": diagnostics}
