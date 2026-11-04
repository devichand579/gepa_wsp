def edit(w, f):
    """
    Closed‑form ridge‑regression update of fc2 weight matrix and bias
    using the entire 2000‑example feedback pool.
    """
    import numpy as np

    # ------------------------------------------------------------------
    # 1. Gather feedback arrays (read‑only)
    # ------------------------------------------------------------------
    X = f["features"].astype(np.float32)          # (2000, 128)
    y = f["y"].astype(np.int64)                   # (2000,)
    C = 10                                        # number of classes
    lam = 0.1                                     # ridge regularisation

    # ------------------------------------------------------------------
    # 2. One‑hot encode the labels
    # ------------------------------------------------------------------
    Y = np.eye(C, dtype=np.float32)[y]            # (2000, 10)

    # ------------------------------------------------------------------
    # 3. Ridge‑regression solution for fc2 weight (10,128)
    # ------------------------------------------------------------------
    D = X.shape[1]                                # 128
    XtX = X.T @ X                                 # (128, 128)
    I = np.eye(D, dtype=np.float32)
    A = XtX + lam * I                             # regularised Gram matrix
    XtY = X.T @ Y                                 # (128, 10)
    W_temp = np.linalg.solve(A, XtY)             # (128, 10)  temporary

    # fc2.weight is [out, in] → transpose back
    fc2_weight_new = W_temp.T.astype(np.float32)  # (10, 128)

    # ------------------------------------------------------------------
    # 4. Bias term: mean residual after applying the new weight
    # ------------------------------------------------------------------
    # logits_pred = X @ fc2_weight_new.T   (N,10)
    residual = Y - X @ fc2_weight_new.T          # (2000, 10)
    fc2_bias_new = np.mean(residual, axis=0).astype(np.float32)  # (10,)

    # ------------------------------------------------------------------
    # 5. Update the weight dictionary (fc1 left untouched)
    # ------------------------------------------------------------------
    w["fc2.weight"] = fc2_weight_new
    w["fc2.bias"] = fc2_bias_new

    # ------------------------------------------------------------------
    # 6. Diagnostics: recompute logits/accuracy/cross‑entropy on the
    #    full feedback pool using the *original* features (pre‑edit)
    # ------------------------------------------------------------------
    logits_new = X @ fc2_weight_new.T + fc2_bias_new   # (2000, 10)
    probs_new = np.exp(logits_new) / np.sum(np.exp(logits_new), axis=1, keepdims=True)
    acc = float(np.mean(np.argmax(probs_new, axis=1) == y))
    # numerically stable cross‑entropy
    ce = -float(np.mean(np.log(probs_new[np.arange(2000), y])))

    weight_norm = float(np.sqrt(np.sum(fc2_weight_new ** 2)))

    diagnostics = {
        "accuracy": acc,
        "cross_entropy": ce,
        "fc2_weight_norm": weight_norm,
        "fc2_bias": fc2_bias_new.tolist()
    }

    # ------------------------------------------------------------------
    # 7. Return updated weights and diagnostics
    # ------------------------------------------------------------------
    return {"weights": w, "diagnostics": diagnostics}
