def edit(w, f):
    """
    Closed‑form ridge regression of fc2 on the full feedback pool.
    fc1 weights are left unchanged; only fc2.weight and fc2.bias are edited.
    Returns the modified weight dict and a small diagnostics dict.
    """
    import numpy as np

    # ------------------------------------------------------------------
    # 1. Pull feedback arrays
    # ------------------------------------------------------------------
    features = f["features"]                # shape (2000, 128)
    y        = f["y"]                       # shape (2000,), int64, values 0‑9

    # ------------------------------------------------------------------
    # 2. Build one‑hot label matrix Y (2000, 10)
    # ------------------------------------------------------------------
    Y = np.eye(10, dtype=np.float32)[y]    # (2000, 10)

    # ------------------------------------------------------------------
    # 3. Ridge‑regression solution for fc2.weight (10, 128)
    # ------------------------------------------------------------------
    lam = 1e-3                              # small ridge constant for stability
    A = features.T @ features + lam * np.eye(128, dtype=np.float32)   # (128,128)
    B = features.T @ Y                     # (128, 10)
    # solve A * X = B  =>  X is (128,10); we need weight = X.T -> (10,128)
    X = np.linalg.solve(A, B)              # (128, 10)
    fc2_weight_new = X.T                   # (10, 128)

    # ------------------------------------------------------------------
    # 4. Compute corresponding bias (10,)
    # ------------------------------------------------------------------
    # logits = features @ weight.T + bias  =>  bias = mean(Y - features @ weight.T)
    logits_pred = features @ fc2_weight_new.T   # (2000, 10)
    fc2_bias_new = np.mean(Y - logits_pred, axis=0)  # (10,)

    # ------------------------------------------------------------------
    # 5. Apply the new weights / bias (in‑place on the supplied dict)
    # ------------------------------------------------------------------
    w["fc2.weight"] = fc2_weight_new.astype(np.float32)
    w["fc2.bias"]   = fc2_bias_new.astype(np.float32)

    # ------------------------------------------------------------------
    # 6. Evaluate on the feedback pool (no autograd, just NumPy)
    # ------------------------------------------------------------------
    # softmax for probabilities
    logits_all = features @ w["fc2.weight"].T + w["fc2.bias"]   # (2000,10)
    # numerically stable softmax
    max_logit = logits_all.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits_all - max_logit)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)   # (2000,10)

    # accuracy
    pred_labels = np.argmax(probs, axis=1)          # (2000,)
    accuracy = np.mean(pred_labels == y)

    # cross‑entropy
    eps = 1e-12
    ce = -np.mean(np.sum(Y * np.log(probs + eps), axis=1))

    # ------------------------------------------------------------------
    # 7. Diagnostics (JSON‑compatible, small)
    # ------------------------------------------------------------------
    diag = {
        "fc2_weight_mean": float(np.mean(w["fc2.weight"])),
        "fc2_weight_std":  float(np.std(w["fc2.weight"])),
        "fc2_bias_mean":   float(np.mean(w["fc2.bias"])),
        "fc2_bias_std":    float(np.std(w["fc2.bias"])),
        "feedback_accuracy": float(accuracy),
        "feedback_cross_entropy": float(ce),
    }

    return {"weights": w, "diagnostics": diag}
