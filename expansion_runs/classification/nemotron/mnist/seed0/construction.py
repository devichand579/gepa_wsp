def edit(w, f):
    """
    One‑shot feedback‑assisted construction of fc2 weights via ordinary least squares.
    No autograd, back‑prop, or iterative optimiser is used.
    """
    import numpy as np

    # ---- current second‑layer parameters ---------------------------------
    fc2_w = w["fc2.weight"]          # shape (10, 128), float32
    fc2_b = w["fc2.bias"]            # shape (10,),   float32

    # ---- reflection batch ------------------------------------------------
    batch_x = f["batch.x"]           # (128, 1, 28, 28)
    batch_y = f["batch.y"]           # (128,)
    # hidden activations after fc1 ReLU for the batch
    h_batch = f["layer.fc1.output"]  # (128, 128)

    # ---- one‑hot targets -------------------------------------------------
    Y_onehot = np.zeros((128, 10), dtype=np.float32)
    Y_onehot[np.arange(128), batch_y] = 1.0

    # ---- design matrix (features + bias term) ---------------------------
    X = np.hstack([h_batch, np.ones((128, 1), dtype=np.float32)])  # (128, 129)

    # ---- closed‑form least‑squares fit ----------------------------------
    # np.linalg.lstsq returns B of shape (129, 10) solving X @ B ≈ Y_onehot
    B, _, _, _ = np.linalg.lstsq(X, Y_onehot, rcond=None)

    # extract weight matrix (10, 128) and bias vector (10,)
    fc2_new_weight = B[:128, :].astype(np.float32).T   # (10, 128)
    fc2_new_bias   = B[128, :].astype(np.float32)      # (10,)

    # ---- update the weight dictionary -----------------------------------
    w["fc2.weight"] = fc2_new_weight
    w["fc2.bias"]   = fc2_new_bias

    # ---- diagnostics on the full feedback pool -------------------------
    features_full = f["features"]                # (2000, 128)
    logits_full = features_full @ fc2_new_weight.T + fc2_new_bias[np.newaxis, :]  # (2000, 10)

    # numerically stable softmax
    logits_max = logits_full.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits_full - logits_max)
    probs_full = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    pred_full = np.argmax(probs_full, axis=1)
    accuracy_full = float((pred_full == f["y"].astype(np.int32)).mean())

    eps = 1e-12
    probs_clipped = np.clip(probs_full, eps, 1.0 - eps)
    ce_full = float(-np.mean(np.log(probs_clipped[np.arange(2000), f["y"]])))

    weight_mean_change = float(np.abs(fc2_new_weight - fc2_w).mean())
    bias_mean_change   = float(np.abs(fc2_new_bias - fc2_b).mean())

    diagnostics = {
        "full_pool_accuracy": accuracy_full,
        "full_pool_cross_entropy": ce_full,
        "fc2_weight_mean_change": weight_mean_change,
        "fc2_bias_mean_change": bias_mean_change,
        "fc2_weight_norm": float(np.linalg.norm(fc2_new_weight)),
        "fc2_bias_norm": float(np.linalg.norm(fc2_new_bias)),
        "batch_used": int(batch_y.shape[0])
    }

    return {"weights": w, "diagnostics": diagnostics}
