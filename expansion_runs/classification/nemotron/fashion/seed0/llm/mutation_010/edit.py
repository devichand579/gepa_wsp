def edit(w, f):
    """
    Edit the network head via a closed‑form ridge‑regression fit on the
    entire permitted feedback pool.  All other parameters (conv1, conv2)
    are left unchanged.

    Parameters
    ----------
    w : dict
        Current full parameter dictionary (contains conv1/2 and head weights/biases).
    f : dict
        Feedback arrays: f["features"] (N,1568), f["y"] (N,), f["logits"], f["probs"], etc.

    Returns
    -------
    dict or {"weights": w, "diagnostics": {...}}
        Updated weight dictionary and a small JSON‑compatible statistics dict.
    """
    import numpy as np

    # ------------------------------------------------------------------
    # 1. Gather feedback data
    # ------------------------------------------------------------------
    X = f["features"]                # shape (2000, 1568), float32
    y = f["y"].astype(np.int64)      # shape (2000,), int64 labels 0‑9

    # One‑hot encode labels → (2000, 10)
    Y = np.eye(10, dtype=np.float32)[y]   # (2000,10)

    # ------------------------------------------------------------------
    # 2. Ridge‑regression normal equation (closed‑form)
    # ------------------------------------------------------------------
    lam = 1e-3                     # small ridge regularisation
    d = X.shape[1]                 # 1568
    I = np.eye(d, dtype=np.float32)

    XtX = X.T @ X                  # (1568,1568)
    XtX_reg = XtX + lam * I        # regularised Gram matrix

    # Solve (XᵀX + λI) W = XᵀY  for W of shape (1568,10)
    # np.linalg.solve expects (n,n) @ (n,k) → (n,k)
    W_raw = np.linalg.solve(XtX_reg, X.T @ Y)   # (1568,10)

    # The network expects head.weight of shape (out,in) = (10,1568)
    W_new = W_raw.T.astype(np.float32)          # (10,1568)

    # ------------------------------------------------------------------
    # 3. Compute bias term from residuals
    # ------------------------------------------------------------------
    # Logits = X @ W_new.T  → (2000,10)
    logits_new = X @ W_new.T   # (2000,10)

    # Bias = mean of (Y - logits_new) over the sample axis
    b_new = (Y - logits_new).mean(axis=0).astype(np.float32)   # (10,)

    # ------------------------------------------------------------------
    # 4. Update the weight dictionary
    # ------------------------------------------------------------------
    w["head.weight"] = W_new
    w["head.bias"] = b_new

    # ------------------------------------------------------------------
    # 5. Diagnostics (accuracy & cross‑entropy on the feedback pool)
    # ------------------------------------------------------------------
    # Softmax probabilities
    exp_logits = np.exp(logits_new)
    probs_new = exp_logits / exp_logits.sum(axis=1, keepdims=True)   # (2000,10)

    pred_labels = np.argmax(probs_new, axis=1)          # (2000,)
    acc = float(np.mean(pred_labels == y))

    # Cross‑entropy loss (average over samples, correct‑class only)
    # Numerically stable: clip probs to avoid log(0)
    eps = 1e-12
    probs_clipped = np.clip(probs_new, eps, 1.0 - eps)
    ce = -float(np.mean(np.log(probs_clipped[np.arange(2000), y])))

    diag = {
        "head_weight_mean": float(np.mean(w["head.weight"])),
        "head_weight_std": float(np.std(w["head.weight"])),
        "head_bias_mean": float(np.mean(w["head.bias"])),
        "feedback_accuracy": acc,
        "feedback_cross_entropy": ce,
        # optional: keep original conv stats for reference
        "conv1_weight_mean_before": float(np.mean(w["conv1.weight"])),
        "conv2_weight_mean_before": float(np.mean(w["conv2.weight"])),
    }

    # Return the updated weights together with diagnostics
    return {"weights": w, "diagnostics": diag}
