def edit(w, f):
    """
    Edit the network's head (linear classifier) using a closed‑form
    least‑squares fit on the full feedback pool.

    Parameters
    ----------
    w : dict
        Current full‑precision parameter dictionary (mutable).
    f : dict
        Read‑only feedback arrays as defined in the problem statement.

    Returns
    -------
    {"weights": w, "diagnostics": {...}}
        Updated weight dictionary and a small statistics dict.
    """
    import numpy as np

    N = f["x"].shape[0]                     # 2000 – size of the feedback pool
    # -------------------------------------------------
    # 1️⃣ Build one‑hot target matrix Y ∈ ℝ^{N×10}
    # -------------------------------------------------
    Y = np.zeros((N, 10), dtype=np.float32)
    Y[np.arange(N), f["y"].astype(np.int32)] = 1.0

    # -------------------------------------------------
    # 2️⃣ Closed‑form multivariate linear regression:
    #    B = pinv(features) @ Y   →  B shape (1568, 10)
    #    head.weight = B.T      →  (10, 1568)
    # -------------------------------------------------
    pinv_features = np.linalg.pinv(f["features"])   # (1568, 2000)
    B = pinv_features @ Y                           # (1568, 10)

    w["head.weight"] = B.T.astype(np.float32)       # (10, 1568)

    # -------------------------------------------------
    # 3️⃣ Compute optimal bias for each output unit:
    #    b_c = mean_n( y_{c,n} - (features @ B)_{c,n} )
    # -------------------------------------------------
    logits_pred = f["features"] @ B                 # (2000, 10)
    resid = Y - logits_pred                         # (2000, 10)
    b = resid.mean(axis=0).astype(np.float32)       # (10,)
    w["head.bias"] = b

    # -------------------------------------------------
    # 4️⃣ Diagnostics (tiny JSON‑compatible statistics)
    # -------------------------------------------------
    diag = {
        "head_weight_mean": float(np.mean(w["head.weight"])),
        "head_weight_std":  float(np.std(w["head.weight"])),
        "head_bias_mean":   float(np.mean(w["head.bias"])),
        "train_accuracy":   float((np.argmax(logits_pred, axis=1) == f["y"]).mean()),
        "full_feedback_accuracy": float(np.argmax(f["logits"], axis=1) == f["y"]).mean()
    }

    return {"weights": w, "diagnostics": diag}
