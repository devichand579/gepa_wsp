def edit(w, f):
    """
    One‑shot, closed‑form update of the network’s head layer using the
    permitted feedback arrays.  No back‑prop, no iterative optimiser.
    """
    import numpy as np  # np is assumed already imported by the caller

    # ------------------------------------------------------------------
    # 1. Gather the full feedback pool
    # ------------------------------------------------------------------
    features = f["features"]                # shape (2000, 1568), float32
    y        = f["y"]                       # shape (2000,), int64

    # ------------------------------------------------------------------
    # 2. One‑hot encode the labels (C = 10 classes)
    # ------------------------------------------------------------------
    C = 10
    Y_onehot = np.eye(C, dtype=np.float32)[y]   # (2000, 10)

    # ------------------------------------------------------------------
    # 3. Ridge‑regression solution for head.weight and head.bias
    # ------------------------------------------------------------------
    D = features.shape[1]                     # 1568
    lam = 1e-3                                 # small ridge regulariser

    FT = features.T                            # (1568, 2000)
    HTH = FT @ features                        # (1568, 1568)
    HTY = FT @ Y_onehot                        # (1568, 10)

    # Add ridge to the Gram matrix and solve the linear system
    I = np.eye(D, dtype=np.float32) * lam
    HTH_reg = HTH + I
    # np.linalg.solve is a closed‑form step, not an iterative optimizer
    W_raw = np.linalg.solve(HTH_reg, HTY)    # (1568, 10)

    # The PyTorch convention for linear weights is [out, in] → transpose
    head_weight = W_raw.T.astype(np.float32)   # (10, 1568)

    # Bias: mean residual after the linear map
    pred_logits = features @ W_raw             # (2000, 10)
    head_bias   = (Y_onehot - pred_logits).mean(axis=0).astype(np.float32)  # (10,)

    # ------------------------------------------------------------------
    # 4. Apply the updates (preserve all other weights unchanged)
    # ------------------------------------------------------------------
    w["head.weight"] = head_weight
    w["head.bias"]   = head_bias

    # ------------------------------------------------------------------
    # 5. Evaluate training accuracy on the feedback pool
    # ------------------------------------------------------------------
    # Logits are simply the new head output (features unchanged)
    logits = features @ head_weight + head_bias   # (2000, 10)
    preds = np.argmax(logits, axis=1)             # (2000,)
    acc = float(np.mean(preds == y) * 100)        # percent

    # ------------------------------------------------------------------
    # 6. Diagnostics (JSON‑compatible, small)
    # ------------------------------------------------------------------
    diag = {
        "head_weight_mean":  float(np.mean(head_weight)),
        "head_weight_std":   float(np.std(head_weight)),
        "head_bias_mean":    float(np.mean(head_bias)),
        "training_accuracy_%": acc,
        "ridge_lambda":      float(lam)
    }

    # Return the full weight dict together with diagnostics
    return {"weights": w, "diagnostics": diag}
