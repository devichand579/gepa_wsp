def edit(w, f):
    """
    w  – full current float32 parameter dictionary (mutable copy will be returned)
    f  – read‑only feedback dictionary as described in the prompt
    Returns w (updated) and a small diagnostics dict.
    No autograd, no iterative optimiser, only NumPy linear algebra.
    """
    import numpy as np

    # keep a copy of the original head parameters for diagnostics
    orig_head_weight = w["head.weight"].copy()
    orig_head_bias     = w["head.bias"].copy()

    # -------------------------------------------------
    # 1.  Gather the permitted feedback arrays
    # -------------------------------------------------
    X = f["features"]                # shape (2000, 1568)   – final hidden features
    y = f["y"]                       # shape (2000,)        – integer labels

    # one‑hot encode the labels → (2000, 10)
    T = np.eye(10, dtype=np.float32)[y]

    # -------------------------------------------------
    # 2.  Closed‑form least‑squares for the head layer
    # -------------------------------------------------
    # Add a bias column to the feature matrix so the bias is learned jointly.
    Xb = np.hstack([X, np.ones((2000, 1), dtype=np.float32)])   # (2000, 1569)

    # Solve Xb @ W ≈ T  →  W has shape (1569, 10)
    # np.linalg.lstsq returns the minimum‑norm solution.
    W, _, _, _ = np.linalg.lstsq(Xb.astype(np.float32),
                                 T.astype(np.float32),
                                 rcond=None)       # W: (1569, 10)

    # Split into weight matrix (10, 1568) and bias vector (10,)
    head_weight_new = W[:1568].reshape(10, 1568).astype(np.float32)   # (10, 1568)
    head_bias_new   = W[1568].astype(np.float32)                     # (10,)

    # -------------------------------------------------
    # 3.  Write the new head parameters back into w
    # -------------------------------------------------
    w["head.weight"] = head_weight_new
    w["head.bias"]   = head_bias_new

    # -------------------------------------------------
    # 4.  Diagnostics (small JSON‑compatible dict)
    # -------------------------------------------------
    # Predicted logits on the whole training pool and derived accuracy
    logits_pred = X @ head_weight_new.T + head_bias_new          # (2000, 10)
    pred_labels = np.argmax(logits_pred, axis=1)
    train_accuracy = float(np.mean(pred_labels == y))

    # Parameter‑change magnitudes (for later reflection)
    w_diff = head_weight_new - orig_head_weight          # (10, 1568)
    b_diff = head_bias_new - orig_head_bias              # (10,)

    diagnostics = {
        "head_weight_mean_abs_change": float(np.mean(np.abs(w_diff))),
        "head_weight_std_abs_change":  float(np.std(np.abs(w_diff))),
        "head_bias_mean_abs_change":   float(np.mean(np.abs(b_diff))),
        "head_bias_std_abs_change":    float(np.std(np.abs(b_diff))),
        "train_accuracy":              train_accuracy,
        # optional: keep the raw new weight/bias means for inspection
        "head_weight_mean":          float(np.mean(head_weight_new)),
        "head_weight_std":           float(np.std(head_weight_new)),
        "head_bias_mean":            float(np.mean(head_bias_new)),
        "head_bias_std":             float(np.std(head_bias_new)),
    }

    return w, diagnostics
