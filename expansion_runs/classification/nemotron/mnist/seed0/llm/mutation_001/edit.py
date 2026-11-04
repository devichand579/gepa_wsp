def edit(w, f):
    """
    Closed‑form ridge‑regression update of fc2 using the paired reflection batch.
    Parameters
    ----------
    w : dict
        Current network parameters:
        - w["fc1.weight"] shape (128, 784)
        - w["fc1.bias"]   shape (128,)
        - w["fc2.weight"] shape (10, 128)
        - w["fc2.bias"]   shape (10,)
    f : dict
        Feedback arrays (read‑only).  Keys used:
        - f["batch.ids"]   shape (128,) int64   – indices into the full pool
        - f["x"]          shape (2000,1,28,28) float32
        - f["y"]          shape (2000,)        int64
        - f["features"]   shape (2000,128)    float32  (post‑relu activations)
    Returns
    -------
    {"weights": w, "diagnostics": {...}}
    """
    # ------------------------------------------------------------------
    # 1. Select the 128‑sample reflection batch
    ids = f["batch.ids"]                     # (128,)
    # (We could also use f["batch.x"]/f["batch.y"] directly; indexing
    #  the full arrays with ids is equivalent and keeps everything consistent.)
    x_batch   = f["x"][ids]                  # (128,1,28,28) – not needed for fc2 update
    y_batch   = f["y"][ids]                  # (128,) int64
    H_batch   = f["features"][ids]           # (128,128) float32  (post‑fc1 activations)

    # 2. Build one‑hot targets
    Y_onehot = np.eye(10, dtype=np.float32)[y_batch]   # (128,10)

    # 3. Center features and labels (helps the normal equation)
    Hc = H_batch - np.mean(H_batch, axis=0, keepdims=True)   # (128,128)
    Yc = Y_onehot - np.mean(Y_onehot, axis=0, keepdims=True) # (128,10)

    # 4. Ridge‑regression (lambda = 1e‑3, you can tune)
    lam = 1e-3
    # Solve Hc @ W^T ≈ Yc  for W of shape (10,128)
    # np.linalg.lstsq returns coeff of shape (128,10) where row i corresponds to sample i.
    coeff, _, _, _ = np.linalg.lstsq(Hc, Yc, rcond=None)   # (128,10)
    fc2_weight_new = coeff.T.astype(np.float32)           # (10,128)

    # 5. Bias: mean of the centered residuals
    pred = Hc @ coeff                     # (128,10)
    bias_new = np.mean(Yc - pred, axis=0).astype(np.float32)  # (10,)

    # 6. Apply the new weights (other layers stay untouched)
    w["fc2.weight"] = fc2_weight_new
    w["fc2.bias"]   = bias_new

    # 7. Diagnostics (tiny JSON‑compatible dict)
    diagnostics = {
        "fc2_weight_norm_diff": float(np.linalg.norm(fc2_weight_new -
                      np.float32(w["fc2.weight"] if False else w["fc2.weight"]))),
        "batch_accuracy": float(np.mean(np.argmax(Y_onehot, axis=1) ==
                     np.argmax(pred, axis=1)))
    }

    return {"weights": w, "diagnostics": diagnostics}
