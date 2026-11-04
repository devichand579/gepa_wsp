def edit(w, f):
    """
    Closed‑form ridge‑regression update of the classification head using the
    full feedback pool.  No autograd, no loops, no iterative optimiser.
    """
    import numpy as np

    # ------------------------------------------------------------------
    # 1. Pull the permitted feedback arrays (all 4000 examples)
    # ------------------------------------------------------------------
    X = f["features"].astype(np.float32)          # (4000, 32) – final hidden
    y = f["y"].astype(np.int64)                  # (4000,) – 0=neg, 1=pos

    # one‑hot targets (2 classes)
    Y = np.eye(2, dtype=np.float32)[y]           # (4000, 2)

    # ridge regularisation (small constant; prevents singular normal eq.)
    lam = 1e-3

    # ------------------------------------------------------------------
    # 2. Compute new head.weight via (XᵀX + λI)⁻¹ XᵀY
    # ------------------------------------------------------------------
    XtX = X.T @ X + lam * np.eye(32, dtype=np.float32)   # (32,32)
    XtY = X.T @ Y                                      # (32,2)

    # solve the normal equations (guaranteed invertible because of λ)
    new_weight_raw = np.linalg.solve(XtX, XtY)        # (32,2)
    new_head_weight = new_weight_raw.T.astype(np.float32)   # (2,32)

    # ------------------------------------------------------------------
    # 3. Compute new head.bias as the mean residual after the new weight
    # ------------------------------------------------------------------
    residuals = Y - X @ new_head_weight.T               # (4000,2)
    new_head_bias = residuals.mean(axis=0).astype(np.float32)   # (2,)

    # ------------------------------------------------------------------
    # 4. Write the updated parameters back into the weight dict
    # ------------------------------------------------------------------
    w["head.weight"] = new_head_weight
    w["head.bias"]   = new_head_bias

    # ------------------------------------------------------------------
    # 5. Diagnostics (tiny JSON‑compatible dict) – useful for later reflection
    # ------------------------------------------------------------------
    diag = {
        "head_weight_fro_norm": float(np.linalg.norm(new_head_weight)),
        "head_bias":       new_head_bias.tolist(),
        "ridge_lambda":    float(lam),
        "features_mean":   float(X.mean()),
        "features_std":    float(X.std()),
    }

    # Return the modified weight dict together with diagnostics.
    # The caller may keep the dict or unpack the "weights" key.
    return {"weights": w, "diagnostics": diag}
