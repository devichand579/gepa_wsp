def edit(w, f):
    # --- feedback arrays -------------------------------------------------
    features = f["features"]          # shape (2000, 128)
    y        = f["y"]                 # shape (2000,), int64 labels 0‑9

    K = 10                            # number of classes
    Y = np.eye(K, dtype=np.float32)[y]   # (2000,10) one‑hot targets

    # --- ridge‑regression (closed‑form) -------------------------------
    lam = 1e-3                         # small ridge constant
    I   = np.eye(features.shape[1], dtype=np.float32) * lam
    A   = features.T @ features + lam * np.eye(features.shape[1], dtype=np.float32)   # (128,128)
    B   = features.T @ Y                # (128,10)

    Wt  = np.linalg.solve(A, B)         # solve A @ Wt = B  => Wt shape (128,10)
    W   = Wt.T                            # (10,128) – fc2 weight matrix

    # --- bias term ------------------------------------------------------
    residuals = Y - (features @ W.T)    # (2000,10)
    b = residuals.mean(axis=0)           # (10,) – per‑class bias offset

    # --- apply edits ------------------------------------------------------
    w["fc2.weight"] = W.astype(np.float32)       # shape [10, 128]
    w["fc2.bias"]   = b.astype(np.float32)       # shape [10]

    # --- diagnostics ------------------------------------------------------
    diag = {
        "fc2_weight_mean":  float(np.mean(W)),
        "fc2_weight_std":   float(np.std(W)),
        "fc2_bias_mean":    float(np.mean(b)),
    }
    return {"weights": w, "diagnostics": diag}
