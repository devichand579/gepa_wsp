def edit(w, f):
    """
    Closed‑form ridge‑regression update of the final head layer
    using the entire permitted feedback pool.
    """
    import numpy as np

    # ----- feedback data -----
    X = f["features"].astype(np.float32)          # shape (4000, 32)
    y = f["y"].astype(np.int64)                   # shape (4000,)

    # one‑hot encode labels into (4000, 2)
    Y = np.eye(2, dtype=np.float32)[y]           # shape (4000, 2)

    # ridge regularisation strength (chosen small enough not to dominate)
    lam = 0.1

    # ----- normal equations -----
    XtX = X.T @ X                                 # (32, 32)
    XtY = X.T @ Y                                 # (32, 2)
    I32 = np.eye(32, dtype=np.float32)
    # solve (XᵀX + λI) W = XᵀY  for W of shape (32, 2)
    W_coeff = np.linalg.solve(XtX + lam * I32, XtY)   # (32, 2)

    # ----- update head parameters -----
    # head.weight was [2, 32]; we store the transpose of the fitted matrix
    w["head.weight"] = W_coeff.T.astype(np.float32)   # shape (2, 32)

    # bias: mean residual after the linear map
    residuals = Y - X @ W_coeff          # (4000, 2)
    b = np.mean(residuals, axis=0).astype(np.float32)   # shape (2,)
    w["head.bias"] = b

    # ----- diagnostics -----
    diag = {
        "ridge_lambda": float(lam),
        "head_weight_norm": float(np.linalg.norm(w["head.weight"])),
        "head_bias": w["head.bias"].tolist(),
        "features_mean": X.mean(axis=0).tolist(),
        "y_counts": {int(k): int(v) for k, v in dict(zip(*np.unique(y, return_counts=True))).items()}
    }

    return {"weights": w, "diagnostics": diag}
