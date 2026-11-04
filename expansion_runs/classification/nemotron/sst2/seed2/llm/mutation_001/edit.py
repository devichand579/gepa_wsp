def edit(w, f):
    """
    Closed‑form ridge‑regression update of the final head weights and bias
    using the entire feedback pool (f["features"] and f["y"]).
    All other parameters in w are left untouched.
    """
    import numpy as np

    # ----- data -------------------------------------------------
    X = f["features"]                # shape (4000, 32), float32
    y = f["y"].astype(np.int64)      # shape (4000,), int labels 0/1

    # one‑hot encode targets for both classes
    Y_onehot = np.eye(2, dtype=np.float32)[y]   # (4000, 2)

    # ----- ridge regularisation ---------------------------------
    lam = np.float32(1e-3)            # small ridge coefficient

    # ---- head weight (2,32) ------------------------------------
    # Solve (XᵀX + λI) w = XᵀY  for each output dimension.
    XtX = X.T @ X + lam * np.eye(32, dtype=np.float32)   # (32,32)
    Xty = X.T @ Y_onehot                               # (32,2)
    # linalg.solve returns (32,2); transpose to (2,32) as expected
    head_weight = np.linalg.solve(XtX, Xty).T         # (2,32), float32

    # ---- head bias (2,) ---------------------------------------
    # Add an intercept column to X
    Xb = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float32)])   # (4000,33)
    Xtb = Xb.T @ Xb + lam * np.eye(33, dtype=np.float32)            # (33,33)
    Xtbb = Xb.T @ Y_onehot                                      # (33,2)
    # solve for augmented coefficients, then split weight & bias
    ab = np.linalg.solve(Xtb, Xtbb).T          # (2,33) -> first 32 cols = weight, last = bias
    head_bias = ab[:, 32].astype(np.float32)   # (2,)

    # ----- apply updates -----------------------------------------
    w["head.weight"] = head_weight.astype(np.float32)
    w["head.bias"]   = head_bias.astype(np.float32)

    # ----- diagnostics (tiny JSON‑compatible dict) -------------
    diag = {
        "ridge_lambda": float(lam),
        "head_weight_norm": float(np.linalg.norm(head_weight)),
        "head_bias": head_bias.tolist(),
        "features_shape": list(X.shape),
        "n_feedback": X.shape[0],
    }

    return {"weights": w, "diagnostics": diag}
