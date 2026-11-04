def edit(w, f):
    """
    Closed‑form ridge‑regression fit of the final head layer using the
    entire feedback pool (f["features"], f["y"]).  All other parameters
    in w are left unchanged.
    """
    import numpy as np

    # ----- data -------------------------------------------------------
    X = f["features"]                     # shape (4000, 32), float32
    y = f["y"]                            # shape (4000,), int 0/1

    # one‑hot labels (float32)
    Y = np.eye(2, dtype=np.float32)[y]    # shape (4000, 2)

    # ridge parameter (chosen small enough to stabilise but not dominate)
    lam = np.float32(0.1)

    N, D = X.shape                        # N=4000, D=32

    # ----- closed‑form solution ---------------------------------------
    #   minimise ||X w - Y||_F^2 + lam * ||w||_F^2
    #   normal equation: (XᵀX + lam I) w = Xᵀ Y
    A = X.T @ X + lam * np.eye(D, dtype=np.float32)   # (32,32)
    B = X.T @ Y                                       # (32,2)

    # solve for un‑transposed weight matrix (32,2)
    weight_raw = np.linalg.solve(A, B)               # (32,2)

    # head.weight is (2,32), so transpose back
    new_weight = weight_raw.astype(np.float32).T      # (2,32)

    # ----- bias term ------------------------------------------------
    X_mean = X.mean(axis=0, keepdims=True)           # (1,32)
    Y_mean = Y.mean(axis=0, keepdims=True)           # (1,2)
    # bias so that linear model matches label means on average
    new_bias = (Y_mean - new_weight @ X_mean).ravel().astype(np.float32)  # (2,)

    # ----- assign to weight dictionary -------------------------------
    w["head.weight"] = new_weight
    w["head.bias"]   = new_bias

    # ----- diagnostics -----------------------------------------------
    logits = X @ new_weight.T + new_bias             # (4000,2)
    pred   = np.argmax(logits, axis=1)
    acc    = (pred == y).mean()

    diagnostics = {
        "ridge_lambda": float(lam),
        "head_weight_norm": float(np.linalg.norm(new_weight)),
        "head_bias": new_bias.tolist(),
        "feedback_accuracy": float(acc)
    }

    return {"weights": w, "diagnostics": diagnostics}
