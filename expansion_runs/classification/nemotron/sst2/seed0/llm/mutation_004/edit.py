def edit(w, f):
    """
    Closed‑form ridge‑regression update of the final head layer
    using the full feedback pool (features + labels). All other
    parameters are left untouched.
    """
    import numpy as np

    # --- feedback data -------------------------------------------------
    X = f["features"].astype(np.float32)          # shape (4000, 32)
    y = f["y"].astype(np.float32)                # shape (4000,), 0/1

    # ridge coefficient (small stabiliser)
    lam = 0.1
    I = np.eye(32, dtype=np.float32) * lam

    # --- ridge solution for a single weight vector --------------------
    # we map features -> label; the two head outputs will use +w and -w
    XtX = X.T @ X + I            # (32,32)
    Xty = X.T @ y              # (32,)
    w_vec = np.linalg.solve(XtX, Xty)   # (32,)

    # --- bias term ----------------------------------------------------
    y_pred = X @ w_vec                 # (4000,)
    bias = float(np.mean(y - y_pred))  # scalar

    # --- build new head parameters -------------------------------------
    w_new = {k: v.copy() for k, v in w.items()}   # shallow copy all weights

    # head.weight shape (2,32); rows are [+w_vec, -w_vec]
    w_new["head.weight"] = np.array([w_vec, -w_vec], dtype=np.float32)

    # head.bias shape (2); class‑1 gets +bias, class‑0 gets -bias
    w_new["head.bias"] = np.array([-bias, bias], dtype=np.float32)

    # all other entries of w_new remain exactly as in the original w
    return {"weights": w_new}
