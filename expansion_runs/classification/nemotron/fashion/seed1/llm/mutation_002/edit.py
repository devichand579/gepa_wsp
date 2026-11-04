import numpy as np

def edit(w, f):
    """
    Closed‑form ridge‑regression update of the network's head (linear classifier)
    using the full feedback pool (f["features"], f["y"]).  All other parameters
    are left unchanged.  Returns the modified weight dict and a tiny diagnostics
    block.
    """
    # ------- data -------
    X = f["features"]                # shape (2000, 1568), float32
    y = f["y"]                       # shape (2000,), int64  (class indices)

    # one‑hot encode labels -> (2000, 10)
    Y = np.zeros((2000, 10), dtype=np.float32)
    Y[np.arange(2000), y] = 1.0

    # ------- ridge regularisation -------
    lam = 1e-3                       # small L2 penalty; can be tuned
    I = np.eye(1568, dtype=np.float32) * lam

    # XᵀX + λI   (1568 × 1568)
    XtX = X.T @ X + I

    # XᵀY       (1568 × 10)
    XtY = X.T @ Y

    # solve (XᵀX + λI) W = XᵀY  →  W has shape (1568, 10)
    W_raw = np.linalg.solve(XtX, XtY)     # numerically stable closed‑form

    # The network expects head.weight of shape [10, 1568]
    w["head.weight"] = W_raw.T.astype(np.float32)

    # ------- bias update -------
    # New bias per class = mean( Y - X @ Wᵀ ) over the 2000 samples
    preds = X @ W_raw.T                     # (2000, 10)
    new_bias = np.mean(Y - preds, axis=0).astype(np.float32)
    w["head.bias"] = new_bias

    # ------- diagnostics (tiny JSON‑compatible) -------
    diag = {
        "ridge_lambda": float(lam),
        "head_weight_norm": float(np.linalg.norm(w["head.weight"])),
        "head_bias": w["head.bias"].tolist()
    }

    return {"weights": w, "diagnostics": diag}
