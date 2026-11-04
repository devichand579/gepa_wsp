def edit(w, f):
    """
    Edit the network's head weights via a closed‑form ridge regression
    from the provided frozen features and labels. All other parameters
    remain unchanged.
    """
    # ----- feedback data -----
    X = f["features"]                # shape (2000, 1568), float32
    y_idx = f["y"]                   # shape (2000,), int64 labels 0‑9
    C = 10
    Y = np.eye(C)[y_idx]             # one‑hot (2000, 10), float32

    # ----- ridge regularisation -----
    lam = 1e-3                       # small ridge term

    N, D = X.shape                   # D = 1568
    # augment features with a constant 1 for the bias term
    Xb = np.hstack([X, np.ones((N, 1), dtype=np.float32)], axis=1)  # (2000, 1569)

    # Gram matrix (D+1 × D+1) with ridge
    Gram = Xb.T @ Xb + lam * np.eye(D + 1, dtype=np.float64)

    # cross‑term Xb^T Y  (D+1 × C)
    XTY = Xb.T @ Y

    # closed‑form solve for combined weight+bias per class
    WT = np.linalg.solve(Gram, XTY)   # (1569, 10)

    # split into head.weight and head.bias
    head_weight = WT[:D].T.astype(np.float32)   # (10, 1568)
    head_bias   = WT[D, :].astype(np.float32)   # (10,)

    # copy current weights and replace only the head
    new_w = {k: v.copy() for k, v in w.items()}
    new_w["head.weight"] = head_weight
    new_w["head.bias"]   = head_bias

    # ----- diagnostics (small JSON‑compatible dict) -----
    diag = {
        "head_weight_norm": float(np.linalg.norm(head_weight)),
        "head_bias": head_bias.tolist(),
        "ridge_lambda": float(lam),
    }
    return {"weights": new_w, "diagnostics": diag}
