def edit(w, f):
    """
    Closed‑form ridge‑regression update of the final head layer.
    - X = f["features"]   shape (4000, 32)
    - y = f["y"]          shape (4000,)   int labels {0,1}
    - lambda = 1e-3 (small ridge)
    """
    X = f["features"]            # (4000, 32)
    y = f["y"].astype(np.float32)  # (4000,)

    lam = 1e-3
    I32 = np.eye(32, dtype=np.float32)

    # Ridge solution: w = (X^T X + λI)^{-1} X^T y   (shape (32,))
    w = np.linalg.solve(X.T @ X + lam * I32, X.T @ y)

    # Build head weight [2, 32] : row0 = +w, row1 = -w
    head_weight = np.array([w, -w], dtype=np.float32)   # (2, 32)

    # Bias so that average logits are zero: b0 = -mean(X@w), b1 = +mean(X@w)
    logits_center = X @ w                               # (4000,)
    head_bias = np.array([-float(np.mean(logits_center)),
                          float(np.mean(logits_center))],
                         dtype=np.float32)           # (2,)

    # Edit the dictionary in‑place, preserving shapes
    w["head.weight"] = head_weight
    w["head.bias"]   = head_bias

    # Optional diagnostics (tiny JSON‑compatible dict)
    diag = {
        "head_weight_norm": float(np.linalg.norm(head_weight)),
        "head_bias":      head_bias.tolist(),
        "ridge_lambda":   lam,
        "features_mean":  float(np.mean(X, axis=0)),
    }
    return {"weights": w, "diagnostics": diag}
