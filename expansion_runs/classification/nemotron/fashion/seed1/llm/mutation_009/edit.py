def edit(w, f):
    """
    Closed‑form ridge‑regression update of the network head.
    w  – full weight dict (will be modified in‑place then returned)
    f  – feedback dict containing:
         f["features"]  shape (2000, 1568)   final hidden activations
         f["y"]         shape (2000,)        integer labels 0‑9
    No autograd / iterative optimisation is used.
    """
    import numpy as np

    X = f["features"].astype(np.float32)          # (2000, 1568)
    y = f["y"].astype(np.int64)                   # (2000,)

    # one‑hot encode labels → (2000, 10)
    Y = np.eye(10, dtype=np.float32)[y]

    # ridge parameter (small λ stabilises the normal equations)
    lam = 1e-3
    I1568 = np.eye(1568, dtype=np.float32)

    # normal equations: (XᵀX + λI) w = XᵀY
    A = X.T @ X + lam * I1568          # (1568, 1568)
    b = X.T @ Y                        # (1568, 10)

    # solve for coefficients (1568,10) then transpose to (10,1568)
    head_coef = np.linalg.solve(A, b)  # (1568, 10)
    new_head_weight = head_coef.T.astype(np.float32)   # (10, 1568)

    # bias = mean residual across the training set
    residuals = Y - X @ head_coef      # (2000, 10)
    new_head_bias = np.mean(residuals, axis=0).astype(np.float32)  # (10,)

    # replace head parameters, keep everything else unchanged
    w["head.weight"] = new_head_weight
    w["head.bias"] = new_head_bias

    # lightweight diagnostics (JSON‑compatible)
    diag = {
        "head_weight_mean": float(np.mean(new_head_weight)),
        "head_weight_std":  float(np.std(new_head_weight)),
        "head_bias_mean":   float(np.mean(new_head_bias)),
        "head_bias_std":    float(np.std(new_head_bias)),
    }

    return {"weights": w, "diagnostics": diag}
