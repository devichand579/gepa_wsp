def edit(w, f):
    """
    Edit the final head weights via closed‑form ridge regression on the
    entire feedback pool.  No autograd, back‑prop, or iterative optimisation
    is used – a single analytic solve step.
    """
    import numpy as np

    # ----- data ---------------------------------------------------------
    X = f["features"]                     # shape (4000, 32), float32
    y = f["y"]                            # shape (4000,), int 0/1
    N = X.shape[0]

    # one‑hot labels (2 classes)
    Y = np.eye(2, dtype=np.float32)[y]    # (4000, 2)

    # augment features with a constant 1 so bias is learned together
    X_aug = np.hstack([X, np.ones((N, 1), dtype=np.float32)])  # (4000, 33)

    # ----- ridge regression ---------------------------------------------
    lam = 1e-3                              # small ridge penalty
    G = X_aug.T @ X_aug + lam * np.eye(33, dtype=np.float32)   # (33,33)
    C = X_aug.T @ Y                         # (33, 2)
    A = np.linalg.solve(G, C)              # (33, 2)  – analytic solution

    # split into weight matrix and bias vector
    head_weight = A[:32].T.astype(np.float32)   # (2, 32)
    head_bias   = A[32].astype(np.float32)      # (2,)

    # ----- apply edits ---------------------------------------------------
    w["head.weight"][:] = head_weight
    w["head.bias"][:]   = head_bias

    # ----- diagnostics (optional) ---------------------------------------
    diagnostics = {
        "ridge_lambda": float(lam),
        "head_weight_norm": float(np.linalg.norm(head_weight)),
        "head_bias": head_bias.tolist(),
        "feedback_n": int(N),
        "feedback_accuracy_before": float(np.mean(
            np.argmax(X @ head_weight.T + head_bias, axis=1) == y)
        ),  # simple in‑sample check
    }

    return {"weights": w, "diagnostics": diagnostics}
