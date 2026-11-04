def edit(w, f):
    """
    One‑shot, closed‑form fit of the network's head weights using the full
    permitted feedback pool.  Only NumPy linear algebra (ridge regression) is
    used – no autograd, back‑prop, or iterative optimisation.
    """
    import numpy as np

    # ----- feedback pool -------------------------------------------------
    X = f["features"]                # shape (2000, 1568)
    y = f["y"]                       # shape (2000,), int64, values 0‑9
    N = X.shape[0]

    # one‑hot targets (N, 10)
    Y = np.eye(10, dtype=np.float32)[y]

    # ridge regularisation (small constant)
    lam = np.float32(1e-3)

    # augment features with a constant 1 for bias learning
    X_aug = np.hstack([X, np.ones((N, 1), dtype=np.float32)])   # (N, 1569)

    # covariance of augmented features + ridge
    C_aug = (X_aug.T @ X_aug) / N + lam * np.eye(1569, dtype=np.float32)

    # cross‑covariance between augmented features and one‑hot targets
    Cy_aug = X_aug.T @ Y / N                                      # (1569, 10)

    # closed‑form solution: A = (XᵀX+λI)⁻¹ XᵀY   →   A shape (1569, 10)
    A = np.linalg.solve(C_aug, Cy_aug.T).T   # (10, 1569)

    # split into weight matrix and bias vector
    head_weight = A[:, :-1].astype(np.float32)   # (10, 1568)
    head_bias   = A[:, -1].astype(np.float32)    # (10,)

    # ----- edit the weight dictionary ------------------------------------
    w["head.weight"] = head_weight
    w["head.bias"]   = head_bias

    # ----- diagnostics ----------------------------------------------------
    # training‑pool accuracy after the edit
    logits = X @ head_weight.T + head_bias          # (2000, 10)
    preds  = np.argmax(logits, axis=1)
    train_acc = float(np.mean(preds == y))

    diag = {
        "train_accuracy": train_acc,
        "ridge_lambda": float(lam),
        "head_weight_norm": float(np.linalg.norm(head_weight)),
        "head_bias": head_bias.tolist()
    }

    return {"weights": w, "diagnostics": diag}
