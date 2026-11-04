def edit(w, f):
    """
    Closed‑form ridge‑regression update of the network head using the
    provided feedback arrays.  Only the head weights/bias are changed;
    all conv layers stay untouched.
    """
    import numpy as np

    # work on a copy so the original dict is not mutated
    w = {k: v.copy() for k, v in w.items()}

    # ----- data -----
    X = f["features"]                # (2000, 1568) float32
    y = f["y"]                      # (2000,) int64, values 0‑9
    # one‑hot targets
    Y = np.eye(10, dtype=np.float32)[y]   # (2000, 10)

    # ----- ridge regularisation -----
    lam = 1e-3                       # small Tikhonov coefficient
    d = X.shape[1]                   # 1568

    # normal‑equation matrices
    XtX = X.T @ X                    # (1568, 1568)
    XtX_reg = XtX + lam * np.eye(d, dtype=np.float32)
    XtY = X.T @ Y                    # (1568, 10)

    # closed‑form solution: (XᵀX + λI)^{-1} XᵀY
    W_coeff = np.linalg.solve(XtX_reg, XtY)   # (1568, 10)

    # ----- replace head parameters -----
    # head.weight shape (10, 1568)  →  rows are the coefficient vectors
    w["head.weight"] = W_coeff.T.astype(np.float32)

    # bias: mean residual after the linear fit
    logits_pred = X @ W_coeff.T          # (2000, 10)
    bias = np.mean(Y - logits_pred, axis=0).astype(np.float32)
    w["head.bias"] = bias

    # ----- diagnostics -----
    preds = np.argmax(logits_pred, axis=1)
    train_acc = float(np.mean(preds == y))

    diag = {
        "train_accuracy": train_acc,
        "head_weight_norm": float(np.linalg.norm(w["head.weight"])),
        "head_bias": w["head.bias"].tolist(),
    }

    return {"weights": w, "diagnostics": diag}
