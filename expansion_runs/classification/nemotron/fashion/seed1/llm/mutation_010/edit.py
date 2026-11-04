def edit(w, f):
    # ----- full training pool -----
    X = f["features"]                     # shape (2000, 1568), float32
    y = f["y"]                            # shape (2000,), int64

    # one‑hot encode labels → (2000, 10)
    Y = np.eye(10, dtype=np.float32)[y]

    # ridge regularisation (small constant)
    lam = 1e-3
    in_size = X.shape[1]                  # 1568
    out_size = 10

    # ----- normal equations (ridge) -----
    XtX = X.T @ X                         # (1568, 1568)
    XtY = X.T @ Y                         # (1568, 10)

    # solve (XᵀX + λI) coeff = XᵀY  → coeff shape (1568, 10)
    coeff = np.linalg.solve(XtX + lam * np.eye(in_size), XtY)

    # head.weight is (out, in) → transpose coeff
    new_weight = coeff.T.astype(np.float32)   # (10, 1568)

    # ----- bias from mean residual -----
    logits_pred = X @ new_weight.T          # (2000, 10)
    new_bias = (Y - logits_pred).mean(axis=0).astype(np.float32)   # (10,)

    # ----- edit the weight dictionary -----
    w["head.weight"] = new_weight
    w["head.bias"] = new_bias

    # ----- diagnostics (JSON‑compatible) -----
    diag = {
        "head_weight_mean": float(new_weight.mean()),
        "head_weight_std": float(new_weight.std()),
        "head_bias_mean": float(new_bias.mean()),
        "head_bias_std": float(new_bias.std()),
        "train_accuracy": float((logits_pred.argmax(axis=1) == y).mean()),
        "cross_entropy": float(
            -(Y * np.log(logits_pred + 1e-12)).sum(axis=1).mean()
        ),
    }

    return {"weights": w, "diagnostics": diag}
