def edit(w, f):
    """
    Closed‑form ridge‑regression fit of fc2 to the full feedback pool.
    Returns the updated weight dict together with a tiny diagnostics block.
    """
    import numpy as np

    # ----- feedback pool -------------------------------------------------
    X = f["features"]                     # shape (2000, 128), float32
    y = f["y"]                            # shape (2000,), int64, values 0‑9
    C = 10                                # number of classes

    # one‑hot targets
    Y_onehot = np.eye(C, dtype=np.float32)[y]   # (2000, 10)

    # ridge regularisation (small scalar)
    lam = 1e-3 * np.var(X)            # scales with feature variance
    Gram = X.T @ X                    # (128, 128)
    reg  = Gram + lam * np.eye(128, dtype=np.float32)

    # solve (XᵀX + λI) Wᵀ = XᵀY  →  W_raw shape (128, 10)
    XTY = X.T @ Y_onehot              # (128, 10)
    W_raw = np.linalg.solve(reg, XTY) # (128, 10)

    # fc2.weight is [out, in] = (10, 128)
    weight = W_raw.T.astype(np.float32)   # (10, 128)

    # bias: mean residual after the linear map
    residuals = Y_onehot - X @ weight.T   # (2000, 10)
    bias = residuals.mean(axis=0).astype(np.float32)   # (10,)

    # ----- apply the new parameters ---------------------------------------
    w["fc2.weight"] = weight
    w["fc2.bias"]   = bias

    # ----- evaluate on the feedback pool (no autograd, pure numpy) -------
    logits_new = X @ weight.T + bias       # (2000, 10)
    pred       = np.argmax(logits_new, axis=1)
    acc        = float(np.mean(pred == y))

    # diagnostics (JSON‑compatible, small)
    diag = {
        "fc2_weight_norm": float(np.sum(weight ** 2)),
        "fc2_bias_mean":   float(np.mean(bias)),
        "train_accuracy":  acc,
        "train_cross_entropy": float(
            -np.mean(np.log(logits_new[np.arange(2000), y] + 1e-8))
        ),
    }

    return {"weights": w, "diagnostics": diag}
