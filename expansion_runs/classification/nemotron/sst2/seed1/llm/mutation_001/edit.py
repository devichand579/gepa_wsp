def edit(w, f):
    # --- extract permitted feedback arrays ---
    X = f["features"].astype(np.float32)          # [4000, 32]
    y = f["y"].astype(np.int64)                   # [4000]

    # --- one‑hot encode labels ---
    Y = np.zeros((len(y), 2), dtype=np.float32)
    Y[y == 1, 1] = 1.0
    Y[y == 0, 0] = 1.0

    # --- closed‑form ridge regression: Y ≈ X @ W + b ---
    lam = np.float32(1e-4)                        # small ridge regularisation
    XtX = X.T @ X + lam * np.eye(32, dtype=np.float32)
    XtY = X.T @ Y                                 # [32, 2]
    W = np.linalg.solve(XtX, XtY)               # [32, 2]  (W[i,j] = weight from feature j to class i)

    # --- bias as mean residual ---
    resid = Y - (X @ W)
    b = np.mean(resid, axis=0, dtype=np.float32)  # [2]

    # --- update head parameters, preserving shapes ---
    w["head.weight"] = W.T.astype(np.float32)    # [2, 32]
    w["head.bias"]   = b.astype(np.float32)      # [2]

    # --- diagnostics (JSON‑compatible) ---
    preds = X @ W + b
    pred_labels = np.argmax(preds, axis=1)
    acc = float(np.mean(pred_labels == y))
    ce  = float(-np.mean(np.log(preds[np.arange(len(y)), y] + 1e-8)))

    diag = {
        "head_weight_mean": float(np.mean(W.T)),
        "head_weight_std":  float(np.std(W.T)),
        "head_bias_mean":   float(np.mean(b)),
        "head_bias_std":    float(np.std(b)),
        "fit_accuracy":     acc,
        "fit_cross_entropy": ce,
        "num_examples":     int(len(y))
    }

    return {"weights": w, "diagnostics": diag}
