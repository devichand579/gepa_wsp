def edit(w, f):
    """
    Update fc2.weight and fc2.bias via a single closed‑form ridge‑regression
    on the full 2000‑sample feedback pool. All other parameters (fc1) are left
    untouched. Returns a dict with the modified weights and a small diagnostics
    dictionary.
    """
    import numpy as np

    # ----- feedback pool -------------------------------------------------
    features = f["features"]                # shape (2000, 128), float32
    y        = f["y"]                       # shape (2000,), int64

    # one‑hot encode labels
    Y = np.eye(10, dtype=np.float32)[y]    # (2000, 10)

    # ridge regularisation (small constant)
    lam = 0.1

    # ----- ridge regression for fc2.weight -------------------------------
    # Solve (features^T @ features + λ I) W^T = features^T @ Y   for W^T
    Gram = features.T @ features + lam * np.eye(128, dtype=np.float32)   # (128,128)
    A    = features.T @ Y                                          # (128,10)
    WT   = np.linalg.solve(Gram, A)                               # (128,10)
    W_new = WT.T                                                    # (10, 128)  → matches fc2.weight shape

    # ----- bias -----------------------------------------------------------
    # Compute bias as the mean residual after the weight update
    Y_hat = features @ W_new.T                                     # (2000, 10)
    b = np.mean(Y - Y_hat, axis=0)                                 # (10,)
    b = np.clip(b, -1000, 1000).astype(np.float32)

    # ----- apply updates --------------------------------------------------
    w["fc2.weight"] = W_new.astype(np.float32)
    w["fc2.bias"]   = b

    # ----- diagnostics ----------------------------------------------------
    # Batch accuracy after edit using cached batch activations
    batch_features = f["layer.fc1.output"]          # shape (128, 128)
    logits_batch   = batch_features @ W_new.T + b   # (128, 10)
    preds          = np.argmax(logits_batch, axis=1)
    acc            = float(np.mean(preds == f["batch.y"].astype(np.int32)))

    diag = {
        "fc2_weight_norm": float(np.linalg.norm(W_new, ord='fro')),
        "fc2_bias_norm":   float(np.linalg.norm(b)),
        "batch_accuracy_after_edit": acc,
        "ridge_lambda":   float(lam)
    }

    return {"weights": w, "diagnostics": diag}
