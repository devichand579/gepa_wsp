def edit(w, f):
    """
    Edit the network weights using a single closed‑form ridge‑regression step
    on the full feedback pool. Only fc2 parameters are changed; fc1 stays as‑is.
    Returns the modified weight dict and a small diagnostics block.
    """
    # ------- data from feedback ----------
    features = f["features"]                     # (2000, 128), post‑fc1 activations
    y        = f["y"]                            # (2000,), int64 class indices
    one_hot  = np.eye(10, dtype=np.float32)[y]  # (2000, 10) one‑hot targets

    # ------- ridge regularisation ----------
    lam = np.float32(1e-3)                       # small ridge lambda
    I128  = np.eye(128, dtype=np.float32)

    # Normal‑equation matrices
    XtX = features.T @ features + lam * I128      # (128, 128)
    XtY = features.T @ one_hot                   # (128, 10)

    # Closed‑form solve (no gradient, no loop)
    weight_128_10 = np.linalg.solve(XtX, XtY)   # (128, 10)

    # Reshape to PyTorch‑style [out, in] = [10, 128]
    fc2_weight_new = weight_128_10.T.astype(np.float32)

    # Bias from mean residual
    residuals = one_hot - features @ weight_128_10   # (2000, 10)
    fc2_bias_new = residuals.mean(axis=0).astype(np.float32)   # (10,)

    # ------- apply updates in‑place ----------
    w["fc2.weight"][:] = fc2_weight_new
    w["fc2.bias"][:]   = fc2_bias_new

    # ------- diagnostics on reflection batch ----------
    h_batch = f["layer.fc1.output"]               # (128, 128) pre‑edit activations
    logits_batch = h_batch @ fc2_weight_new.T + fc2_bias_new   # (128, 10)
    pred_batch = np.argmax(logits_batch, axis=1)
    batch_acc = float(np.mean(pred_batch == f["batch.y"].astype(np.int64)))

    diag = {
        "fc2_weight_norm": float(np.linalg.norm(fc2_weight_new)),
        "fc2_bias_norm":   float(np.linalg.norm(fc2_bias_new)),
        "batch_accuracy_after_edit": batch_acc,
        "ridge_lambda":    float(lam)
    }

    return {"weights": w, "diagnostics": diag}
