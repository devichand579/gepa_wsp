def edit(w, f):
    # reflection batch indices into the full feedback pool
    ids = f["batch.ids"]                     # shape [128], int64
    X = f["features"][ids]                   # shape [128, 128], float32, post-fc1 activations
    y = f["y"][ids]                          # shape [128], int64
    
    # one‑hot encode the reflection‑batch labels
    Y = np.eye(10, dtype=np.float32)[y]      # shape [128, 10]
    
    # ridge‑regression normal equation: (XᵀX + λI) Ŵᵀ = XᵀY
    lam = 0.1                                # regularization coefficient (closed‑form, permitted)
    I128 = np.eye(128, dtype=np.float32)
    Wt = np.linalg.solve(X.T @ X + lam * I128, X.T @ Y)  # shape [128, 10]
    W_new = Wt.T.astype(np.float32)          # shape [10, 128], matches fc2.weight
    
    # bias from residuals on the reflection batch
    pred_logits = X @ W_new.T                # shape [128, 10]
    b_new = (Y - pred_logits).mean(axis=0).astype(np.float32)  # shape [10], matches fc2.bias
    
    # edit the fixed architecture weights in‑place
    w["fc2.weight"] = W_new
    w["fc2.bias"] = b_new
    
    # diagnostics (JSON‑compatible, small)
    acc = float(np.mean(np.argmax(pred_logits, axis=1) == y))
    diag = {
        "batch_accuracy": acc,
        "ridge_lambda": float(lam),
        "weight_norm": float(np.linalg.norm(W_new)),
        "bias_norm": float(np.linalg.norm(b_new)),
        "weight_mean": float(W_new.mean()),
        "bias_mean": float(b_new.mean())
    }
    return {"weights": w, "diagnostics": diag}
