def edit(w, f):
    """
    Edit the network weights using a closed‑form ridge‑regression fit on the full
    feedback pool. Only the head layer is modified; convolutional weights remain
    unchanged so downstream cached features stay valid.
    """
    import numpy as np

    # ----- data from feedback arrays -----
    features = f["features"]                 # shape (2000, 1568), float32
    y        = f["y"]                        # shape (2000,), int64

    # ----- ridge‑regression parameters -----
    lam = np.float32(0.001)                  # small L2 regularisation

    # one‑hot encode labels -> (2000, 10)
    Y_onehot = np.eye(10, dtype=np.float32)[y]

    # ----- compute head.weight via normal equation -----
    # FtF = XᵀX + λI   (1568 × 1568)
    FtF = features.T @ features + lam * np.eye(1568, dtype=np.float32)
    # FtY = XᵀY        (1568 × 10)
    FtY = features.T @ Y_onehot

    # solve (1568 × 1568) * coeff = FtY  → coeff shape (1568, 10)
    coeff = np.linalg.solve(FtF, FtY)

    # transpose to match head.weight shape (10, 1568)
    head_weight_new = coeff.T.astype(np.float32)

    # ----- compute head.bias -----
    # bias = mean( Y_onehot - X @ W_new )
    bias_new = (Y_onehot - features @ head_weight_new).mean(axis=0).astype(np.float32)  # shape (10,)

    # ----- apply the new weights (in‑place) -----
    w["head.weight"][:] = head_weight_new
    w["head.bias"][:]   = bias_new

    # ----- diagnostics -----
    # predicted labels on the feedback pool
    pred_logits = features @ head_weight_new + bias_new          # (2000, 10)
    pred_labels = np.argmax(pred_logits, axis=1)               # (2000,)
    train_accuracy = float(np.mean(pred_labels == y))

    diag = {
        "train_accuracy": train_accuracy,
        "head_weight_mean": float(head_weight_new.mean()),
        "head_bias_mean":   float(bias_new.mean()),
        "lambda":          float(lam)
    }

    return {"weights": w, "diagnostics": diag}
