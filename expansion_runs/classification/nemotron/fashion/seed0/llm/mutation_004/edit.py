import numpy as np

def edit(w, f):
    """
    Closed‑form ridge‑regression fit of the head layer using the full
    permitted feedback pool.  Only `head.weight` and `head.bias` are
    modified; all other parameters stay untouched.
    """
    # ----- data from feedback -----
    X = f["features"]                # shape (2000, 1568), float32
    y = f["y"].astype(np.int64)      # shape (2000,), int64 labels 0‑9

    # one‑hot encode targets
    Y = np.eye(10, dtype=np.float32)[y]   # (2000, 10)

    # ridge regularisation (small lambda to avoid over‑fit)
    lam = np.float32(0.001)

    # ----- ridge solution for head.weight (10 × 1568) -----
    # Gram matrix XᵀX  (1568 × 1568)
    GTX = X.T @ X                                 # (1568,1568)
    GTX_reg = GTX + lam * np.eye(1568, dtype=np.float32)

    # XᵀY  (1568 × 10)
    GTY = X.T @ Y                                 # (1568,10)

    # solve (GTX_reg) @ Wᵀ = GTY  →  W_new shape (10,1568)
    W_new = np.linalg.solve(GTX_reg, GTY).T      # (10,1568) back to weight layout
    W_new = W_new.astype(np.float32)

    # ----- head.bias -----
    # compute bias as mean of residual targets across the pool
    resid = Y - X @ W_new                         # (2000,10)
    bias = np.mean(resid, axis=0).astype(np.float32)   # (10,)

    # ----- apply edits -----
    w["head.weight"] = W_new
    w["head.bias"]   = bias

    # ----- diagnostics -----
    # predicted logits and accuracy on the feedback pool
    logits_new = X @ W_new + bias                 # (2000,10)
    pred_labels = np.argmax(logits_new, axis=1)
    acc = float(np.mean(pred_labels == y))

    diag = {
        "head_weight_mean": float(np.mean(W_new)),
        "head_weight_std":  float(np.std(W_new)),
        "head_bias_mean":   float(np.mean(bias)),
        "train_accuracy":   acc,
        "lambda":           float(lam),
    }

    return {"weights": w, "diagnostics": diag}
