def edit(w, f):
    import numpy as np

    # ----- feedback data -----
    X = f["features"].astype(np.float32)            # [2000, 1568]
    y = f["y"].astype(np.int64)                     # [2000]

    # one‑hot encode labels
    Y = np.eye(10, dtype=np.float32)[y]             # [2000, 10]

    # ----- ridge regularisation -----
    lam = 1e-3                                      # small ridge term
    d = X.shape[1]                                  # 1568
    XtX = X.T @ X + lam * np.eye(d, dtype=np.float32)   # [1568,1568]
    XtY = X.T @ Y                                    # [1568,10]

    # closed‑form solve (no autograd, no iterative optimizer)
    W_raw = np.linalg.solve(XtX, XtY)              # [1568,10]
    head_weight = W_raw.T.astype(np.float32)       # [10,1568]  matches head.weight shape

    # bias: mean residual after fitting weight
    pred_logits = X @ W_raw.T                        # [2000,10]
    residual = Y - pred_logits
    head_bias = residual.mean(axis=0).astype(np.float32)  # [10]

    # ----- apply edits, preserve shapes -----
    w["head.weight"] = head_weight
    w["head.bias"]   = head_bias

    # ----- diagnostics (JSON‑compatible) -----
    # training predictions & accuracy
    train_logits = X @ head_weight.T + head_bias   # [2000,10]
    train_pred   = np.argmax(train_logits, axis=1) # [2000]
    train_acc    = (train_pred == y).mean().item()

    diag = {
        "train_accuracy": float(train_acc),
        "head_weight_norm": float(np.linalg.norm(head_weight)),
        "head_bias": head_bias.tolist()
    }

    return {"weights": w, "diagnostics": diag}
