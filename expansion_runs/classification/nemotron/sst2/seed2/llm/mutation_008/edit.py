def edit(w, f):
    import numpy as np

    # ----- feedback data -----
    X = f["features"].astype(np.float32)          # (4000, 32)
    y = f["y"].astype(np.int64)                   # (4000,)
    Y = np.eye(2, dtype=np.float32)[y]            # (4000, 2) one‑hot

    lam = 1e-3                                    # ridge penalty

    # ----- ridge‑regression on head weights (32×2) -----
    A = X.T @ X + lam * np.eye(32, dtype=np.float32)   # (32,32)
    B = X.T @ Y                                          # (32,2)
    W = np.linalg.solve(A, B)                            # (32,2)

    new_head_weight = W.T.astype(np.float32)             # (2,32)  -> matches w["head.weight"]

    # ----- bias per class -----
    resid = Y - X @ W                                    # (4000,2)
    new_head_bias = np.mean(resid, axis=0).astype(np.float32)  # (2,)

    # ----- apply edits -----
    w["head.weight"] = new_head_weight
    w["head.bias"] = new_head_bias

    # ----- diagnostics on the feedback pool (using stale features) -----
    logits_new = X @ new_head_weight.T + new_head_bias          # (4000,2)
    exp_logits = np.exp(logits_new - np.max(logits_new, axis=1, keepdims=True))
    probs_new = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    pred = np.argmax(probs_new, axis=1)
    acc = float((pred == y).mean())

    diag = {
        "ridge_lambda": float(lam),
        "feedback_accuracy": acc,
    }

    return {"weights": w, "diagnostics": diag}
