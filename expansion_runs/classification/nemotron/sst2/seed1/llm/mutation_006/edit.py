def edit(w, f):
    """
    Closed‑form ridge regression on the final hidden features to fit the head
    weight matrix and bias using the entire permitted feedback pool.
    No autograd, back‑propagation, or iterative optimisers are used.
    """
    import numpy as np

    X = f["features"]                 # shape (4000, 32)
    y = f["y"]                        # shape (4000,), int labels 0/1
    Y_onehot = np.eye(2)[y]           # (4000, 2)

    lam = 1e-3                        # ridge regularisation strength
    I32 = np.eye(32, dtype=np.float32)

    # Solve (XᵀX + λI) w = XᵀY  for the weight matrix (32,2)
    XtX = X.T @ X + lam * I32
    XtY = X.T @ Y_onehot              # (32, 2)
    W_temp = np.linalg.solve(XtX, XtY)   # (32, 2)  – numerically stable vs inv

    # Transpose to match head.weight shape (2, 32)
    w["head.weight"] = W_temp.T.astype(np.float32)

    # Bias: mean of (target - predicted logits) over the pool
    pred_logits = X @ W_temp            # (4000, 2)
    b = np.mean(Y_onehot - pred_logits, axis=0).astype(np.float32)  # (2,)
    w["head.bias"] = b

    # Minimal diagnostics
    diag = {
        "head_weight_mean": float(np.mean(w["head.weight"])),
        "head_weight_std": float(np.std(w["head.weight"])),
        "head_bias": w["head.bias"].tolist(),
        "feedback_cross_entropy": float(f["full_feedback_metrics"]["cross_entropy"]),
    }

    return {"weights": w, "diagnostics": diag}
