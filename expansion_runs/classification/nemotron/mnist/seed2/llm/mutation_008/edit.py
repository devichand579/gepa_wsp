import numpy as np

def edit(w, f):
    """
    Closed‑form ridge regression of fc2 on the full feedback pool.
    Returns updated weight dict (fc2.weight, fc2.bias) and diagnostics.
    """
    # ----- data from feedback arrays -----
    X = f["features"]                     # shape (2000, 128), float32
    y = f["y"]                            # shape (2000,), int64

    # one‑hot encode labels -> (2000, 10)
    Y = np.eye(10, dtype=np.float32)[y]   # rows are one‑hot vectors

    # ----- ridge hyper‑parameter -----
    lam = 0.1                             # small L2 penalty for stability

    # ----- compute ridge‑regressed fc2.weight (out=10, in=128) -----
    # Solve (X^T X + λ I) W^T = X^T Y   =>   W^T = (X^T X + λ I)^{-1} X^T Y
    Gram = X.T @ X + lam * np.eye(128, dtype=np.float32)   # (128,128)
    # Compute X^T Y  (128,10)
    XTY = X.T @ Y                                      # (128,10)
    # Solve linear system for each output column
    # np.linalg.solve returns (128,10) matrix
    Wt = np.linalg.solve(Gram, XTY)                    # (128,10)
    fc2_weight = Wt.T.astype(np.float32)               # (10,128)

    # ----- compute fc2.bias -----
    # After fixing weight, bias = mean(Y - X @ weight.T, axis=0)
    logits = X @ fc2_weight.T + np.mean(Y - X @ fc2_weight.T, axis=0, keepdims=True)
    # Simpler: bias = mean of (Y - X @ fc2_weight.T)
    fc2_bias = (Y - X @ fc2_weight.T).mean(axis=0).astype(np.float32)  # (10,)

    # ----- update w -----
    w["fc2.weight"] = fc2_weight
    w["fc2.bias"]   = fc2_bias

    # ----- diagnostics -----
    # compute logits, probs (softmax) and metrics on the feedback pool
    logits_pool = X @ fc2_weight.T + fc2_bias   # (2000,10)
    # numerically stable softmax
    max_logits = logits_pool.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits_pool - max_logits)
    probs_pool = exp_logits / exp_logits.sum(axis=1, keepdims=True)   # (2000,10)

    acc = np.mean(np.argmax(probs_pool, axis=1) == y)
    # cross‑entropy (average over samples)
    # clip probs to avoid log(0)
    eps = 1e-12
    ce = -np.mean(np.log(probs_pool[np.arange(2000), y] + eps))

    diag = {
        "fc2_weight_mean": float(fc2_weight.mean()),
        "fc2_weight_std":  float(fc2_weight.std()),
        "fc2_bias_mean":   float(fc2_bias.mean()),
        "fc2_bias_std":    float(fc2_bias.std()),
        "feedback_accuracy": float(acc),
        "feedback_cross_entropy": float(ce),
    }

    # return updated weights together with diagnostics
    return {"weights": w, "diagnostics": diag}
