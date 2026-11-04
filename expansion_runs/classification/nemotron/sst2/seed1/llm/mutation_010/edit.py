def edit(w, f):
    """
    Closed‑form ridge‑regression update of the network head using the full
    feedback pool.  Only the head.weight and head.bias are modified; all
    other parameters are left unchanged.
    """
    import numpy as np

    # ----- data ---------------------------------------------------------
    X = f["features"]                     # shape (N, 32), float32
    y = f["y"]                            # shape (N,), int64  {0,1}
    N = X.shape[0]

    # one‑hot labels
    Y_onehot = np.zeros((N, 2), dtype=np.float32)
    Y_onehot[np.arange(N), y] = 1.0

    # ----- ridge regression (closed‑form) -----------------------------
    lam = 1e-3                            # small ridge regularisation
    A = X.T @ X + lam * np.eye(32, dtype=np.float32)   # (32,32)
    B = X.T @ Y_onehot                    # (32,2)
    W_raw = np.linalg.solve(A, B)         # (32,2)  ->  head.weight should be (2,32)

    head_weight_new = W_raw.T.astype(np.float32)   # (2,32)

    # bias: mean of (Y - X @ W) over the two classes
    resid = Y_onehot - X @ W_raw          # (N,2)
    bias_new = np.mean(resid, axis=0).astype(np.float32)  # (2,)

    # ----- apply updates ------------------------------------------------
    w = {k: v.copy() for k, v in w.items()}   # shallow copy of all params
    w["head.weight"] = head_weight_new
    w["head.bias"]   = bias_new

    # ----- diagnostics -------------------------------------------------
    # new logits on the feedback pool
    logits_new = X @ head_weight_new.T + bias_new   # (N,2)
    # numerically stable softmax
    max_logit = logits_new.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits_new - max_logit)
    probs_new = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    # cross‑entropy and accuracy on the feedback pool
    ce = -np.mean(np.log(probs_new[np.arange(N), y] + 1e-12))
    acc  = np.mean(np.argmax(probs_new, axis=1) == y)

    diag = {
        "head_weight_mean": float(np.mean(head_weight_new)),
        "head_weight_std":  float(np.std(head_weight_new)),
        "head_bias_mean":   float(np.mean(head_bias_new)),
        "cross_entropy":    float(ce),
        "accuracy":         float(acc),
        "ridge_lambda":     float(lam),
    }

    # return either the updated weights dict or a wrapper with diagnostics
    return {"weights": w, "diagnostics": diag}
