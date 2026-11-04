def edit(w, f):
    import numpy as np

    # ----- feedback pool -----
    X = f["features"]                # shape (2000, 128), float32
    y = f["y"]                       # shape (2000,), int64
    N, d = X.shape
    C = 10

    # one‑hot labels
    Y = np.eye(C, dtype=np.float32)[y]   # (N, C)

    # ridge regularisation (small lambda for stability)
    lam = 1e-3

    # ----- closed‑form ridge regression for fc2 -----
    # A = XᵀX + λI   (d × d)
    XtX = X.T @ X + lam * np.eye(d, dtype=np.float32)
    Ainv = np.linalg.inv(XtX)

    # M = YᵀX   (C × d)
    M = Y.T @ X

    # W = M @ Ainv   (C × d) → fc2.weight
    W = M @ Ainv

    # optimal bias b = mean(Y - X·Wᵀ, axis=0)   (C,)
    XWt = X @ W.T                     # (N, C)
    b = np.mean(Y - XWt, axis=0)      # (C,)

    # ----- edit the network weights -----
    w["fc2.weight"] = W.astype(np.float32)     # shape (10, 128)
    w["fc2.bias"]   = b.astype(np.float32)     # shape (10,)

    # ----- diagnostics (optional, for later reflection) -----
    # recompute logits & probabilities for the whole feedback pool
    logits_all = X @ W.T + b               # (N, C)
    logits_max = logits_all.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits_all - logits_max)
    probs_all = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    acc = (np.argmax(probs_all, axis=1) == y).mean()
    ce  = -np.log(probs_all[np.arange(N), y]).mean()

    diag = {
        "train_accuracy": float(acc),
        "train_cross_entropy": float(ce),
        "fc2_weight_norm": float(np.linalg.norm(W, "fro")),
        "fc2_bias": b.tolist()
    }

    return {"weights": w, "diagnostics": diag}
