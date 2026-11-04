def edit(w, f):
    """
    Closed‑form ridge‑regression fit of the network head on the entire feedback pool.
    Only the head.weight (shape [2,32]) and head.bias (shape [2]) are modified.
    Returns the updated weight dict and a small diagnostics dict.
    """
    import numpy as np

    # ----- data from feedback -----
    features = f["features"]            # shape (4000, 32), float32
    y        = f["y"]                   # shape (4000,), int64 0/1

    N, D = features.shape               # N=4000, D=32

    # one‑hot encode labels → (N,2)
    Y = np.eye(2, dtype=np.float32)[y]   # 0 → [1,0], 1 → [0,1]

    # ridge regularisation (small jitter to guarantee invertibility)
    lam = 1e-3

    # Normal equations: (XᵀX + λI) W = Xᵀ Y   →   W = (XᵀX + λI)^{-1} Xᵀ Y
    # XᵀX is (D,D), XᵀY is (D,2)
    A = features.T @ features + lam * np.eye(D, dtype=np.float32)
    B = features.T @ Y                     # (D,2)

    # Solve for un‑transposed weight matrix (D,2)
    W_raw = np.linalg.solve(A, B)          # (D,2)

    # head.weight expects shape (2,32) → transpose
    new_head_weight = W_raw.astype(np.float32).T   # (2,32)

    # bias: per‑class mean residual after fitting
    # residuals = Y - X @ W_rawᵀ   →   (N,2)
    residuals = Y - (features @ W_raw.T)
    new_head_bias = np.mean(residuals, axis=0).astype(np.float32)   # (2,)

    # ----- apply edits -----
    w["head.weight"] = new_head_weight
    w["head.bias"]   = new_head_bias

    # ----- diagnostics (computed on the same cached features) -----
    logits = features @ new_head_weight.T + new_head_bias   # (N,2)
    preds  = np.argmax(logits, axis=1)
    accuracy = float(np.mean(preds == y))

    # softmax cross‑entropy
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs      = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    ce         = -float(np.mean(np.log(probs[np.arange(N), y])))

    diagnostics = {
        "accuracy": accuracy,
        "cross_entropy": ce,
        "head_weight_mean": float(new_head_weight.mean()),
        "head_bias_mean": float(new_head_bias.mean()),
        "ridge_lambda": float(lam)
    }

    return {"weights": w, "diagnostics": diagnostics}
