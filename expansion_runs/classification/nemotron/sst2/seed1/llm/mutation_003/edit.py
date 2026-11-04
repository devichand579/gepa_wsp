def edit(w, f):
    """
    Closed‑form ridge‑regression fit of the network head on the full feedback pool.
    Only head.weight and head.bias are altered; every other parameter is copied
    verbatim so downstream cached activations remain valid.
    """
    import numpy as np

    # ----- feedback pool -------------------------------------------------
    F = f["features"].astype(np.float32)          # [4000, 32]  final hidden
    y = f["y"].astype(np.int64)                  # [4000]       labels 0/1
    N, D = F.shape

    # one‑hot targets
    Y = np.eye(2, dtype=np.float32)[y]           # [4000, 2]

    # ridge regularisation (tiny value to guarantee invertibility)
    lam = 1e-3

    # normal‑equation matrices
    FtF = F.T @ F + lam * np.eye(D, dtype=np.float32)   # [32,32]
    FtY = F.T @ Y                                      # [32,2]

    # direct solve – no iterative optimiser
    Wcoef = np.linalg.solve(FtF, FtY)                 # [32,2]  weight per class

    # new head parameters (shapes unchanged)
    new_head_weight = Wcoef.T.astype(np.float32)      # [2,32]
    new_head_bias   = np.mean(Y - F @ Wcoef.T, axis=0).astype(np.float32)  # [2]

    # copy current weights and replace only the head
    w_new = {k: v.copy() for k, v in w.items()}
    w_new["head.weight"] = new_head_weight
    w_new["head.bias"]   = new_head_bias

    # ----- diagnostics on the feedback pool ----------------------------
    logits_new = F @ new_head_weight.T + new_head_bias[np.newaxis, :]   # [4000,2]
    exp_logits = np.exp(logits_new - np.max(logits_new, axis=1, keepdims=True))
    probs_new  = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    pred       = np.argmax(probs_new, axis=1)
    accuracy   = float(np.mean(pred == y))
    ce         = float(-np.mean(np.log(probs_new[np.arange(N), y] + 1e-12)))

    diagnostics = {
        "accuracy":              accuracy,
        "cross_entropy":         ce,
        "head_weight_mean":      float(new_head_weight.mean()),
        "head_weight_std":       float(new_head_weight.std()),
        "head_bias_mean":        float(new_head_bias.mean()),
    }

    return {"weights": w_new, "diagnostics": diagnostics}
