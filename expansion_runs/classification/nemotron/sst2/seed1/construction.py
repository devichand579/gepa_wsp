def edit(w, f):
    """
    One‑shot feedback‑assisted construction.
    Adjusts the classification head (head.weight, head.bias) by
    ordinary least‑squares / ridge regression on the full feedback
    feature / label pool.  No autograd, no loops, no iterative optimiser.
    """
    import numpy as np

    # ----- data from feedback dict -----
    X = f["features"]                 # shape (4000, 32), float32
    y = f["y"]                        # shape (4000,), int64 {0,1}

    # ----- one‑hot encode labels -----
    Y = np.zeros((len(y), 2), dtype=np.float32)
    Y[np.arange(len(y)), y] = 1.0     # column for class 1 gets 1, class 0 stays 0

    # ----- ridge‑regularised normal equation -----
    lam = 1e-3                         # tiny ridge to keep matrix well‑conditioned
    XtX = X.T @ X + lam * np.eye(32, dtype=np.float32)   # 32×32
    XtY = X.T @ Y                      # 32×2
    # solve for head weight matrix W_head (2×32)
    W_head = np.linalg.solve(XtX, XtY).T   # transpose to (2,32)

    # ----- bias -----
    pred_logits = X @ W_head.T          # (4000,2)
    b_head = np.mean(Y - pred_logits, axis=0).astype(np.float32)  # (2,)

    # ----- clip to allowed range & ensure float32 -----
    W_head = np.clip(W_head, -1000, 1000).astype(np.float32)
    b_head = np.clip(b_head, -1000, 1000).astype(np.float32)

    # ----- write back into the weight dict, preserving shapes -----
    w["head.weight"] = W_head.reshape(2, 32)
    w["head.bias"]   = b_head.reshape(2)

    # ----- diagnostics (training metrics on the feedback pool) -----
    # logits after edit
    logits_edit = pred_logits + b_head   # (4000,2)
    # softmax probabilities
    exp_logits = np.exp(logits_edit - np.max(logits_edit, axis=1, keepdims=True))
    probs_edit = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    pred_edit = np.argmax(probs_edit, axis=1)
    acc = (pred_edit == y).mean()
    # binary cross‑entropy (mean over examples)
    ce = -np.mean(
        np.log(probs_edit[np.arange(len(y)), y] + 1e-12)
    )

    diagnostics = {
        "train_accuracy": float(acc),
        "train_cross_entropy": float(ce),
        "head_weight_mean": float(W_head.mean()),
        "head_weight_std": float(W_head.std()),
        "head_bias_mean": float(b_head.mean()),
        "head_bias_std": float(b_head.std()),
    }

    # return modified weights together with a tiny statistics dict
    return {"weights": w, "diagnostics": diagnostics}
