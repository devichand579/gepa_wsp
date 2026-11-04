def edit(w, f):
    """
    Closed‑form ridge‑regression update of the classification head.
    - f["features"]:  [2000, 1568]  final hidden activations (flattened conv2 output)
    - f["y"]        :  [2000]       integer labels (0‑9)
    - w["head.weight"]: [10, 1568]  current head weights
    - w["head.bias"]  : [10]        current head bias
    No autograd, no iterative optimiser – a single normal‑equation solve.
    """
    import numpy as np

    # ----- data preparation -------------------------------------------------
    X = f["features"]                     # [2000, 1568]
    y = f["y"]                            # [2000] int64

    # one‑hot encode labels → [2000, 10]
    Y = np.eye(10, dtype=np.float32)[y]   # rows are one‑hot vectors

    # ----- ridge solution for head.weight ---------------------------------
    lam = 1e-3                            # tiny ridge coefficient
    # A = Xᵀ X + λ I   (1568 × 1568)
    A = X.T @ X + lam * np.eye(1568, dtype=np.float32)
    # B = Xᵀ Y   (1568 × 10)
    B = X.T @ Y
    # solve A * W_raw = B  →  W_raw is [1568, 10]
    W_raw = np.linalg.solve(A, B)        # numerically stable, no loop

    # map to the stored layout: head.weight is [10, 1568]
    w["head.weight"] = W_raw.T.astype(np.float32)

    # ----- ridge solution for head.bias ------------------------------------
    # bias = mean( Y - X @ W_rawᵀ , axis=0 )
    pred_logits = X @ W_raw.T            # [2000, 10]
    b = (Y - pred_logits).mean(axis=0)   # [10]
    w["head.bias"] = b.astype(np.float32)

    # ----- diagnostics (computed on the same pool) ------------------------
    # logits, probabilities, loss, accuracy
    logits = X @ w["head.weight"].T + w["head.bias"]   # [2000, 10]
    probs = np.exp(logits) / np.exp(logits).sum(axis, keepdims=True)  # softmax
    # cross‑entropy (mean over pool)
    ce = -np.log(probs[np.arange(2000), y]).mean()
    # top‑1 accuracy
    pred_labels = np.argmax(probs, axis=1)
    acc = (pred_labels == y).mean()

    diagnostics = {
        "cross_entropy": float(ce),
        "accuracy": float(acc),
        "head_weight_norm": float(np.linalg.norm(w["head.weight"])),
        "head_bias": w["head.bias"].tolist()
    }

    # ----- return -----------------------------------------------------------
    # The prompt allows returning just the weight dict or a wrapper.
    # We return the full w dict together with a tiny diagnostics JSON.
    return {"weights": w, "diagnostics": diagnostics}
