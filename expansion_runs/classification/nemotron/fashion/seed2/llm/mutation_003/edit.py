def edit(w, f):
    """
    Edit the network head via a closed‑form ridge regression from
    the provided feedback features to the training labels.
    No autograd, gradients, or iterative optimisers are used.
    """
    # ------------------------------------------------------------------
    # 1. Pull the permitted feedback arrays (read‑only)
    # ------------------------------------------------------------------
    X = f["features"].astype(np.float32)          # (2000, 1568)
    y = f["y"].astype(np.int64)                   # (2000,)

    # ------------------------------------------------------------------
    # 2. Convert labels to one‑hot matrix Y (2000, 10)
    # ------------------------------------------------------------------
    Y = np.eye(10, dtype=np.float32)[y]           # (2000, 10)

    # ------------------------------------------------------------------
    # 3. Ridge‑regularised normal equation: (FᵀF + λI)⁻¹ Fᵀ Y
    # ------------------------------------------------------------------
    lam = np.float32(0.001)                       # small ridge constant
    FtF = X.T @ X                                 # (1568, 1568)
    I   = np.eye(FtF.shape[0], dtype=np.float32)
    A   = FtF + lam * I                           # (1568, 1568), positive‑definite
    B   = X.T @ Y                                 # (1568, 10)

    # Solve A * W_tilde = B  →  W_tilde shape (1568, 10)
    W_tilde = np.linalg.solve(A, B).astype(np.float32)

    # Transpose to PyTorch convention: (10, 1568)
    head_weight = W_tilde.T

    # ------------------------------------------------------------------
    # 4. Compute bias as mean residual (Y - X @ head_weightᵀ)
    # ------------------------------------------------------------------
    residuals = Y - X @ head_weight.T          # (2000, 10)
    head_bias = residuals.mean(axis=0).astype(np.float32)   # (10,)

    # ------------------------------------------------------------------
    # 5. Update the weight dictionary in‑place
    # ------------------------------------------------------------------
    w["head.weight"] = head_weight
    w["head.bias"]   = head_bias

    # ------------------------------------------------------------------
    # 6. Diagnostics (train‑time metrics on the full feedback pool)
    # ------------------------------------------------------------------
    logits = X @ head_weight.T + head_bias[np.newaxis, :]   # (2000, 10)
    pred = np.argmax(logits, axis=1)
    train_acc = float(np.mean(pred == y))

    # Numerically stable cross‑entropy
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    ce = -float(np.mean(np.log(probs[np.arange(2000), y])))

    diag = {
        "train_accuracy": train_acc,
        "train_cross_entropy": ce,
        "head_weight_norm": float(np.linalg.norm(head_weight)),
        "head_bias": head_bias.tolist()
    }

    # Return the full weight dict together with a tiny diagnostics block.
    return {"weights": w, "diagnostics": diag}
