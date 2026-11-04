def edit(w, f):
    """
    Closed‑form ridge‑regression fit of the head layer using the full feedback pool.
    Returns updated weight dictionary and diagnostics.
    """
    import numpy as np

    # ------------------------------------------------------------------
    # 1.  Gather feedback arrays
    # ------------------------------------------------------------------
    F = f["features"]                     # shape (2000, 1568), float32
    y = f["y"]                            # shape (2000,), int64
    N = F.shape[0]
    C = 10                                # number of classes
    D = 1568                              # feature dimension

    # ------------------------------------------------------------------
    # 2.  Build one‑hot target matrix Y (N x C)
    # ------------------------------------------------------------------
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # ------------------------------------------------------------------
    # 3.  Augment features with a bias term (column of ones) → N x (D+1)
    # ------------------------------------------------------------------
    F_aug = np.hstack([F.astype(np.float64),
                       np.ones((N, 1), dtype=np.float64)])  # N x (D+1)

    # ------------------------------------------------------------------
    # 4.  Ridge regularisation (small λ to avoid singularity)
    # ------------------------------------------------------------------
    lam = 1e-3
    A = F_aug.T @ F_aug + lam * np.eye(D + 1)      # (D+1) x (D+1)
    B = F_aug.T @ Y                                # (D+1) x C

    # ------------------------------------------------------------------
    # 5.  Solve the normal equations  A @ W_aug^T = B  for W_aug (C x D+1)
    # ------------------------------------------------------------------
    W_aug_T = np.linalg.solve(A, B)                # (D+1) x C
    W_aug = W_aug_T.T                              # C x (D+1)

    # ------------------------------------------------------------------
    # 6.  Split into weight matrix and bias vector
    # ------------------------------------------------------------------
    new_head_weight = W_aug[:, :D].astype(np.float32)      # (C, D) → [out,in]
    new_head_bias   = W_aug[:, D].astype(np.float32)       # (C,)

    # ------------------------------------------------------------------
    # 7.  Apply the new head parameters (leave all other modules untouched)
    # ------------------------------------------------------------------
    w["head.weight"] = new_head_weight
    w["head.bias"]   = new_head_bias

    # ------------------------------------------------------------------
    # 8.  Evaluate the fitted model on the same feedback pool (closed‑form)
    # ------------------------------------------------------------------
    # new logits = features · weight^T + bias
    new_logits = F.astype(np.float64) @ new_head_weight.T + \
                 new_head_bias[:, np.newaxis]          # N x C

    # numerically stable softmax
    exp_logits = np.exp(new_logits - np.max(new_logits, axis=1, keepdims=True))
    new_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    new_accuracy = (np.argmax(new_probs, axis=1) == y).mean()
    new_cross_ent = -np.mean(np.log(new_probs[np.arange(N), y]))

    # baseline metrics from the original pool (before edit)
    old_probs = f["probs"]                     # shape (2000,10) float32
    old_accuracy = (np.argmax(old_probs, axis=1) == y).mean()
    old_cross_ent = -np.mean(np.log(old_probs[np.arange(N), y]))

    # ------------------------------------------------------------------
    # 9.  Diagnostics (JSON‑compatible)
    # ------------------------------------------------------------------
    diag = {
        "lambda": float(lam),
        "weight_mean": float(new_head_weight.mean()),
        "bias_mean": float(new_head_bias.mean()),
        "old_accuracy": float(old_accuracy),
        "new_accuracy": float(new_accuracy),
        "accuracy_gain": float(new_accuracy - old_accuracy),
        "old_cross_entropy": float(old_cross_ent),
        "new_cross_entropy": float(new_cross_ent),
        "cross_entropy_reduction": float(old_cross_ent - new_cross_ent),
    }

    return {"weights": w, "diagnostics": diag}
