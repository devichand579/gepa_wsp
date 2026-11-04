def edit(w, f):
    """
    Closed‑form ridge‑regression update of the final head layer using the
    entire feedback pool (features + labels). No autograd, no iterative
    optimiser, single manual normal‑equation step.
    """
    # ----- data -----
    X = f["features"]                     # (4000, 32) float32
    y = f["y"].astype(np.int32)          # (4000,) int64 labels 0/1

    # one‑hot encode labels → (4000, 2)
    Y = np.zeros((len(y), 2), dtype=np.float32)
    Y[np.arange(len(y)), y] = 1.0

    # augment features with a bias column → (4000, 33)
    Xb = np.hstack([X, np.ones((len(y), 1), dtype=np.float32)])

    # ridge regularisation (small constant)
    lam = 0.1
    I33 = np.eye(33, dtype=np.float32)

    # normal equation: Wb = (Xbᵀ Xb + λ I)⁻¹ Xbᵀ Y
    # use solve for numerical stability
    Wb = np.linalg.solve(Xb.T @ Xb + lam * I33, Xb.T @ Y)   # (33, 2)

    # split into weight matrix (2,32) and bias vector (2,)
    head_weight = Wb[:32, :].T.astype(np.float32)          # (2, 32)
    head_bias   = Wb[32, :].astype(np.float32)            # (2,)

    # assign the new parameters back into the weight dict
    w["head.weight"] = head_weight
    w["head.bias"]   = head_bias

    # ----- diagnostics -----
    logits = X @ head_weight.T + head_bias                # (4000, 2)
    pred   = np.argmax(logits, axis=1)
    acc    = float(np.mean(pred == y))

    # numerically stable cross‑entropy
    logits_max = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    sum_exp    = exp_logits.sum(axis=1, keepdims=True)
    log_sum    = np.log(sum_exp + 1e-12)
    # pick the true‑class log‑prob
    true_logp  = logits[np.arange(len(y)), y] - logits_max[:, 0] - log_sum[:, 0]
    ce         = float(-np.mean(true_logp))

    diag = {
        "ridge_lambda": float(lam),
        "feedback_accuracy": acc,
        "feedback_cross_entropy": ce,
        "head_weight_norm": float(np.linalg.norm(head_weight)),
        "head_bias": head_bias.tolist()
    }

    return {"weights": w, "diagnostics": diag}
