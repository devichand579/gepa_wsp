def edit(w, f):
    import numpy as np

    # ------- feedback data -------
    X = f["features"].astype(np.float32)          # shape (4000, 32)
    y = f["y"].astype(np.int64)                   # shape (4000,)

    N, D = X.shape

    # ------- one‑hot encode labels -------
    Y = np.zeros((N, 2), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    # ------- ridge regularisation -------
    lam = 1e-3

    # ------- augment input with a bias term -------
    X_aug = np.hstack([X, np.ones((N, 1), dtype=np.float32)])   # (N, 33)

    # ------- closed‑form normal equations -------
    A = X_aug.T @ X_aug + lam * np.eye(D + 1, dtype=np.float32)   # (33, 33)
    B = X_aug.T @ Y                                          # (33, 2)
    W_aug = np.linalg.solve(A, B)                            # (33, 2)

    # ------- split into weights and bias -------
    W_features = W_aug[:D, :]          # (32, 2)
    b_head   = W_aug[D, :]             # (2,)

    # ------- assign to the head parameters -------
    # original head.weight shape is (2, 32); we need its transpose
    w["head.weight"] = W_features.astype(np.float32).T   # (2, 32)
    w["head.bias"]   = b_head.astype(np.float32)         # (2,)

    # ------- diagnostics -------
    logits = X @ W_features.T + b_head[np.newaxis, :]      # (N, 2)
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs    = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    pred     = np.argmax(probs, axis=1)
    acc      = float(np.mean(pred == y))
    ce       = float(-np.mean(np.log(probs[np.arange(N), y])))

    diagnostics = {
        "accuracy": acc,
        "cross_entropy": ce,
        "lambda": float(lam),
        "features_mean": float(X.mean()),
        "features_std": float(X.std()),
    }

    return {"weights": w, "diagnostics": diagnostics}
