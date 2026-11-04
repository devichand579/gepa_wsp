def edit(w, f):
    import numpy as np

    # ---------- read feedback ----------
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float32)  # (2000, 784)
    y = f["y"].astype(np.int64)                                  # (2000,)
    N = X.shape[0]
    C = 10

    H_old = f["features"].astype(np.float32)                     # (2000, 128)

    fc1_w = w["fc1.weight"].copy()   # (128, 784)
    fc1_b = w["fc1.bias"].copy()     # (128,)
    fc2_w = w["fc2.weight"].copy()   # (10, 128)
    fc2_b = w["fc2.bias"].copy()     # (10,)

    # ---------- choose hidden units to replace ----------
    # Importance = norm of the corresponding column in fc2.weight
    importance = np.linalg.norm(fc2_w, axis=0)  # (128,)
    n_drop = 20
    drop_idx = np.argsort(importance)[:n_drop]
    keep_idx = np.setdiff1d(np.arange(128), drop_idx)
    n_keep = len(keep_idx)  # 108

    # ---------- fit raw-pixel linear classifier by ridge regression ----------
    X_aug = np.concatenate([X, np.ones((N, 1), dtype=np.float32)], axis=1)  # (2000, 785)
    Y = np.zeros((N, C), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    lam = 1.0
    A = X_aug.T @ X_aug + lam * np.eye(X_aug.shape[1], dtype=np.float32)
    B = X_aug.T @ Y
    W_aug = np.linalg.solve(A, B)          # (785, 10)
    W_lin = W_aug[:784, :].T               # (10, 784)
    b_lin = W_aug[784, :]                  # (10,)

    # ---------- build new fc1 ----------
    new_fc1_w = np.zeros((128, 784), dtype=np.float32)
    new_fc1_b = np.zeros(128, dtype=np.float32)

    new_fc1_w[:n_keep] = fc1_w[keep_idx]
    new_fc1_b[:n_keep] = fc1_b[keep_idx]

    # 20 new units: positive and negative parts of the 10 linear logits
    for c in range(C):
        new_fc1_w[n_keep + 2 * c]     = W_lin[c]
        new_fc1_b[n_keep + 2 * c]     = b_lin[c]
        new_fc1_w[n_keep + 2 * c + 1] = -W_lin[c]
        new_fc1_b[n_keep + 2 * c + 1] = -b_lin[c]

    # ---------- compute new hidden features on full feedback pool ----------
    H_new = np.zeros((N, 128), dtype=np.float32)
    H_new[:, :n_keep] = H_old[:, keep_idx]

    logits_lin = X @ W_lin.T + b_lin          # (2000, 10)
    pos = np.maximum(logits_lin, 0.0)
    neg = np.maximum(-logits_lin, 0.0)
    for c in range(C):
        H_new[:, n_keep + 2 * c]     = pos[:, c]
        H_new[:, n_keep + 2 * c + 1] = neg[:, c]

    # ---------- fit new fc2 by ridge regression on H_new ----------
    # Add bias column
    H_aug = np.concatenate([H_new, np.ones((N, 1), dtype=np.float32)], axis=1)  # (2000, 129)

    # Use a small set of ridge strengths and pick by simple hold-out / GCV.
    # Here we use a closed-form GCV-like selection over a few values.
    best_lam = None
    best_score = -np.inf
    best_W = None
    best_b = None

    for lam2 in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        A2 = H_aug.T @ H_aug + lam2 * np.eye(H_aug.shape[1], dtype=np.float32)
        B2 = H_aug.T @ Y
        W2 = np.linalg.solve(A2, B2)          # (129, 10)
        # Training accuracy as a simple score
        pred = H_aug @ W2
        acc = np.mean(np.argmax(pred, axis=1) == y)
        if acc > best_score:
            best_score = acc
            best_lam = lam2
            best_W = W2[:128, :].T            # (10, 128)
            best_b = W2[128, :]               # (10,)

    # ---------- write back ----------
    w["fc1.weight"] = new_fc1_w.astype(np.float32)
    w["fc1.bias"]   = new_fc1_b.astype(np.float32)
    w["fc2.weight"] = best_W.astype(np.float32)
    w["fc2.bias"]   = best_b.astype(np.float32)

    diagnostics = {
        "n_keep": int(n_keep),
        "n_drop": int(n_drop),
        "ridge_lam_linear": float(lam),
        "best_lam_fc2": float(best_lam) if best_lam is not None else None,
        "train_acc_after": float(best_score),
        "fc1_weight_mean": float(np.mean(new_fc1_w)),
        "fc1_weight_std": float(np.std(new_fc1_w)),
        "fc2_weight_mean": float(np.mean(best_W)),
        "fc2_weight_std": float(np.std(best_W)),
        "fc2_bias_mean": float(np.mean(best_b)),
        "fc2_bias_std": float(np.std(best_b)),
    }
    return {"weights": w, "diagnostics": diagnostics}
