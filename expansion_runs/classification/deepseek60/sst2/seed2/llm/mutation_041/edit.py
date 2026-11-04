def edit(w, f):
    import numpy as np
    X = np.asarray(f["features"], dtype=np.float64)      # [N,32] pooled LN features
    y = np.asarray(f["y"], dtype=np.int64)               # [N]
    N, D = X.shape

    # one-hot targets
    Y = np.zeros((N, 2), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # augment with bias column
    Xa = np.concatenate([X, np.ones((N, 1), dtype=np.float64)], axis=1)  # [N,33]

    # ridge normal equations (closed form)
    lam = 1e-3
    A = Xa.T @ Xa + lam * np.eye(D + 1)
    B = Xa.T @ Y
    W = np.linalg.solve(A, B)                            # [33,2]

    head_w = W[:D, :].T.astype(np.float32)               # [2,32]
    head_b = W[D, :].astype(np.float32)                  # [2]

    # sanity: clip to allowed range
    head_w = np.clip(head_w, -1000.0, 1000.0)
    head_b = np.clip(head_b, -1000.0, 1000.0)

    w["head.weight"] = head_w
    w["head.bias"] = head_b

    # diagnostics on the new head (training-pool fit quality)
    logits = Xa @ W                                      # [N,2]
    pred = np.argmax(logits, axis=1)
    acc = float(np.mean(pred == y))
    # per-class accuracy
    accs = {}
    for c in (0, 1):
        m = (y == c)
        accs[str(c)] = float(np.mean(pred[m] == y[m])) if m.any() else 0.0

    diag = {
        "fit_train_acc": acc,
        "fit_class_acc": accs,
        "head_w_norm": float(np.linalg.norm(head_w)),
        "head_b": [float(head_b[0]), float(head_b[1])],
        "lambda": lam,
        "n_samples": int(N),
    }
    return {"weights": w, "diagnostics": diag}
