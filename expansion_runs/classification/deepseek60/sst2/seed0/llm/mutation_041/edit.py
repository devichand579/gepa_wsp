import numpy as np

def edit(w, f):
    X = np.asarray(f["features"], dtype=np.float64)
    y = np.asarray(f["y"], dtype=np.float64)
    y = 2.0 * y - 1.0  # {-1, +1}

    # centre for unregularised intercept
    X_mean = X.mean(axis=0)
    y_mean = y.mean()
    Xc = X - X_mean
    yc = y - y_mean

    # 5-fold CV over a small lambda grid
    n = X.shape[0]
    idx = np.arange(n)
    rng = np.random.RandomState(0)
    rng.shuffle(idx)
    folds = np.array_split(idx, 5)

    lambdas = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    best_lam = None
    best_acc = -1.0

    for lam in lambdas:
        accs = []
        for k in range(5):
            val = folds[k]
            tr = np.concatenate([folds[j] for j in range(5) if j != k])
            Xtr = Xc[tr]
            ytr = yc[tr]
            Xva = Xc[val]
            yva = y[val]

            # solve (Xtr^T Xtr + lam I) w = Xtr^T ytr
            A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
            b = Xtr.T @ ytr
            w_cv = np.linalg.solve(A, b)
            # intercept from training means
            b_cv = y_mean - X_mean @ w_cv
            pred = np.sign(Xva @ w_cv + b_cv)
            accs.append(np.mean(pred == yva))
        acc = float(np.mean(accs))
        if acc > best_acc:
            best_acc = acc
            best_lam = lam

    # fit on full data with best lambda
    lam = best_lam if best_lam is not None else 1.0
    A = Xc.T @ Xc + lam * np.eye(Xc.shape[1])
    b = Xc.T @ yc
    w_full = np.linalg.solve(A, b)
    b_full = y_mean - X_mean @ w_full

    # write head: logit1 = w_full·x + b_full, logit0 = 0
    w["head.weight"] = np.zeros_like(w["head.weight"], dtype=np.float32)
    w["head.weight"][1, :] = w_full.astype(np.float32)
    w["head.bias"] = np.zeros_like(w["head.bias"], dtype=np.float32)
    w["head.bias"][1] = np.float32(b_full)

    # diagnostics
    pred_full = np.sign(X @ w_full + b_full)
    train_acc = float(np.mean(pred_full == y))
    diag = {
        "best_lambda": float(lam),
        "cv_acc": float(best_acc),
        "train_acc": train_acc,
        "w_norm": float(np.linalg.norm(w_full)),
        "bias": float(b_full),
    }
    return {"weights": w, "diagnostics": diag}
