import numpy as np

def edit(w, f):
    """
    Hypothesis: The current network is a degenerate length-normalised unigram model.
    The final hidden features (f["features"]) are effectively 3-dimensional because
    the input to the final LayerNorm has only dims 0 and 1 non-zero.  Dims 2..31
    are all identical.  The current head only uses dims 0 and 1.  By fitting a
    regularised linear classifier (ridge regression) on all 32 features we can
    exploit the third distinct feature (the mean-subtraction term) and improve
    held-out accuracy.
    """
    X = f["features"].astype(np.float64)          # (4000, 32)
    y = f["y"].astype(np.float64)                 # (4000,)

    # Standardise features (important because dims 0,1 have much larger variance)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Xs = (X - mu) / sd

    # 5-fold cross-validation to choose ridge lambda
    n = X.shape[0]
    idx = np.arange(n)
    rng = np.random.RandomState(42)
    rng.shuffle(idx)
    folds = np.array_split(idx, 5)

    lambdas = np.logspace(-4, 3, 20)
    best_lam = 1.0
    best_cv_acc = -1.0

    for lam in lambdas:
        accs = []
        for k in range(5):
            val_idx = folds[k]
            tr_idx = np.concatenate([folds[j] for j in range(5) if j != k])
            Xtr, ytr = Xs[tr_idx], y[tr_idx]
            Xval, yval = Xs[val_idx], y[val_idx]

            # centre
            Xtr_m = Xtr.mean(axis=0)
            ytr_m = ytr.mean()
            Xtr_c = Xtr - Xtr_m
            ytr_c = ytr - ytr_m

            # ridge closed form
            A = Xtr_c.T @ Xtr_c + lam * np.eye(Xtr.shape[1])
            b = Xtr_c.T @ ytr_c
            w_std = np.linalg.solve(A, b)
            b_std = ytr_m - Xtr_m @ w_std

            # predict
            pred = Xval @ w_std + b_std
            pred_label = (pred > 0.5).astype(int)
            acc = (pred_label == yval).mean()
            accs.append(acc)
        cv_acc = np.mean(accs)
        if cv_acc > best_cv_acc:
            best_cv_acc = cv_acc
            best_lam = lam

    # Fit on full data with best lambda
    X_m = Xs.mean(axis=0)
    y_m = y.mean()
    X_c = Xs - X_m
    y_c = y - y_m
    A = X_c.T @ X_c + best_lam * np.eye(X.shape[1])
    b = X_c.T @ y_c
    w_std = np.linalg.solve(A, b)
    b_std = y_m - X_m @ w_std

    # Convert back to raw feature space
    w_raw = w_std / sd
    b_raw = b_std - mu @ w_raw

    # We want logit1 - logit0 = w_raw · features + (b_raw - 0.5)
    # because ridge predicts y = X @ w_raw + b_raw, threshold at 0.5.
    w_diff = w_raw
    b_diff = b_raw - 0.5

    # Set head weights: symmetric split
    head_w = np.zeros_like(w["head.weight"])
    head_b = np.zeros_like(w["head.bias"])
    head_w[0] = -0.5 * w_diff
    head_w[1] =  0.5 * w_diff
    head_b[0] = -0.5 * b_diff
    head_b[1] =  0.5 * b_diff

    w["head.weight"] = head_w.astype(np.float32)
    w["head.bias"]   = head_b.astype(np.float32)

    # Diagnostics
    diag = {
        "best_lambda": float(best_lam),
        "cv_accuracy": float(best_cv_acc),
        "w_norm": float(np.linalg.norm(w_diff)),
        "b_diff": float(b_diff),
        "head_weight_nonzero": int(np.count_nonzero(head_w)),
    }
    return {"weights": w, "diagnostics": diag}
