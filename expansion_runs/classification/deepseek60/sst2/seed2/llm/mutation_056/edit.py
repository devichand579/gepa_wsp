import numpy as np

def edit(w, f):
    # Read pooled features and labels for the entire feedback pool
    X = np.asarray(f["features"], dtype=np.float64)   # [4000, 32]
    y = np.asarray(f["y"], dtype=np.int64)            # [4000]
    N, D = X.shape

    # Standardize features (helps conditioning; absorbable into head)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xs = (X - mu) / sd

    # Build target: +1 for class 1, -1 for class 0
    t = np.where(y == 1, 1.0, -1.0)

    # 5-fold CV over ridge lambda
    rng = np.random.RandomState(0)
    idx = rng.permutation(N)
    folds = np.array_split(idx, 5)

    lambdas = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
    best_lambda = None
    best_acc = -1.0
    cv_scores = {}

    for lam in lambdas:
        accs = []
        for k in range(5):
            val = folds[k]
            tr = np.concatenate([folds[j] for j in range(5) if j != k])
            Xtr, ttr = Xs[tr], t[tr]
            Xval, tval = Xs[val], t[val]

            # Ridge normal equations: (X^T X + lam I) beta = X^T t
            A = Xtr.T @ Xtr + lam * np.eye(D)
            b = Xtr.T @ ttr
            beta = np.linalg.solve(A, b)
            intercept = -beta @ Xtr.mean(axis=0)  # center on training mean
            pred = Xval @ beta + intercept
            acc = np.mean((pred > 0).astype(np.int64) == (tval > 0).astype(np.int64))
            accs.append(acc)
        cv_scores[lam] = float(np.mean(accs))
        if cv_scores[lam] > best_acc:
            best_acc = cv_scores[lam]
            best_lambda = lam

    # Final fit on all data with selected lambda
    A = Xs.T @ Xs + best_lambda * np.eye(D)
    b = Xs.T @ t
    beta = np.linalg.solve(A, b)
    intercept = -beta @ Xs.mean(axis=0)

    # Convert back to original feature space:
    #   beta_s @ ((x - mu)/sd) + intercept
    # = (beta_s / sd) @ x + (intercept - beta_s @ (mu/sd))
    beta_orig = beta / sd[0]
    intercept_orig = intercept - float(beta @ (mu[0] / sd[0]))

    # Head: row1 - row0 = beta_orig, row0 + row1 = 0 (symmetric)
    # So row0 = -beta_orig/2, row1 = +beta_orig/2
    head_w = np.zeros((2, D), dtype=np.float32)
    head_w[0] = (-0.5 * beta_orig).astype(np.float32)
    head_w[1] = ( 0.5 * beta_orig).astype(np.float32)

    # Bias: logit1 - logit0 = beta_orig @ x + intercept_orig
    # head.bias[1] - head.bias[0] = intercept_orig
    head_b = np.zeros(2, dtype=np.float32)
    head_b[0] = np.float32(-0.5 * intercept_orig)
    head_b[1] = np.float32( 0.5 * intercept_orig)

    # Write back
    w["head.weight"] = head_w
    w["head.bias"]   = head_b

    # Diagnostics
    diag = {
        "selected_lambda": float(best_lambda),
        "cv_accuracy_selected": float(best_acc),
        "cv_scores": {str(k): v for k, v in cv_scores.items()},
        "in_sample_accuracy": float(np.mean((Xs @ beta + intercept > 0).astype(np.int64) == (t > 0).astype(np.int64))),
        "beta_norm": float(np.linalg.norm(beta_orig)),
        "intercept": float(intercept_orig),
        "feature_mean_absmax": float(np.abs(mu).max()),
        "feature_std_min": float(sd.min()),
        "feature_std_max": float(sd.max()),
    }
    return {"weights": w, "diagnostics": diag}
