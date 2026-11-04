import numpy as np

def edit(w, f):
    # Extract feedback arrays
    X = f["features"].astype(np.float64)          # (2000, 128)
    y = f["y"].astype(np.int64)                   # (2000,)
    N, D = X.shape
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Current fc2 parameters, including bias as an extra weight
    W_cur = w["fc2.weight"].astype(np.float64)    # (10, 128)
    b_cur = w["fc2.bias"].astype(np.float64)      # (10,)
    Wb_cur = np.hstack([W_cur, b_cur[:, None]])   # (10, 129)

    # Augment features with a bias column
    Xb = np.hstack([X, np.ones((N, 1), dtype=np.float64)])  # (2000, 129)

    def accuracy(Wb, Xb_eval, y_eval):
        logits = Xb_eval @ Wb.T
        pred = np.argmax(logits, axis=1)
        return float(np.mean(pred == y_eval))

    # 5-fold cross-validation
    n_folds = 5
    rng = np.random.RandomState(42)
    indices = np.arange(N)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    # Candidate hyperparameters
    lambdas = [0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 1e5, 1e6]
    alphas = [0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]

    # We will evaluate several families of closed-form solutions.
    # Each family is a function that, given train indices, returns Wb (10,129).
    def fit_current(train_idx):
        return Wb_cur

    def fit_ridge(train_idx, lam):
        Xb_tr = Xb[train_idx]
        Y_tr = Y[train_idx]
        XtX = Xb_tr.T @ Xb_tr
        XtY = Xb_tr.T @ Y_tr
        A = XtX + lam * np.eye(D + 1)
        Wb_T = np.linalg.solve(A, XtY)
        return Wb_T.T

    def fit_bayes(train_idx, lam):
        Xb_tr = Xb[train_idx]
        Y_tr = Y[train_idx]
        XtX = Xb_tr.T @ Xb_tr
        XtY = Xb_tr.T @ Y_tr
        A = XtX + lam * np.eye(D + 1)
        Wb_T = np.linalg.solve(A, XtY + lam * Wb_cur.T)
        return Wb_T.T

    def fit_lda(train_idx, alpha):
        X_tr = X[train_idx]
        y_tr = y[train_idx]
        means = np.zeros((C, D), dtype=np.float64)
        Sw = np.zeros((D, D), dtype=np.float64)
        for c in range(C):
            Xc = X_tr[y_tr == c]
            if len(Xc) > 0:
                mean_c = Xc.mean(axis=0)
                means[c] = mean_c
                Xc_centered = Xc - mean_c
                Sw += Xc_centered.T @ Xc_centered
        global_mean = X_tr.mean(axis=0)
        Sw_reg = Sw + alpha * np.eye(D)
        W_lda_T = np.linalg.solve(Sw_reg, (means - global_mean).T)  # (128, 10)
        W_lda = W_lda_T.T  # (10, 128)
        priors = np.array([np.mean(y_tr == c) for c in range(C)])
        b_lda = np.zeros(C, dtype=np.float64)
        for c in range(C):
            b_lda[c] = -0.5 * (means[c] + global_mean) @ W_lda_T[:, c] + np.log(priors[c] + 1e-12)
        return np.hstack([W_lda, b_lda[:, None]])

    # Build list of (name, fit_fn) candidates
    candidates = [("current", fit_current)]
    for lam in lambdas:
        candidates.append((f"ridge_{lam}", lambda tr, lam=lam: fit_ridge(tr, lam)))
    for lam in lambdas:
        candidates.append((f"bayes_{lam}", lambda tr, lam=lam: fit_bayes(tr, lam)))
    for alpha in alphas:
        candidates.append((f"lda_{alpha}", lambda tr, alpha=alpha: fit_lda(tr, alpha)))

    # Evaluate each candidate by 5-fold CV
    cv_scores = {}
    for name, fit_fn in candidates:
        scores = []
        for i in range(n_folds):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            Wb = fit_fn(train_idx)
            scores.append(accuracy(Wb, Xb[test_idx], y[test_idx]))
        cv_scores[name] = float(np.mean(scores))

    # Choose the best candidate
    best_name = max(cv_scores, key=cv_scores.get)
    best_score = cv_scores[best_name]

    # Refit the chosen model on the full 2000-example pool
    if best_name == "current":
        Wb_best = Wb_cur
    elif best_name.startswith("ridge_"):
        lam = float(best_name.split("_")[1])
        Wb_best = fit_ridge(np.arange(N), lam)
    elif best_name.startswith("bayes_"):
        lam = float(best_name.split("_")[1])
        Wb_best = fit_bayes(np.arange(N), lam)
    elif best_name.startswith("lda_"):
        alpha = float(best_name.split("_")[1])
        Wb_best = fit_lda(np.arange(N), alpha)
    else:
        Wb_best = Wb_cur

    # Write back to w
    w["fc2.weight"] = Wb_best[:, :D].astype(np.float32)
    w["fc2.bias"] = Wb_best[:, D].astype(np.float32)

    # Diagnostics
    diagnostics = {
        "chosen_method": best_name,
        "cv_accuracy": best_score,
        "cv_scores_top5": sorted(cv_scores.items(), key=lambda kv: kv[1], reverse=True)[:5],
        "current_cv_accuracy": cv_scores.get("current", None),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
    }
    return {"weights": w, "diagnostics": diagnostics}
