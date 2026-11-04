import numpy as np

def edit(w, f):
    # Extract arrays
    X = f["features"].astype(np.float64)          # [2000, 128]
    y = f["y"].astype(np.int64)                   # [2000]
    N, D = X.shape
    C = 10

    # Current fc2 weights and bias
    W_cur = w["fc2.weight"].astype(np.float64)    # [10, 128]
    b_cur = w["fc2.bias"].astype(np.float64)      # [10]

    # One-hot labels
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Center features and labels for ridge (optional, but helps conditioning)
    X_mean = X.mean(axis=0, keepdims=True)
    Xc = X - X_mean
    Y_mean = Y.mean(axis=0, keepdims=True)
    Yc = Y - Y_mean

    # Precompute Gram matrix for ridge
    XtX = Xc.T @ Xc
    XtY = Xc.T @ Yc

    # Candidate methods and hyperparameters
    candidates = []

    # 1. Current weights (baseline)
    candidates.append(("current", W_cur, b_cur))

    # 2. Ridge regression on centered one-hot labels with various lambdas
    for lam in [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0]:
        W_ridge = np.linalg.solve(XtX + lam * np.eye(D), XtY).T  # [10, 128]
        b_ridge = Y_mean.flatten() - W_ridge @ X_mean.flatten()
        candidates.append((f"ridge_{lam}", W_ridge, b_ridge))

    # 3. Bayesian ridge blending current weights with ridge solution
    #    W_new = (XtX + lam I)^{-1} (XtY + lam W_cur^T)
    for lam in [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]:
        W_bayes = np.linalg.solve(XtX + lam * np.eye(D), XtY + lam * W_cur.T).T
        b_bayes = Y_mean.flatten() - W_bayes @ X_mean.flatten()
        # For bias, blend with current bias as well (simple average with weight)
        # Actually, we can solve for bias separately: b = mean(Y) - W @ mean(X)
        # But to keep it simple, we just use the ridge bias.
        candidates.append((f"bayes_{lam}", W_bayes, b_bayes))

    # 4. LDA with shrinkage
    # Compute class means and pooled within-class covariance
    mu = np.zeros((C, D))
    for c in range(C):
        mu[c] = X[y == c].mean(axis=0)
    # Pooled covariance
    Sw = np.zeros((D, D))
    for c in range(C):
        Xc_c = X[y == c] - mu[c]
        Sw += Xc_c.T @ Xc_c
    Sw /= (N - C)
    # Shrinkage
    for shrink in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        Sw_reg = (1 - shrink) * Sw + shrink * np.trace(Sw) / D * np.eye(D)
        # LDA weights: W = Sw_reg^{-1} @ mu.T
        W_lda = np.linalg.solve(Sw_reg, mu.T).T  # [10, 128]
        # Bias: -0.5 * mu @ W_lda.T + log prior (uniform)
        b_lda = -0.5 * np.sum(mu * W_lda, axis=1)
        candidates.append((f"lda_{shrink}", W_lda, b_lda))

    # 5. One-vs-rest ridge on labels in {-1, 1}
    Y_ovr = 2 * Y - 1
    Y_ovr_mean = Y_ovr.mean(axis=0, keepdims=True)
    Y_ovr_c = Y_ovr - Y_ovr_mean
    XtY_ovr = Xc.T @ Y_ovr_c
    for lam in [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]:
        W_ovr = np.linalg.solve(XtX + lam * np.eye(D), XtY_ovr).T
        b_ovr = Y_ovr_mean.flatten() - W_ovr @ X_mean.flatten()
        candidates.append((f"ovr_{lam}", W_ovr, b_ovr))

    # Evaluate candidates by 5-fold cross-validation on the full 2000 pool
    # We'll use a simple 5-fold split (deterministic)
    np.random.seed(42)
    indices = np.random.permutation(N)
    folds = np.array_split(indices, 5)

    best_cv_acc = -1.0
    best_candidate = None
    best_name = None

    for name, W_cand, b_cand in candidates:
        # For each fold, train on 4/5 and evaluate on 1/5
        # But we already have the full-data solution. To do proper CV, we need to refit on each fold.
        # However, for simplicity and speed, we can just evaluate the full-data solution on the whole set?
        # That would be training accuracy, not CV. We need CV to avoid overfitting.
        # Let's implement a quick CV for the ridge/bayes/lda methods.
        # Since we have many candidates, we can do CV for each by refitting on the training folds.
        # This is a bit heavy but doable for 2000x128.
        # We'll do 5-fold CV for each candidate method (not for current, just evaluate current on each fold).
        # Actually, we can just evaluate the full-data solution on the whole set? No, that's training accuracy.
        # Let's do proper CV for the top methods. To save time, we can do CV only for the best few.
        # But we need to select the best. Let's do CV for all candidates.
        # We'll write a function to fit a method on a subset and predict.
        pass

    # Since doing full CV for all candidates is complex, let's simplify:
    # We'll use the full 2000 to compute the candidates, and then evaluate them on the paired batch
    # to select the best. But the paired batch is part of the training pool, so this is overfitting.
    # However, the prompt says we can use the batch. But to avoid overfitting, we should use CV.
    # Let's do a proper 5-fold CV for the main methods: ridge, bayes, lda, ovr.
    # We'll implement a generic CV loop.

    def cv_score(method, param):
        accs = []
        for i in range(5):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx], y[test_idx]
            N_tr = len(train_idx)
            Y_tr = np.zeros((N_tr, C))
            Y_tr[np.arange(N_tr), y_tr] = 1.0
            X_mean_tr = X_tr.mean(axis=0, keepdims=True)
            Xc_tr = X_tr - X_mean_tr
            Y_mean_tr = Y_tr.mean(axis=0, keepdims=True)
            Yc_tr = Y_tr - Y_mean_tr
            XtX_tr = Xc_tr.T @ Xc_tr
            XtY_tr = Xc_tr.T @ Yc_tr

            if method == "current":
                W = W_cur
                b = b_cur
            elif method == "ridge":
                W = np.linalg.solve(XtX_tr + param * np.eye(D), XtY_tr).T
                b = Y_mean_tr.flatten() - W @ X_mean_tr.flatten()
            elif method == "bayes":
                W = np.linalg.solve(XtX_tr + param * np.eye(D), XtY_tr + param * W_cur.T).T
                b = Y_mean_tr.flatten() - W @ X_mean_tr.flatten()
            elif method == "lda":
                mu_tr = np.zeros((C, D))
                for c in range(C):
                    mu_tr[c] = X_tr[y_tr == c].mean(axis=0)
                Sw_tr = np.zeros((D, D))
                for c in range(C):
                    Xc_c = X_tr[y_tr == c] - mu_tr[c]
                    Sw_tr += Xc_c.T @ Xc_c
                Sw_tr /= (N_tr - C)
                Sw_reg = (1 - param) * Sw_tr + param * np.trace(Sw_tr) / D * np.eye(D)
                W = np.linalg.solve(Sw_reg, mu_tr.T).T
                b = -0.5 * np.sum(mu_tr * W, axis=1)
            elif method == "ovr":
                Y_ovr_tr = 2 * Y_tr - 1
                Y_ovr_mean_tr = Y_ovr_tr.mean(axis=0, keepdims=True)
                Y_ovr_c_tr = Y_ovr_tr - Y_ovr_mean_tr
                XtY_ovr_tr = Xc_tr.T @ Y_ovr_c_tr
                W = np.linalg.solve(XtX_tr + param * np.eye(D), XtY_ovr_tr).T
                b = Y_ovr_mean_tr.flatten() - W @ X_mean_tr.flatten()
            else:
                raise ValueError

            logits = X_te @ W.T + b
            preds = np.argmax(logits, axis=1)
            acc = np.mean(preds == y_te)
            accs.append(acc)
        return np.mean(accs)

    # Evaluate CV for a set of hyperparameters
    cv_results = []
    # Current
    cv_results.append(("current", None, cv_score("current", None)))
    # Ridge
    for lam in [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0]:
        cv_results.append((f"ridge_{lam}", lam, cv_score("ridge", lam)))
    # Bayes
    for lam in [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]:
        cv_results.append((f"bayes_{lam}", lam, cv_score("bayes", lam)))
    # LDA
    for shrink in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        cv_results.append((f"lda_{shrink}", shrink, cv_score("lda", shrink)))
    # OVR
    for lam in [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]:
        cv_results.append((f"ovr_{lam}", lam, cv_score("ovr", lam)))

    # Select best
    best_name, best_param, best_cv = max(cv_results, key=lambda x: x[2])

    # Now fit the best method on the full 2000 data
    if best_name == "current":
        W_new = W_cur
        b_new = b_cur
    elif best_name.startswith("ridge"):
        lam = best_param
        W_new = np.linalg.solve(XtX + lam * np.eye(D), XtY).T
        b_new = Y_mean.flatten() - W_new @ X_mean.flatten()
    elif best_name.startswith("bayes"):
        lam = best_param
        W_new = np.linalg.solve(XtX + lam * np.eye(D), XtY + lam * W_cur.T).T
        b_new = Y_mean.flatten() - W_new @ X_mean.flatten()
    elif best_name.startswith("lda"):
        shrink = best_param
        mu = np.zeros((C, D))
        for c in range(C):
            mu[c] = X[y == c].mean(axis=0)
        Sw = np.zeros((D, D))
        for c in range(C):
            Xc_c = X[y == c] - mu[c]
            Sw += Xc_c.T @ Xc_c
        Sw /= (N - C)
        Sw_reg = (1 - shrink) * Sw + shrink * np.trace(Sw) / D * np.eye(D)
        W_new = np.linalg.solve(Sw_reg, mu.T).T
        b_new = -0.5 * np.sum(mu * W_new, axis=1)
    elif best_name.startswith("ovr"):
        lam = best_param
        Y_ovr = 2 * Y - 1
        Y_ovr_mean = Y_ovr.mean(axis=0, keepdims=True)
        Y_ovr_c = Y_ovr - Y_ovr_mean
        XtY_ovr = Xc.T @ Y_ovr_c
        W_new = np.linalg.solve(XtX + lam * np.eye(D), XtY_ovr).T
        b_new = Y_ovr_mean.flatten() - W_new @ X_mean.flatten()
    else:
        W_new = W_cur
        b_new = b_cur

    # Ensure finite and within bounds
    W_new = np.clip(W_new, -1000.0, 1000.0).astype(np.float32)
    b_new = np.clip(b_new, -1000.0, 1000.0).astype(np.float32)

    # Update weights
    w["fc2.weight"] = W_new
    w["fc2.bias"] = b_new

    # Diagnostics
    diagnostics = {
        "best_method": best_name,
        "best_cv_accuracy": float(best_cv),
        "cv_results_top5": sorted([(name, float(cv)) for name, _, cv in cv_results], key=lambda x: -x[1])[:5],
        "fc2_weight_norm": float(np.linalg.norm(W_new)),
        "fc2_bias_norm": float(np.linalg.norm(b_new)),
    }

    return {"weights": w, "diagnostics": diagnostics}
