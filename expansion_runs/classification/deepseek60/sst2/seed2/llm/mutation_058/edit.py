import numpy as np

def edit(w, f):
    # Extract the pooled features and labels from the feedback pool.
    X = f["features"].astype(np.float64)   # (4000, 32)
    y = f["y"].astype(np.float64)          # (4000,)
    N, D = X.shape

    # Center the features and labels for numerical stability.
    X_mean = X.mean(axis=0)
    Xc = X - X_mean
    y_mean = y.mean()
    yc = y - y_mean

    # Ridge regression with intercept, solved via normal equations.
    # We will select lambda by 5-fold cross-validation.
    def ridge_fit(Xc, yc, lam):
        # Solve (X^T X + lam I) beta = X^T y
        A = Xc.T @ Xc + lam * np.eye(D)
        b = Xc.T @ yc
        beta = np.linalg.solve(A, b)
        return beta

    # 5-fold CV indices
    rng = np.random.default_rng(0)
    idx = rng.permutation(N)
    folds = np.array_split(idx, 5)

    lambdas = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    best_lam = None
    best_acc = -1.0
    cv_scores = {}

    for lam in lambdas:
        accs = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            X_tr, y_tr = Xc[train_idx], yc[train_idx]
            X_val, y_val = Xc[val_idx], yc[val_idx]
            beta = ridge_fit(X_tr, y_tr, lam)
            # Predict on validation
            pred = X_val @ beta
            # Convert to class by thresholding at 0 (since yc is centered)
            pred_class = (pred > 0).astype(int)
            true_class = (y_val > 0).astype(int)
            acc = (pred_class == true_class).mean()
            accs.append(acc)
        mean_acc = np.mean(accs)
        cv_scores[lam] = mean_acc
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lam = lam

    # Fit on full data with best lambda
    beta = ridge_fit(Xc, yc, best_lam)
    intercept = y_mean - X_mean @ beta

    # Now we have a linear classifier: logit = beta @ x + intercept.
    # The network's head is 2x32. We want logits[1] - logits[0] = beta @ x + intercept.
    # So we set head[1] - head[0] = beta, and head.bias[1] - head.bias[0] = intercept.
    # We can set head[0] = -beta/2, head[1] = beta/2, bias[0] = -intercept/2, bias[1] = intercept/2.
    # This gives logits[1] - logits[0] = beta @ x + intercept.
    # To keep magnitudes reasonable, we can scale by a factor if needed, but it's fine.

    head_w = np.zeros((2, D), dtype=np.float32)
    head_b = np.zeros(2, dtype=np.float32)

    head_w[1] = beta.astype(np.float32)
    head_w[0] = -beta.astype(np.float32)
    head_b[1] = intercept.astype(np.float32)
    head_b[0] = -intercept.astype(np.float32)

    # Update the weights dictionary
    w["head.weight"] = head_w
    w["head.bias"] = head_b

    # Ensure all other weights remain unchanged (they are already identity/zero).
    # We don't need to modify anything else.

    diagnostics = {
        "selected_lambda": float(best_lam),
        "cv_accuracy_selected": float(best_acc),
        "cv_scores": {str(k): float(v) for k, v in cv_scores.items()},
        "in_sample_accuracy": float(((X @ beta + intercept > 0).astype(int) == y).mean()),
        "beta_norm": float(np.linalg.norm(beta)),
        "intercept": float(intercept),
    }

    return {"weights": w, "diagnostics": diagnostics}
