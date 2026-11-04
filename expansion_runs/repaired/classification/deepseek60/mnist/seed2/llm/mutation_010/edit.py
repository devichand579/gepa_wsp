import numpy as np

def edit(w, f):
    # Extract full feedback features and labels
    F = f["features"].astype(np.float64)  # (2000, 128)
    y = f["y"].astype(np.int64)           # (2000,)
    N, D = F.shape
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Augment features with bias column for ridge
    X = np.hstack([F, np.ones((N, 1), dtype=np.float64)])  # (2000, 129)

    # Deterministic 5-fold split
    n_folds = 5
    fold_ids = np.arange(N) % n_folds

    def cv_score_ridge(alpha):
        accs = []
        for k in range(n_folds):
            train = fold_ids != k
            val = fold_ids == k
            X_tr, Y_tr = X[train], Y[train]
            X_val, y_val = X[val], y[val]
            A = X_tr.T @ X_tr + alpha * np.eye(X.shape[1])
            W = np.linalg.solve(A, X_tr.T @ Y_tr).T  # (C, D+1)
            logits = X_val @ W.T
            pred = np.argmax(logits, axis=1)
            accs.append(np.mean(pred == y_val))
        return np.mean(accs)

    def cv_score_lda(alpha):
        # alpha is shrinkage toward scaled identity: S_reg = (1-alpha)*S + alpha*(trace(S)/D)*I
        accs = []
        for k in range(n_folds):
            train = fold_ids != k
            val = fold_ids == k
            F_tr, y_tr = F[train], y[train]
            F_val, y_val = F[val], y[val]

            mus = np.zeros((C, D), dtype=np.float64)
            priors = np.zeros(C, dtype=np.float64)
            for c in range(C):
                mask = y_tr == c
                if np.sum(mask) > 0:
                    mus[c] = F_tr[mask].mean(axis=0)
                    priors[c] = np.sum(mask) / len(y_tr)
                else:
                    mus[c] = 0.0
                    priors[c] = 1.0 / C

            S = np.zeros((D, D), dtype=np.float64)
            for c in range(C):
                mask = y_tr == c
                if np.sum(mask) > 1:
                    diff = F_tr[mask] - mus[c]
                    S += diff.T @ diff
            S /= max(1, len(y_tr) - C)

            S_reg = (1.0 - alpha) * S + alpha * (np.trace(S) / D) * np.eye(D)
            try:
                S_inv = np.linalg.inv(S_reg)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S_reg)

            W = mus @ S_inv  # (C, D)
            b = -0.5 * np.sum(mus * (mus @ S_inv), axis=1) + np.log(priors + 1e-12)
            logits = F_val @ W.T + b
            pred = np.argmax(logits, axis=1)
            accs.append(np.mean(pred == y_val))
        return np.mean(accs)

    # Candidate hyperparameters
    alphas_ridge = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]
    alphas_lda = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99]

    best_ridge_acc = -1.0
    best_ridge_alpha = None
    for alpha in alphas_ridge:
        acc = cv_score_ridge(alpha)
        if acc > best_ridge_acc:
            best_ridge_acc = acc
            best_ridge_alpha = alpha

    best_lda_acc = -1.0
    best_lda_alpha = None
    for alpha in alphas_lda:
        acc = cv_score_lda(alpha)
        if acc > best_lda_acc:
            best_lda_acc = acc
            best_lda_alpha = alpha

    # Choose the better method
    if best_ridge_acc >= best_lda_acc:
        method = "ridge"
        alpha = best_ridge_alpha
        A = X.T @ X + alpha * np.eye(X.shape[1])
        W_full = np.linalg.solve(A, X.T @ Y).T  # (C, D+1)
        W_new = W_full[:, :D]
        b_new = W_full[:, D]
        cv_acc = best_ridge_acc
    else:
        method = "lda"
        alpha = best_lda_alpha
        mus = np.zeros((C, D), dtype=np.float64)
        priors = np.zeros(C, dtype=np.float64)
        for c in range(C):
            mask = y == c
            mus[c] = F[mask].mean(axis=0)
            priors[c] = np.sum(mask) / N

        S = np.zeros((D, D), dtype=np.float64)
        for c in range(C):
            mask = y == c
            diff = F[mask] - mus[c]
            S += diff.T @ diff
        S /= (N - C)

        S_reg = (1.0 - alpha) * S + alpha * (np.trace(S) / D) * np.eye(D)
        try:
            S_inv = np.linalg.inv(S_reg)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S_reg)

        W_new = mus @ S_inv
        b_new = -0.5 * np.sum(mus * (mus @ S_inv), axis=1) + np.log(priors + 1e-12)
        cv_acc = best_lda_acc

    # Clip and cast to float32
    W_new = np.clip(W_new, -1000.0, 1000.0).astype(np.float32)
    b_new = np.clip(b_new, -1000.0, 1000.0).astype(np.float32)

    # Update only fc2
    w["fc2.weight"] = W_new
    w["fc2.bias"] = b_new

    diagnostics = {
        "method": method,
        "alpha": float(alpha),
        "cv_accuracy": float(cv_acc),
        "W_norm": float(np.linalg.norm(W_new)),
        "b_norm": float(np.linalg.norm(b_new)),
        "W_max_abs": float(np.max(np.abs(W_new))),
        "b_max_abs": float(np.max(np.abs(b_new))),
    }

    return {"weights": w, "diagnostics": diagnostics}
