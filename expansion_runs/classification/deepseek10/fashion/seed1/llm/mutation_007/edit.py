import numpy as np

def edit(w, f):
    # Extract full feedback features and labels
    F = f["features"].astype(np.float64)  # (2000, 1568)
    y = f["y"].astype(np.int64)           # (2000,)
    N, D = F.shape
    C = 10

    # One-hot targets for ridge
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Fixed CV split
    indices = np.arange(N)
    rng = np.random.RandomState(0)
    rng.shuffle(indices)
    folds = np.array_split(indices, 5)

    # Candidate regularization strengths (scaled by trace/D)
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

    # ---------- LDA CV ----------
    lda_accs = {lam: [] for lam in lambdas}
    for i in range(5):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
        F_tr = F[train_idx]
        y_tr = y[train_idx]
        F_va = F[val_idx]
        y_va = y[val_idx]

        # Class means on training fold
        mu = np.zeros((C, D))
        for c in range(C):
            mask = (y_tr == c)
            if mask.sum() > 0:
                mu[c] = F_tr[mask].mean(axis=0)

        # Within-class scatter
        S_w = np.zeros((D, D))
        for c in range(C):
            mask = (y_tr == c)
            if mask.sum() > 0:
                Fc = F_tr[mask]
                diff = Fc - mu[c]
                S_w += diff.T @ diff

        trace = np.trace(S_w)
        base_reg = trace / D if trace > 0 else 1.0

        for lam in lambdas:
            S_reg = S_w + lam * base_reg * np.eye(D)
            W = np.linalg.solve(S_reg, mu.T)  # (D, C)
            bias = np.zeros(C)
            for c in range(C):
                n_c = (y_tr == c).sum()
                pi_c = n_c / len(train_idx) if len(train_idx) > 0 else 1.0 / C
                bias[c] = -0.5 * np.dot(mu[c], W[:, c]) + np.log(pi_c + 1e-12)
            scores = F_va @ W + bias
            pred = np.argmax(scores, axis=1)
            lda_accs[lam].append(np.mean(pred == y_va))

    lda_mean = {lam: np.mean(accs) for lam, accs in lda_accs.items()}
    best_lda_lam = max(lda_mean, key=lda_mean.get)
    best_lda_acc = lda_mean[best_lda_lam]

    # ---------- Ridge CV ----------
    ridge_accs = {lam: [] for lam in lambdas}
    for i in range(5):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
        F_tr = F[train_idx]
        Y_tr = Y[train_idx]
        F_va = F[val_idx]
        y_va = y[val_idx]

        mu_F = F_tr.mean(axis=0)
        mu_Y = Y_tr.mean(axis=0)
        F_c = F_tr - mu_F
        Y_c = Y_tr - mu_Y

        FtF = F_c.T @ F_c
        FtY = F_c.T @ Y_c
        trace = np.trace(FtF)
        base_reg = trace / D if trace > 0 else 1.0

        for lam in lambdas:
            FtF_reg = FtF + lam * base_reg * np.eye(D)
            W = np.linalg.solve(FtF_reg, FtY)  # (D, C)
            bias = mu_Y - mu_F @ W
            scores = F_va @ W + bias
            pred = np.argmax(scores, axis=1)
            ridge_accs[lam].append(np.mean(pred == y_va))

    ridge_mean = {lam: np.mean(accs) for lam, accs in ridge_accs.items()}
    best_ridge_lam = max(ridge_mean, key=ridge_mean.get)
    best_ridge_acc = ridge_mean[best_ridge_lam]

    # ---------- Choose better method and refit on full data ----------
    if best_lda_acc >= best_ridge_acc:
        # Full-data LDA
        mu = np.zeros((C, D))
        for c in range(C):
            mask = (y == c)
            mu[c] = F[mask].mean(axis=0)

        S_w = np.zeros((D, D))
        for c in range(C):
            mask = (y == c)
            Fc = F[mask]
            diff = Fc - mu[c]
            S_w += diff.T @ diff

        trace = np.trace(S_w)
        base_reg = trace / D if trace > 0 else 1.0
        S_reg = S_w + best_lda_lam * base_reg * np.eye(D)
        W = np.linalg.solve(S_reg, mu.T)  # (D, C)
        bias = np.zeros(C)
        for c in range(C):
            n_c = (y == c).sum()
            pi_c = n_c / N
            bias[c] = -0.5 * np.dot(mu[c], W[:, c]) + np.log(pi_c + 1e-12)

        method = "LDA"
        best_lam = best_lda_lam
        cv_acc = best_lda_acc
    else:
        # Full-data ridge
        mu_F = F.mean(axis=0)
        mu_Y = Y.mean(axis=0)
        F_c = F - mu_F
        Y_c = Y - mu_Y
        FtF = F_c.T @ F_c
        FtY = F_c.T @ Y_c
        trace = np.trace(FtF)
        base_reg = trace / D if trace > 0 else 1.0
        FtF_reg = FtF + best_ridge_lam * base_reg * np.eye(D)
        W = np.linalg.solve(FtF_reg, FtY)
        bias = mu_Y - mu_F @ W

        method = "Ridge"
        best_lam = best_ridge_lam
        cv_acc = best_ridge_acc

    # Write new head parameters
    w["head.weight"] = W.T.astype(np.float32)   # (10, 1568)
    w["head.bias"]   = bias.astype(np.float32)  # (10,)

    # Diagnostics
    logits = F @ w["head.weight"].T + w["head.bias"]
    pred = np.argmax(logits, axis=1)
    train_acc = float(np.mean(pred == y))

    logits_max = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    ce = float(-np.mean(np.log(probs[np.arange(N), y] + 1e-12)))

    diagnostics = {
        "method": method,
        "best_lambda": float(best_lam),
        "cv_accuracy": float(cv_acc),
        "train_accuracy": train_acc,
        "train_cross_entropy": ce,
        "head_weight_mean": float(w["head.weight"].mean()),
        "head_weight_std": float(w["head.weight"].std()),
        "head_bias": w["head.bias"].tolist()
    }
    return {"weights": w, "diagnostics": diagnostics}
