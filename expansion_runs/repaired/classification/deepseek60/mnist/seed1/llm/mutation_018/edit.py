import numpy as np
import math

def edit(w, f):
    # Extract full feedback pool
    H = f["features"].astype(np.float64)          # (2000, 128)
    y = f["y"].astype(np.int64)                   # (2000,)
    N, D = H.shape
    C = 10

    # One-hot labels
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Current fc2 parameters (for prior methods)
    W0 = w["fc2.weight"].astype(np.float64)       # (10, 128)
    b0 = w["fc2.bias"].astype(np.float64)         # (10,)

    # Helper: ridge regression with intercept
    def fit_ridge(H_tr, Y_tr, lam):
        mu = H_tr.mean(axis=0)
        Hc = H_tr - mu
        Yc = Y_tr - Y_tr.mean(axis=0)
        A = Hc.T @ Hc + lam * np.eye(D)
        W = np.linalg.solve(A, Hc.T @ Yc).T       # (10, 128)
        b = Y_tr.mean(axis=0) - mu @ W.T
        return W, b

    # Helper: LDA with regularized within-class scatter
    def fit_lda(H_tr, y_tr, lam):
        mu_c = np.zeros((C, D))
        counts = np.zeros(C)
        for c in range(C):
            mask = (y_tr == c)
            counts[c] = mask.sum()
            if counts[c] > 0:
                mu_c[c] = H_tr[mask].mean(axis=0)
        mu = H_tr.mean(axis=0)
        Sw = np.zeros((D, D))
        for c in range(C):
            mask = (y_tr == c)
            if counts[c] > 1:
                diff = H_tr[mask] - mu_c[c]
                Sw += diff.T @ diff
        Sw_reg = Sw + lam * np.eye(D)
        invSw = np.linalg.inv(Sw_reg)
        W = mu_c @ invSw                            # (10, 128)
        b = -0.5 * np.sum(mu_c * (mu_c @ invSw), axis=1) + np.log(counts / counts.sum() + 1e-12)
        return W, b

    # Helper: ridge with prior toward current fc2
    def fit_ridge_prior(H_tr, Y_tr, lam, lam_b, W0, b0):
        N_tr = H_tr.shape[0]
        u = H_tr.sum(axis=0)                        # (D,)
        S = H_tr.T @ H_tr + lam * np.eye(D)
        M = S - (1.0 / (N_tr + lam_b)) * np.outer(u, u)
        RHS = Y_tr.T @ H_tr + lam * W0 - (1.0 / (N_tr + lam_b)) * np.outer(Y_tr.sum(axis=0) + lam_b * b0, u)
        W = np.linalg.solve(M.T, RHS.T).T           # (10, 128)
        b = (Y_tr.sum(axis=0) + lam_b * b0 - W @ u) / (N_tr + lam_b)
        return W, b

    # Cross-validation setup
    n_folds = 5
    indices = np.arange(N)
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    best_acc = -1.0
    best_method = None
    best_params = None

    # Lambda grids
    lam_ridge = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    lam_lda   = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    lam_prior = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

    # Evaluate standard ridge
    for lam in lam_ridge:
        accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            tr_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            W, b = fit_ridge(H[tr_idx], Y[tr_idx], lam)
            scores = H[val_idx] @ W.T + b
            pred = np.argmax(scores, axis=1)
            accs.append(np.mean(pred == y[val_idx]))
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_method = "ridge"
            best_params = (lam,)

    # Evaluate LDA
    for lam in lam_lda:
        accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            tr_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            W, b = fit_lda(H[tr_idx], y[tr_idx], lam)
            scores = H[val_idx] @ W.T + b
            pred = np.argmax(scores, axis=1)
            accs.append(np.mean(pred == y[val_idx]))
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_method = "lda"
            best_params = (lam,)

    # Evaluate ridge with prior
    for lam in lam_prior:
        accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            tr_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            W, b = fit_ridge_prior(H[tr_idx], Y[tr_idx], lam, lam, W0, b0)
            scores = H[val_idx] @ W.T + b
            pred = np.argmax(scores, axis=1)
            accs.append(np.mean(pred == y[val_idx]))
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_method = "ridge_prior"
            best_params = (lam,)

    # Fit best method on all data
    if best_method == "ridge":
        lam = best_params[0]
        W, b = fit_ridge(H, Y, lam)
    elif best_method == "lda":
        lam = best_params[0]
        W, b = fit_lda(H, y, lam)
    else:  # ridge_prior
        lam = best_params[0]
        W, b = fit_ridge_prior(H, Y, lam, lam, W0, b0)

    # Update weights
    w["fc2.weight"] = W.astype(np.float32)
    w["fc2.bias"] = b.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "method": best_method,
        "params": best_params,
        "cv_accuracy": float(best_acc),
        "fc2_weight_mean": float(w["fc2.weight"].mean()),
        "fc2_weight_std": float(w["fc2.weight"].std()),
        "fc2_bias_mean": float(w["fc2.bias"].mean()),
        "fc2_bias_std": float(w["fc2.bias"].std()),
    }

    return {"weights": w, "diagnostics": diagnostics}
