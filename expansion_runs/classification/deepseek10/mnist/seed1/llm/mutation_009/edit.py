import numpy as np

def edit(w, f):
    X = f["features"].astype(np.float64)   # (2000,128)
    y = f["y"].astype(np.int64)            # (2000,)
    n, d = X.shape
    C = 10

    # One-hot targets
    Y = np.zeros((n, C))
    Y[np.arange(n), y] = 1

    # ---------- Ridge with GCV ----------
    Xb = np.hstack([X, np.ones((n, 1))])   # (2000,129)
    D = np.eye(d + 1)
    D[d, d] = 0.0                          # do not regularize bias

    A = Xb.T @ Xb
    lambdas = np.logspace(-6, 2, 50)
    gcv_scores = []
    for lam in lambdas:
        B = A + lam * D
        B_inv = np.linalg.inv(B)
        W = B_inv @ Xb.T @ Y
        trace_H = np.trace(B_inv @ A)
        resid = Y - Xb @ W
        rss = np.sum(resid ** 2)
        gcv = (rss / n) / (1.0 - trace_H / n) ** 2
        gcv_scores.append(gcv)
    best_lam = lambdas[np.argmin(gcv_scores)]

    B = A + best_lam * D
    W_ridge = np.linalg.solve(B, Xb.T @ Y)   # (129,10)
    fc2_w_ridge = W_ridge[:d, :].T           # (10,128)
    fc2_b_ridge = W_ridge[d, :]              # (10,)

    # ---------- LDA ----------
    mu_c = np.zeros((C, d))
    for c in range(C):
        mu_c[c] = np.mean(X[y == c], axis=0)
    S_w = np.zeros((d, d))
    for c in range(C):
        Xc = X[y == c] - mu_c[c]
        S_w += Xc.T @ Xc
    alpha = 1e-6 * np.trace(S_w) / d
    S_w_reg = S_w + alpha * np.eye(d)
    S_w_inv = np.linalg.inv(S_w_reg)
    W_lda = S_w_inv @ mu_c.T                 # (d, C)
    b_lda = -0.5 * np.sum(mu_c * (S_w_inv @ mu_c.T).T, axis=1)  # (C,)
    fc2_w_lda = W_lda.T                      # (10,128)
    fc2_b_lda = b_lda                        # (10,)

    # ---------- 5-fold CV to choose method ----------
    rng = np.random.RandomState(0)
    idx = rng.permutation(n)
    folds = np.array_split(idx, 5)
    acc_ridge, acc_lda = [], []
    for i in range(5):
        val = folds[i]
        tr = np.concatenate([folds[j] for j in range(5) if j != i])
        X_tr, y_tr = X[tr], y[tr]
        X_val, y_val = X[val], y[val]
        n_tr = len(tr)
        Y_tr = np.zeros((n_tr, C))
        Y_tr[np.arange(n_tr), y_tr] = 1

        # Ridge on training fold (use same lambda selection logic)
        Xb_tr = np.hstack([X_tr, np.ones((n_tr, 1))])
        A_tr = Xb_tr.T @ Xb_tr
        best_lam_tr = best_lam  # reuse global lambda for speed
        B_tr = A_tr + best_lam_tr * D
        W_tr = np.linalg.solve(B_tr, Xb_tr.T @ Y_tr)
        w_r = W_tr[:d, :].T
        b_r = W_tr[d, :]
        pred_r = np.argmax(X_val @ w_r.T + b_r, axis=1)
        acc_ridge.append(np.mean(pred_r == y_val))

        # LDA on training fold
        mu_c_tr = np.zeros((C, d))
        for c in range(C):
            mu_c_tr[c] = np.mean(X_tr[y_tr == c], axis=0)
        S_w_tr = np.zeros((d, d))
        for c in range(C):
            Xc = X_tr[y_tr == c] - mu_c_tr[c]
            S_w_tr += Xc.T @ Xc
        alpha_tr = 1e-6 * np.trace(S_w_tr) / d
        S_w_inv_tr = np.linalg.inv(S_w_tr + alpha_tr * np.eye(d))
        W_lda_tr = S_w_inv_tr @ mu_c_tr.T
        b_lda_tr = -0.5 * np.sum(mu_c_tr * (S_w_inv_tr @ mu_c_tr.T).T, axis=1)
        pred_l = np.argmax(X_val @ W_lda_tr + b_lda_tr, axis=1)
        acc_lda.append(np.mean(pred_l == y_val))

    cv_ridge = np.mean(acc_ridge)
    cv_lda = np.mean(acc_lda)

    # Choose the better method
    if cv_lda > cv_ridge:
        w["fc2.weight"] = fc2_w_lda.astype(np.float32)
        w["fc2.bias"]   = fc2_b_lda.astype(np.float32)
        chosen = "lda"
        cv_acc = cv_lda
    else:
        w["fc2.weight"] = fc2_w_ridge.astype(np.float32)
        w["fc2.bias"]   = fc2_b_ridge.astype(np.float32)
        chosen = "ridge"
        cv_acc = cv_ridge

    # Diagnostics
    diagnostics = {
        "chosen_method": chosen,
        "cv_ridge": float(cv_ridge),
        "cv_lda": float(cv_lda),
        "best_lambda": float(best_lam),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
    }
    return {"weights": w, "diagnostics": diagnostics}
