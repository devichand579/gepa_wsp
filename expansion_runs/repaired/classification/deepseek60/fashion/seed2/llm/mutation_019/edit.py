def edit(w, f):
    X = f["features"].astype(np.float64)
    y = f["y"].astype(np.int64)
    N, D = X.shape
    C = 10

    W_cur = w["head.weight"].astype(np.float64)
    b_cur = w["head.bias"].astype(np.float64)

    def evaluate(W, b):
        logits = X @ W.T + b
        logits_max = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        preds = np.argmax(logits, axis=1)
        acc = np.mean(preds == y)
        log_probs = logits - logits_max - np.log(np.sum(np.exp(logits - logits_max), axis=1, keepdims=True))
        ce = -np.mean(log_probs[np.arange(N), y])
        return acc, ce, preds

    acc_cur, ce_cur, _ = evaluate(W_cur, b_cur)

    # LDA
    mu = np.zeros((C, D))
    for c in range(C):
        mu[c] = X[y == c].mean(axis=0)
    X_T_X = X.T @ X
    S_w = X_T_X.copy()
    for c in range(C):
        n_c = np.sum(y == c)
        S_w -= n_c * np.outer(mu[c], mu[c])
    lambda_lda = 1e-6 * np.trace(S_w) / D
    S_w_reg = S_w + lambda_lda * np.eye(D)
    W_lda = np.linalg.solve(S_w_reg, mu.T).T
    priors = np.array([np.sum(y == c) for c in range(C)]) / N
    b_lda = -0.5 * np.sum(mu * W_lda, axis=1) + np.log(priors)
    acc_lda, ce_lda, _ = evaluate(W_lda, b_lda)

    # Ridge
    X_aug = np.hstack([X, np.ones((N, 1))])
    Y = np.zeros((N, C))
    Y[np.arange(N), y] = 1.0
    X_aug_T_X_aug = X_aug.T @ X_aug
    lambda_ridge = 1e-6 * np.trace(X_aug_T_X_aug) / (D + 1)
    W_aug_ridge = np.linalg.solve(X_aug_T_X_aug + lambda_ridge * np.eye(D + 1), X_aug.T @ Y)
    W_ridge = W_aug_ridge[:D].T
    b_ridge = W_aug_ridge[D]
    acc_ridge, ce_ridge, _ = evaluate(W_ridge, b_ridge)

    candidates = [
        (acc_cur, ce_cur, W_cur, b_cur, "current"),
        (acc_lda, ce_lda, W_lda, b_lda, "lda"),
        (acc_ridge, ce_ridge, W_ridge, b_ridge, "ridge")
    ]
    candidates.sort(key=lambda x: (-x[0], x[1]))
    best_acc, best_ce, best_W, best_b, best_name = candidates[0]

    best_W = np.clip(best_W, -1000.0, 1000.0)
    best_b = np.clip(best_b, -1000.0, 1000.0)

    w["head.weight"] = best_W.astype(np.float32)
    w["head.bias"] = best_b.astype(np.float32)

    diagnostics = {
        "chosen": best_name,
        "acc_current": float(acc_cur),
        "ce_current": float(ce_cur),
        "acc_lda": float(acc_lda),
        "ce_lda": float(ce_lda),
        "acc_ridge": float(acc_ridge),
        "ce_ridge": float(ce_ridge),
        "best_acc": float(best_acc),
        "best_ce": float(best_ce),
        "max_abs_head_weight": float(np.max(np.abs(best_W))),
        "max_abs_head_bias": float(np.max(np.abs(best_b)))
    }
    return {"weights": w, "diagnostics": diagnostics}
