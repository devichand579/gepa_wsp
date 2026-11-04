import numpy as np

def edit(w, f):
    X = f["features"].astype(np.float64)
    y = f["y"].astype(np.int64)
    N, D = X.shape
    C = 10

    W_cur = w["head.weight"].astype(np.float64)
    b_cur = w["head.bias"].astype(np.float64)

    def compute_metrics(W, b):
        logits = X @ W.T + b
        preds = np.argmax(logits, axis=1)
        acc = np.mean(preds == y)
        logits_max = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        ce = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))
        return acc, ce

    acc_cur, ce_cur = compute_metrics(W_cur, b_cur)

    # Augment features with a bias column
    X_aug = np.hstack([X, np.ones((N, 1))])
    D_aug = X_aug.shape[1]

    # Ridge regression solver
    def ridge_solve(Y, lam=1e-3):
        XtX = X_aug.T @ X_aug
        XtY = X_aug.T @ Y
        reg = lam * np.eye(D_aug)
        reg[-1, -1] = 0.0  # do not regularize bias
        W = np.linalg.solve(XtX + reg, XtY)
        return W

    # Candidate 1: ±1 targets
    Y_pm1 = -np.ones((N, C))
    Y_pm1[np.arange(N), y] = 1.0
    W_pm1 = ridge_solve(Y_pm1)
    W_head_pm1 = W_pm1[:-1, :].T
    b_head_pm1 = W_pm1[-1, :]
    acc_pm1, ce_pm1 = compute_metrics(W_head_pm1, b_head_pm1)

    # Candidate 2: one-hot targets
    Y_oh = np.zeros((N, C))
    Y_oh[np.arange(N), y] = 1.0
    W_oh = ridge_solve(Y_oh)
    W_head_oh = W_oh[:-1, :].T
    b_head_oh = W_oh[-1, :]
    acc_oh, ce_oh = compute_metrics(W_head_oh, b_head_oh)

    # Pick the best candidate
    candidates = [
        (acc_pm1, ce_pm1, W_head_pm1, b_head_pm1),
        (acc_oh, ce_oh, W_head_oh, b_head_oh),
    ]
    candidates.sort(key=lambda x: (-x[0], x[1]))
    best_acc, best_ce, best_W, best_b = candidates[0]

    # Scale to maximum allowed magnitude (1000)
    max_abs_w = np.max(np.abs(best_W))
    max_abs_b = np.max(np.abs(best_b))
    if max_abs_w > 0:
        scale = 1000.0 / max_abs_w
        if max_abs_b > 0:
            scale = min(scale, 1000.0 / max_abs_b)
        best_W *= scale
        best_b *= scale
        best_acc, best_ce = compute_metrics(best_W, best_b)

    # Update if better than current
    if best_acc > acc_cur or (best_acc == acc_cur and best_ce < ce_cur):
        w["head.weight"] = best_W.astype(np.float32)
        w["head.bias"] = best_b.astype(np.float32)
        used_new = True
    else:
        used_new = False

    diagnostics = {
        "acc_cur": float(acc_cur),
        "ce_cur": float(ce_cur),
        "acc_pm1": float(acc_pm1),
        "ce_pm1": float(ce_pm1),
        "acc_oh": float(acc_oh),
        "ce_oh": float(ce_oh),
        "best_acc": float(best_acc),
        "best_ce": float(best_ce),
        "used_new": used_new,
        "max_abs_weight": float(np.max(np.abs(w["head.weight"]))),
        "max_abs_bias": float(np.max(np.abs(w["head.bias"]))),
    }
    return {"weights": w, "diagnostics": diagnostics}
