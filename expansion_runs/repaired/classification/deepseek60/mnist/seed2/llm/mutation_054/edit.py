import numpy as np

def edit(w, f):
    X = f["features"].astype(np.float64)          # (2000, 128)
    y = f["y"].astype(np.int64)                   # (2000,)
    N, D = X.shape
    C = 10

    # Current readout metrics
    logits_cur = f["logits"].astype(np.float64)
    pred_cur = np.argmax(logits_cur, axis=1)
    acc_cur = float(np.mean(pred_cur == y))

    def softmax_ce(logits):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return float(-np.mean(np.log(probs[np.arange(N), y] + 1e-12)))

    ce_cur = softmax_ce(logits_cur)

    # Standardize features for numerical stability
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-6] = 1.0
    X_norm = (X - X_mean) / X_std

    # One-hot targets
    Y = np.zeros((N, C))
    Y[np.arange(N), y] = 1.0

    # ---------- LDA on normalized features ----------
    mu = np.zeros((C, D))
    for c in range(C):
        mu[c] = X_norm[y == c].mean(axis=0)

    S = np.zeros((D, D))
    for c in range(C):
        Xc = X_norm[y == c]
        Xc_centered = Xc - mu[c]
        S += Xc_centered.T @ Xc_centered
    S /= (N - C)

    alpha_lda = 1e-3 * np.trace(S) / D
    S_reg = S + alpha_lda * np.eye(D)
    S_inv = np.linalg.inv(S_reg)

    W_lda_norm = mu @ S_inv                         # (C, D)
    b_lda_norm = -0.5 * np.sum(mu * (mu @ S_inv), axis=1) + np.log(1.0 / C)

    # Transform back to original feature space
    W_lda = W_lda_norm / X_std
    b_lda = b_lda_norm - (X_mean / X_std) @ W_lda_norm.T

    logits_lda = X @ W_lda.T + b_lda
    pred_lda = np.argmax(logits_lda, axis=1)
    acc_lda = float(np.mean(pred_lda == y))
    ce_lda = softmax_ce(logits_lda)

    # ---------- Ridge regression on normalized features ----------
    X_norm_aug = np.hstack([X_norm, np.ones((N, 1))])
    alpha_ridge = 1.0
    A = X_norm_aug.T @ X_norm_aug + alpha_ridge * np.eye(D + 1)
    W_ridge_norm = np.linalg.solve(A, X_norm_aug.T @ Y)   # (D+1, C)

    W_ridge_norm_w = W_ridge_norm[:D, :].T                # (C, D)
    b_ridge_norm = W_ridge_norm[D, :]                     # (C,)

    W_ridge = W_ridge_norm_w / X_std
    b_ridge = b_ridge_norm - (X_mean / X_std) @ W_ridge_norm_w.T

    logits_ridge = X @ W_ridge.T + b_ridge
    pred_ridge = np.argmax(logits_ridge, axis=1)
    acc_ridge = float(np.mean(pred_ridge == y))
    ce_ridge = softmax_ce(logits_ridge)

    # ---------- Select best candidate ----------
    candidates = [
        ("current", acc_cur, ce_cur, w["fc2.weight"].copy(), w["fc2.bias"].copy()),
        ("lda",     acc_lda, ce_lda, W_lda.astype(np.float32), b_lda.astype(np.float32)),
        ("ridge",   acc_ridge, ce_ridge, W_ridge.astype(np.float32), b_ridge.astype(np.float32)),
    ]
    candidates.sort(key=lambda x: (-x[1], x[2]))
    best_name, best_acc, best_ce, best_W, best_b = candidates[0]

    # Sanitize and enforce |param| <= 1000
    best_W = np.nan_to_num(best_W, nan=0.0, posinf=1000.0, neginf=-1000.0)
    best_b = np.nan_to_num(best_b, nan=0.0, posinf=1000.0, neginf=-1000.0)

    max_abs = max(np.max(np.abs(best_W)), np.max(np.abs(best_b)))
    if max_abs > 1000.0:
        scale = 1000.0 / max_abs
        best_W = best_W * scale
        best_b = best_b * scale

    w["fc2.weight"] = best_W.astype(np.float32)
    w["fc2.bias"] = best_b.astype(np.float32)

    diagnostics = {
        "chosen": best_name,
        "acc_cur": acc_cur,
        "ce_cur": ce_cur,
        "acc_lda": acc_lda,
        "ce_lda": ce_lda,
        "acc_ridge": acc_ridge,
        "ce_ridge": ce_ridge,
        "best_acc": best_acc,
        "best_ce": best_ce,
        "fc2_weight_norm": float(np.linalg.norm(best_W)),
        "fc2_bias_norm": float(np.linalg.norm(best_b)),
    }
    return {"weights": w, "diagnostics": diagnostics}
