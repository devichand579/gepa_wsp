import numpy as np

def edit(w, f):
    # Extract features and labels from the full feedback pool
    X = f["features"].astype(np.float64)  # [2000, 1568]
    y = f["y"].astype(np.int64)           # [2000]
    N, D = X.shape
    C = 10

    # Current head
    W_cur = w["head.weight"].astype(np.float64)  # [10, 1568]
    b_cur = w["head.bias"].astype(np.float64)    # [10]

    # Current logits and metrics
    Z_cur = X @ W_cur.T + b_cur
    pred_cur = np.argmax(Z_cur, axis=1)
    acc_cur = np.mean(pred_cur == y)
    # Cross-entropy
    Z_cur_shift = Z_cur - Z_cur.max(axis=1, keepdims=True)
    exp_cur = np.exp(Z_cur_shift)
    prob_cur = exp_cur / exp_cur.sum(axis=1, keepdims=True)
    ce_cur = -np.mean(np.log(prob_cur[np.arange(N), y] + 1e-12))

    # Compute class means
    mu = np.zeros((C, D), dtype=np.float64)
    for c in range(C):
        mu[c] = X[y == c].mean(axis=0)

    # Compute within-class scatter matrix
    S_w = np.zeros((D, D), dtype=np.float64)
    for c in range(C):
        Xc = X[y == c]
        Xc_centered = Xc - mu[c]
        S_w += Xc_centered.T @ Xc_centered

    # Regularize S_w
    reg = 1e-3 * np.trace(S_w) / D
    S_w += reg * np.eye(D)

    # Invert S_w (use pinv for stability)
    S_w_inv = np.linalg.pinv(S_w)

    # LDA weights and bias
    W_lda = mu @ S_w_inv  # [10, 1568]
    b_lda = -0.5 * np.sum(mu * (mu @ S_w_inv), axis=1) + np.log(1.0 / C)

    # Compute LDA logits
    Z_lda = X @ W_lda.T + b_lda

    # Scale LDA logits to match current logit standard deviation
    std_cur = Z_cur.std()
    std_lda = Z_lda.std()
    if std_lda > 1e-12:
        alpha = std_cur / std_lda
    else:
        alpha = 1.0

    W_new = alpha * W_lda
    b_new = alpha * b_lda

    # Evaluate new head
    Z_new = X @ W_new.T + b_new
    pred_new = np.argmax(Z_new, axis=1)
    acc_new = np.mean(pred_new == y)
    Z_new_shift = Z_new - Z_new.max(axis=1, keepdims=True)
    exp_new = np.exp(Z_new_shift)
    prob_new = exp_new / exp_new.sum(axis=1, keepdims=True)
    ce_new = -np.mean(np.log(prob_new[np.arange(N), y] + 1e-12))

    # Choose the better head based on accuracy first, then CE
    if acc_new > acc_cur or (acc_new == acc_cur and ce_new < ce_cur):
        W_final = W_new
        b_final = b_new
    else:
        W_final = W_cur
        b_final = b_cur

    # Clip to allowed range
    W_final = np.clip(W_final, -1000.0, 1000.0).astype(np.float32)
    b_final = np.clip(b_final, -1000.0, 1000.0).astype(np.float32)

    w["head.weight"] = W_final
    w["head.bias"] = b_final

    diagnostics = {
        "acc_cur": float(acc_cur),
        "ce_cur": float(ce_cur),
        "acc_new": float(acc_new),
        "ce_new": float(ce_new),
        "acc_final": float(np.mean(np.argmax(X @ W_final.T + b_final, axis=1) == y)),
        "ce_final": float(-np.mean(np.log(
            np.exp((X @ W_final.T + b_final) - (X @ W_final.T + b_final).max(axis=1, keepdims=True)) /
            np.exp((X @ W_final.T + b_final) - (X @ W_final.T + b_final).max(axis=1, keepdims=True)).sum(axis=1, keepdims=True)
        )[np.arange(N), y] + 1e-12)),
        "scale_alpha": float(alpha),
        "max_abs_weight": float(np.max(np.abs(W_final))),
        "max_abs_bias": float(np.max(np.abs(b_final))),
    }

    return {"weights": w, "diagnostics": diagnostics}
