import numpy as np

def edit(w, f):
    # Save old head for comparison
    old_W = w["head.weight"].copy()
    old_b = w["head.bias"].copy()

    X = f["features"].astype(np.float64)  # N x D
    y = f["y"].astype(np.int64)
    N, D = X.shape
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Class means and priors
    M = np.zeros((C, D), dtype=np.float64)
    Nc = np.zeros(C, dtype=np.float64)
    for c in range(C):
        idx = (y == c)
        Nc[c] = idx.sum()
        if Nc[c] > 0:
            M[c] = X[idx].mean(axis=0)
    priors = Nc / N

    # --- LDA ---
    XtX = X.T @ X
    S = XtX - M.T @ (Nc[:, None] * M)
    dof = N - C
    S /= dof
    trace_S = np.trace(S)
    lam_lda = 1e-3 * trace_S / D
    S_reg = S + lam_lda * np.eye(D)
    W_T_lda = np.linalg.solve(S_reg, M.T)  # D x C
    W_lda = W_T_lda.T  # C x D
    b_lda = -0.5 * np.sum(M * W_T_lda.T, axis=1) + np.log(priors)

    # --- Ridge regression ---
    X_mean = X.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    Xc = X - X_mean
    Yc = Y - Y_mean
    XtX_c = Xc.T @ Xc
    trace_XtX = np.trace(XtX_c)
    lam_ridge = 1e-3 * trace_XtX / D
    W_ridge = np.linalg.solve(XtX_c + lam_ridge * np.eye(D), Xc.T @ Yc)  # D x C
    b_ridge = Y_mean - X_mean @ W_ridge
    W_ridge = W_ridge.T  # C x D

    # Evaluate on paired batch
    Xb = f["layer.head.input0"].astype(np.float64)  # 128 x D
    yb = f["batch.y"].astype(np.int64)

    def eval_head(W, b):
        logits = Xb @ W.T + b
        pred = np.argmax(logits, axis=1)
        return (pred == yb).mean()

    acc_old = eval_head(old_W.astype(np.float64), old_b.astype(np.float64))
    acc_lda = eval_head(W_lda, b_lda)
    acc_ridge = eval_head(W_ridge, b_ridge)

    # Choose better on the paired batch
    if acc_lda >= acc_ridge:
        W_new = W_lda
        b_new = b_lda
        chosen = "lda"
    else:
        W_new = W_ridge
        b_new = b_ridge
        chosen = "ridge"

    # If old head is better on this batch, keep it
    if acc_old > max(acc_lda, acc_ridge):
        W_new = old_W.astype(np.float64)
        b_new = old_b.astype(np.float64)
        chosen = "old"

    # Clip to safe range
    W_new = np.clip(W_new, -1000, 1000)
    b_new = np.clip(b_new, -1000, 1000)

    w["head.weight"] = W_new.astype(np.float32)
    w["head.bias"] = b_new.astype(np.float32)

    # Diagnostics
    logits_all = X @ W_new.T + b_new
    pred_all = np.argmax(logits_all, axis=1)
    acc_all = (pred_all == y).mean()

    diag = {
        "chosen": chosen,
        "acc_old_batch": float(acc_old),
        "acc_lda_batch": float(acc_lda),
        "acc_ridge_batch": float(acc_ridge),
        "acc_new_batch": float(eval_head(W_new, b_new)),
        "acc_new_all": float(acc_all),
        "max_abs_weight": float(np.max(np.abs(W_new))),
        "max_abs_bias": float(np.max(np.abs(b_new))),
    }

    return {"weights": w, "diagnostics": diag}
