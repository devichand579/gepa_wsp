import numpy as np

def edit(w, f):
    X = f["features"].astype(np.float64)   # (2000, 128)
    y = f["y"].astype(np.int64)            # (2000,)
    N, D = X.shape
    C = 10

    # Standardize features
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1e-6
    X_std = (X - mean) / std

    # ---------- LDA ----------
    mu = np.zeros((C, D))
    for c in range(C):
        mu[c] = X_std[y == c].mean(axis=0)
    mu0 = X_std.mean(axis=0)

    S_w = np.zeros((D, D))
    for c in range(C):
        Xc = X_std[y == c] - mu[c]
        S_w += Xc.T @ Xc

    reg = 1e-3 * np.trace(S_w) / D
    S_w_reg = S_w + reg * np.eye(D)
    W_lda = np.linalg.solve(S_w_reg, (mu - mu0).T).T   # (C, D)
    b_lda = -0.5 * np.sum(mu * W_lda, axis=1)          # (C,)

    # ---------- Ridge regression ----------
    X_aug = np.hstack([X_std, np.ones((N, 1))])
    Y = np.zeros((N, C))
    Y[np.arange(N), y] = 1.0

    reg_mat = np.eye(D + 1) * 1.0
    reg_mat[D, D] = 0.0  # do not regularize bias
    W_ridge = np.linalg.solve(X_aug.T @ X_aug + reg_mat, X_aug.T @ Y)
    W_ridge_w = W_ridge[:D].T   # (C, D)
    b_ridge = W_ridge[D]        # (C,)

    # ---------- Quick validation on paired batch ----------
    batch_ids = f["batch.ids"]
    X_batch = f["features"][batch_ids].astype(np.float64)
    X_batch_std = (X_batch - mean) / std
    y_batch = f["batch.y"]

    def acc(W, b):
        logits = X_batch_std @ W.T + b
        pred = np.argmax(logits, axis=1)
        return np.mean(pred == y_batch)

    acc_lda = acc(W_lda, b_lda)
    acc_ridge = acc(W_ridge_w, b_ridge)

    if acc_lda >= acc_ridge:
        W_final, b_final, method = W_lda, b_lda, "LDA"
    else:
        W_final, b_final, method = W_ridge_w, b_ridge, "Ridge"

    # Convert back to original feature space
    W_orig = W_final / std
    b_orig = b_final - W_final @ (mean / std)

    # Clip to safe range
    W_orig = np.clip(W_orig, -1000.0, 1000.0)
    b_orig = np.clip(b_orig, -1000.0, 1000.0)

    w["fc2.weight"] = W_orig.astype(np.float32)
    w["fc2.bias"] = b_orig.astype(np.float32)

    diagnostics = {
        "method": method,
        "acc_lda": float(acc_lda),
        "acc_ridge": float(acc_ridge),
        "fc2_weight_norm": float(np.linalg.norm(W_orig)),
        "fc2_bias_norm": float(np.linalg.norm(b_orig)),
    }
    return {"weights": w, "diagnostics": diagnostics}
