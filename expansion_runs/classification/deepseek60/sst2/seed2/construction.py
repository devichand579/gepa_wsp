import numpy as np

def edit(w, f):
    X = f["features"].astype(np.float64)  # (4000, 32)
    y = f["y"].astype(np.int64)           # (4000,)
    N, D = X.shape

    # One-hot targets
    Y = np.zeros((N, 2), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Ridge regression with small regularization
    X_aug = np.hstack([X, np.ones((N, 1))])
    lam = 1e-3
    A = X_aug.T @ X_aug + lam * np.eye(D + 1)
    B = X_aug.T @ Y
    W_aug = np.linalg.solve(A, B)
    W_ridge = W_aug[:D, :].T          # (2, D)
    b_ridge = W_aug[D, :]             # (2,)

    # LDA
    mu0 = X[y == 0].mean(axis=0)
    mu1 = X[y == 1].mean(axis=0)
    X0 = X[y == 0] - mu0
    X1 = X[y == 1] - mu1
    S0 = X0.T @ X0
    S1 = X1.T @ X1
    n0 = (y == 0).sum()
    n1 = (y == 1).sum()
    Sigma = (S0 + S1) / (n0 + n1 - 2)
    Sigma += 1e-6 * np.eye(D)
    invSigma = np.linalg.inv(Sigma)
    w_lda = invSigma @ (mu1 - mu0)    # (D,)
    b_lda = -0.5 * (mu1 + mu0) @ w_lda + np.log(n1 / n0)
    W_lda = np.vstack([-w_lda, w_lda])  # (2, D)
    b_lda = np.array([-b_lda, b_lda])   # (2,)

    # Evaluate training accuracy
    def accuracy(W, b):
        logits = X @ W.T + b
        pred = np.argmax(logits, axis=1)
        return (pred == y).mean()

    acc_ridge = accuracy(W_ridge, b_ridge)
    acc_lda = accuracy(W_lda, b_lda)

    # Choose the better one
    if acc_ridge >= acc_lda:
        W_final = W_ridge
        b_final = b_ridge
        method = "ridge"
        acc_final = acc_ridge
    else:
        W_final = W_lda
        b_final = b_lda
        method = "lda"
        acc_final = acc_lda

    # Update weights
    w["head.weight"] = W_final.astype(np.float32)
    w["head.bias"] = b_final.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "method": method,
        "acc_ridge": float(acc_ridge),
        "acc_lda": float(acc_lda),
        "acc_final": float(acc_final),
        "acc_original": float((np.argmax(f["logits"], axis=1) == y).mean()),
        "W_norm": float(np.linalg.norm(W_final)),
        "b_norm": float(np.linalg.norm(b_final))
    }

    return {"weights": w, "diagnostics": diagnostics}
