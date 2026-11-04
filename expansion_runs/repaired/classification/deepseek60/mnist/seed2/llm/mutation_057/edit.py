import numpy as np

def edit(w, f):
    X = f["features"].astype(np.float64)          # [2000, 128]
    y = f["y"].astype(np.int64)                   # [2000]
    N, D = X.shape
    C = 10

    # Current readout for baseline
    W_cur = w["fc2.weight"].astype(np.float64).copy()   # [10, 128]
    b_cur = w["fc2.bias"].astype(np.float64).copy()     # [10]

    def logits_from(W, b):
        return X @ W.T + b

    def accuracy(logits):
        return np.mean(np.argmax(logits, axis=1) == y)

    def cross_entropy(logits):
        # stable softmax CE
        m = np.max(logits, axis=1, keepdims=True)
        z = logits - m
        logsumexp = np.log(np.sum(np.exp(z), axis=1, keepdims=True))
        logp = z - logsumexp
        return -np.mean(logp[np.arange(N), y])

    cur_logits = logits_from(W_cur, b_cur)
    cur_acc = accuracy(cur_logits)
    cur_ce = cross_entropy(cur_logits)

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Center features and targets for ridge with intercept
    X_mean = X.mean(axis=0, keepdims=True)
    Y_mean = Y.mean(axis=0, keepdims=True)
    Xc = X - X_mean
    Yc = Y - Y_mean

    best = {
        "name": "current",
        "W": W_cur,
        "b": b_cur,
        "acc": cur_acc,
        "ce": cur_ce,
    }

    # Try several ridge penalties
    alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    for alpha in alphas:
        # Solve (Xc^T Xc + alpha I) W = Xc^T Yc
        A = Xc.T @ Xc + alpha * np.eye(D)
        B = Xc.T @ Yc
        W = np.linalg.solve(A, B).T          # [C, D]
        b = (Y_mean - X_mean @ W.T).ravel()  # [C]
        logits = logits_from(W, b)
        acc = accuracy(logits)
        ce = cross_entropy(logits)
        if acc > best["acc"] or (acc == best["acc"] and ce < best["ce"]):
            best = {
                "name": f"ridge_alpha_{alpha}",
                "W": W,
                "b": b,
                "acc": acc,
                "ce": ce,
            }

    # Also try LDA with shrinkage
    # Compute class means and within-class covariance
    means = np.zeros((C, D), dtype=np.float64)
    for c in range(C):
        idx = (y == c)
        if np.any(idx):
            means[c] = X[idx].mean(axis=0)
    overall_mean = X.mean(axis=0)
    Sw = np.zeros((D, D), dtype=np.float64)
    for c in range(C):
        idx = (y == c)
        if np.any(idx):
            diff = X[idx] - means[c]
            Sw += diff.T @ diff
    Sw /= (N - C)

    for shrink in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
        Sw_reg = Sw + shrink * np.eye(D) * np.trace(Sw) / D
        try:
            Sw_inv = np.linalg.inv(Sw_reg)
        except np.linalg.LinAlgError:
            continue
        W_lda = np.zeros((C, D), dtype=np.float64)
        b_lda = np.zeros(C, dtype=np.float64)
        for c in range(C):
            W_lda[c] = Sw_inv @ means[c]
            b_lda[c] = -0.5 * means[c] @ Sw_inv @ means[c]
        # Add log priors (uniform here, but keep general)
        priors = np.array([np.mean(y == c) for c in range(C)])
        b_lda += np.log(priors + 1e-12)
        logits = logits_from(W_lda, b_lda)
        acc = accuracy(logits)
        ce = cross_entropy(logits)
        if acc > best["acc"] or (acc == best["acc"] and ce < best["ce"]):
            best = {
                "name": f"lda_shrink_{shrink}",
                "W": W_lda,
                "b": b_lda,
                "acc": acc,
                "ce": ce,
            }

    # Install best candidate
    W_new = best["W"].astype(np.float32)
    b_new = best["b"].astype(np.float32)

    # Clamp to safe range
    W_new = np.clip(W_new, -1000.0, 1000.0)
    b_new = np.clip(b_new, -1000.0, 1000.0)

    w["fc2.weight"] = W_new
    w["fc2.bias"] = b_new

    diagnostics = {
        "chosen": best["name"],
        "train_acc": float(best["acc"]),
        "train_ce": float(best["ce"]),
        "current_acc": float(cur_acc),
        "current_ce": float(cur_ce),
        "fc2_weight_norm": float(np.linalg.norm(W_new)),
        "fc2_bias_norm": float(np.linalg.norm(b_new)),
    }
    return {"weights": w, "diagnostics": diagnostics}
