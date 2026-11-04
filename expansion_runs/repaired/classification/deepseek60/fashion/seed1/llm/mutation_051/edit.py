import numpy as np

def edit(w, f):
    features = f["features"]          # (2000, 1568)
    y = f["y"]                        # (2000,)
    N, D = features.shape
    K = 10

    def evaluate(W, b):
        logits = features @ W.T + b
        pred = np.argmax(logits, axis=1)
        return np.mean(pred == y)

    best_acc = -1.0
    best_W = None
    best_b = None

    # ----- 1. LDA with various regularisation strengths -----
    mu = np.zeros((K, D))
    for c in range(K):
        mu[c] = features[y == c].mean(axis=0)

    S = np.zeros((D, D))
    for c in range(K):
        Xc = features[y == c] - mu[c]
        S += Xc.T @ Xc
    S /= (N - K)

    for lam in [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        if lam == 0.0:
            try:
                W = np.linalg.solve(S, mu.T).T
            except np.linalg.LinAlgError:
                W = (np.linalg.pinv(S) @ mu.T).T
        else:
            S_reg = S + lam * np.eye(D)
            W = np.linalg.solve(S_reg, mu.T).T
        b = -0.5 * np.sum(mu * W, axis=1) + np.log(1.0 / K)
        acc = evaluate(W, b)
        if acc > best_acc:
            best_acc = acc
            best_W = W
            best_b = b

    # ----- 2. Ridge regression with one-hot and +1/-1 targets -----
    X_aug = np.hstack([features, np.ones((N, 1))])
    Y_onehot = np.eye(K)[y]
    Y_pm = -np.ones((N, K))
    Y_pm[np.arange(N), y] = 1.0

    for Y in [Y_onehot, Y_pm]:
        for lam in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]:
            XtX = X_aug.T @ X_aug
            XtY = X_aug.T @ Y
            W_aug = np.linalg.solve(XtX + lam * np.eye(D + 1), XtY)
            W = W_aug[:D, :].T
            b = W_aug[D, :]
            acc = evaluate(W, b)
            if acc > best_acc:
                best_acc = acc
                best_W = W
                best_b = b

    # ----- 3. Nearest centroid -----
    W = mu.copy()
    b = -0.5 * np.sum(mu * mu, axis=1) + np.log(1.0 / K)
    acc = evaluate(W, b)
    if acc > best_acc:
        best_acc = acc
        best_W = W
        best_b = b

    # ----- Scale to max absolute value 1000 -----
    max_abs = max(np.max(np.abs(best_W)), np.max(np.abs(best_b)))
    if max_abs > 0:
        scale = 1000.0 / max_abs
        best_W *= scale
        best_b *= scale

    # ----- Write back to weights -----
    w["head.weight"] = best_W.astype(np.float32)
    w["head.bias"] = best_b.astype(np.float32)

    diagnostics = {
        "full_pool_accuracy": float(best_acc),
        "max_abs_weight": float(np.max(np.abs(best_W))),
        "max_abs_bias": float(np.max(np.abs(best_b))),
    }
    return {"weights": w, "diagnostics": diagnostics}
