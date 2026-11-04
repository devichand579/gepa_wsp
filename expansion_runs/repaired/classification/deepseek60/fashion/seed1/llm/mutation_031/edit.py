import numpy as np

def edit(w, f):
    X = np.asarray(f["features"], dtype=np.float64)      # (2000,1568)
    y = np.asarray(f["y"], dtype=np.int64)               # (2000,)
    Xb = np.asarray(f["layer.head.input0"], dtype=np.float64)  # (128,1568) paired batch feats
    yb = np.asarray(f["batch.y"], dtype=np.int64)        # (128,)

    N, D = X.shape
    C = 10
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # center features (do not regularize bias)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    Yc = Y - Y.mean(axis=0, keepdims=True)

    # precompute Gram
    G = Xc.T @ Xc                                  # (D,D)
    XtY = Xc.T @ Yc                                # (D,C)

    def fit(lmb):
        A = G + lmb * np.eye(D)
        W = np.linalg.solve(A, XtY).T              # (C,D)
        b = Y.mean(axis=0) - W @ mu.ravel()
        return W, b

    def acc(W, b, Xe, ye):
        logits = Xe @ W.T + b
        return (logits.argmax(axis=1) == ye).mean()

    # current head accuracy on pool and batch (sanity)
    Wc = np.asarray(w["head.weight"], dtype=np.float64)
    bc = np.asarray(w["head.bias"], dtype=np.float64)
    cur_pool = acc(Wc, bc, X, y)
    cur_batch = acc(Wc, bc, Xb, yb)

    # cross-validation over lambda grid
    rng = np.random.default_rng(0)
    idx = rng.permutation(N)
    K = 5
    folds = np.array_split(idx, K)
    lambdas = [1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]

    best_l, best_cv = None, -1.0
    cv_scores = {}
    for lmb in lambdas:
        correct = 0
        for k in range(K):
            va = folds[k]
            tr = np.concatenate([folds[j] for j in range(K) if j != k])
            Xtr, Ytr = X[tr], Y[tr]
            m = Xtr.mean(axis=0, keepdims=True)
            Xtrc = Xtr - m
            Ytrc = Ytr - Ytr.mean(axis=0, keepdims=True)
            A = Xtrc.T @ Xtrc + lmb * np.eye(D)
            Wk = np.linalg.solve(A, Xtrc.T @ Ytrc).T
            bk = Ytr.mean(axis=0) - Wk @ m.ravel()
            lg = X[va] @ Wk.T + bk
            correct += (lg.argmax(axis=1) == y[va]).sum()
        cv = correct / N
        cv_scores[lmb] = cv
        if cv > best_cv:
            best_cv, best_l = cv, lmb

    # also evaluate each lambda's full fit on the paired batch
    batch_scores = {}
    for lmb in lambdas:
        W, b = fit(lmb)
        batch_scores[lmb] = acc(W, b, Xb, yb)

    # choose lambda: prefer high CV, tie-break by batch accuracy
    # restrict to lambdas whose batch accuracy >= current batch accuracy
    cand = [l for l in lambdas if batch_scores[l] >= cur_batch - 1e-9]
    if not cand:
        cand = lambdas
    # among candidates pick best CV, then largest lambda (more regularized)
    cand.sort(key=lambda l: (cv_scores[l], l), reverse=True)
    chosen = cand[0]

    W, b = fit(chosen)

    # safety: never let pool accuracy drop below current by more than a hair
    new_pool = acc(W, b, X, y)
    if new_pool < cur_pool - 1e-9 or batch_scores[chosen] < cur_batch - 1e-9:
        # fall back to current head
        W = Wc.copy()
        b = bc.copy()
        chosen = None

    # clip to allowed magnitude
    W = np.clip(W, -1000.0, 1000.0).astype(np.float32)
    b = np.clip(b, -1000.0, 1000.0).astype(np.float32)

    w["head.weight"] = W
    w["head.bias"] = b

    diag = {
        "cur_pool_acc": float(cur_pool),
        "cur_batch_acc": float(cur_batch),
        "cv_scores": {str(k): float(v) for k, v in cv_scores.items()},
        "batch_scores": {str(k): float(v) for k, v in batch_scores.items()},
        "chosen_lambda": (None if chosen is None else float(chosen)),
        "new_pool_acc": float(new_pool),
        "new_batch_acc": float(acc(W, b, Xb, yb)),
        "head_w_max": float(np.abs(W).max()),
        "head_b_max": float(np.abs(b).max()),
    }
    return {"weights": w, "diagnostics": diag}
