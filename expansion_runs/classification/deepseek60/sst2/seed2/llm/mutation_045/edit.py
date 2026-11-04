import numpy as np

def edit(w, f):
    X = np.asarray(f["features"], dtype=np.float64)      # [4000,32] pooled LN features
    y = np.asarray(f["y"], dtype=np.int64)               # [4000]
    N, D = X.shape
    Y = np.zeros((N, 2), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # ---- standardize features (helps conditioning; head can absorb scale) ----
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-6
    Xs = (X - mu) / sd

    # ---- ridge normal equations:  W = (Xs^T Xs + lam I)^-1 Xs^T Y ----
    G = Xs.T @ Xs
    XtY = Xs.T @ Y
    I = np.eye(D)

    def ridge(lam):
        return np.linalg.solve(G + lam * I, XtY)          # [D,2]

    def acc_of(W):
        # bias = mean over classes of residual so the two rows split at 0.5
        logits = Xs @ W
        b = np.array([1.0, 0.0]) - logits.mean(axis=0)     # center each column
        pred = np.argmax(logits + b, axis=1)
        return (pred == y).mean(), b

    # ---- cross-validated ridge coefficient ----
    rng = np.random.default_rng(0)
    idx = rng.permutation(N)
    folds = np.array_split(idx, 5)
    lam_grid = [1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    best_lam, best_cv = None, -1.0
    for lam in lam_grid:
        scores = []
        for k in range(5):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(5) if j != k])
            Gtr = Xs[tr].T @ Xs[tr]
            Wtr = np.linalg.solve(Gtr + lam * I, Xs[tr].T @ Y[tr])
            lg = Xs[te] @ Wtr
            b = np.array([1.0, 0.0]) - lg.mean(axis=0)
            pr = np.argmax(lg + b, axis=1)
            scores.append((pr == y[te]).mean())
        cv = float(np.mean(scores))
        if cv > best_cv:
            best_cv, best_lam = cv, lam

    W_ridge = ridge(best_lam)
    acc_ridge, b_ridge = acc_of(W_ridge)

    # ---- LDA direction (class moments + pooled covariance) ----
    m0 = Xs[y == 0].mean(axis=0)
    m1 = Xs[y == 1].mean(axis=0)
    Xc = Xs.copy()
    Xc[y == 0] -= m0
    Xc[y == 1] -= m1
    Sw = (Xc.T @ Xc) / N + 1e-3 * I
    d = np.linalg.solve(Sw, m1 - m0)                       # [D]
    W_lda = np.stack([-d, d], axis=1)                      # row0 vs row1
    acc_lda, b_lda = acc_of(W_lda)

    # ---- pick best of ridge / LDA / blend ----
    cands = [("ridge", W_ridge, b_ridge, acc_ridge),
             ("lda", W_lda, b_lda, acc_lda)]
    # blend (normalize each to unit scale first)
    def unit(W):
        n = np.linalg.norm(W, axis=0, keepdims=True) + 1e-9
        return W / n
    Wb = unit(W_ridge) + unit(W_lda)
    acc_b, b_b = acc_of(Wb)
    cands.append(("blend", Wb, b_b, acc_b))

    name, W, b, tr_acc = max(cands, key=lambda t: t[3])

    # ---- write head back (undo standardization: feat_s = (feat-mu)/sd) ----
    # logits = feat_s @ W + b = (feat - mu)/sd @ W + b
    Wn = (W / sd[:, None]).T                                # [2,32]
    bn = b - (mu / sd) @ W                                  # [2]
    w["head.weight"] = Wn.astype(np.float32)
    w["head.bias"] = bn.astype(np.float32)

    # sanity clip
    for k in ("head.weight", "head.bias"):
        arr = np.asarray(w[k], dtype=np.float32)
        arr = np.clip(arr, -1000.0, 1000.0)
        w[k] = arr.astype(np.float32)

    diag = {
        "chosen": name,
        "cv_best_lambda": float(best_lam),
        "cv_acc_ridge": float(best_cv),
        "train_acc_ridge": float(acc_ridge),
        "train_acc_lda": float(acc_lda),
        "train_acc_blend": float(acc_b),
        "head_w_norm": float(np.linalg.norm(w["head.weight"])),
        "head_bias": [float(x) for x in np.asarray(w["head.bias"])],
    }
    return {"weights": w, "diagnostics": diag}
