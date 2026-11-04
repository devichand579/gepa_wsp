import numpy as np

def edit(w, f):
    X = f["features"].astype(np.float64)          # [N,32] pooled features (current)
    y = f["y"].astype(np.int64)                   # [N]
    N, D = X.shape

    # ---- center features for numerical conditioning ----
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu

    # ---- 5-fold CV ridge on the pooled features ----
    rng = np.random.RandomState(0)
    idx = rng.permutation(N)
    folds = np.array_split(idx, 5)
    lambdas = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
    best_lam, best_acc = None, -1.0
    cv_scores = {}
    for lam in lambdas:
        accs = []
        for k in range(5):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(5) if j != k])
            Xtr, Xte = Xc[tr], Xc[te]
            ytr, yte = y[tr], y[te]
            # targets: +1 for class1, -1 for class0
            t = np.where(ytr == 1, 1.0, -1.0)
            A = Xtr.T @ Xtr + lam * np.eye(D)
            b = Xtr.T @ t
            wv = np.linalg.solve(A, b)
            pred = (Xte @ wv > 0).astype(np.int64)
            accs.append((pred == yte).mean())
        m = float(np.mean(accs))
        cv_scores[str(lam)] = m
        if m > best_acc:
            best_acc, best_lam = m, lam

    # ---- final fit on all data with selected lambda ----
    t = np.where(y == 1, 1.0, -1.0)
    A = Xc.T @ Xc + best_lam * np.eye(D)
    b = Xc.T @ t
    wv = np.linalg.solve(A, b)                    # [32]
    # intercept in centered space
    intercept = -float(mu @ wv)

    # ---- write into head: logit1 - logit0 = wv @ pooled + intercept ----
    # head.weight[0] = -wv/2, head.weight[1] = +wv/2, bias = ±intercept/2
    wv = wv.astype(np.float32)
    head_w = np.zeros((2, D), dtype=np.float32)
    head_w[0] = -0.5 * wv
    head_w[1] =  0.5 * wv
    head_b = np.array([-0.5 * intercept, 0.5 * intercept], dtype=np.float32)

    w["head.weight"] = head_w
    w["head.bias"] = head_b

    # ---- keep embed well-conditioned: unit-norm rows (LayerNorm invariant to scale) ----
    E = w["embed.weight"].astype(np.float64)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    w["embed.weight"] = (E / norms).astype(np.float32)

    # ---- diagnostics ----
    logits = X @ (head_w[1] - head_w[0]) + (head_b[1] - head_b[0])
    pred = (logits > 0).astype(np.int64)
    in_sample_acc = float((pred == y).mean())
    diag = {
        "selected_lambda": best_lam,
        "cv_accuracy_selected": best_acc,
        "cv_scores": cv_scores,
        "in_sample_accuracy": in_sample_acc,
        "head_dir_norm": float(np.linalg.norm(head_w[1] - head_w[0])),
        "intercept": float(head_b[1] - head_b[0]),
        "embed_row_norm_mean": float(np.linalg.norm(w["embed.weight"], axis=1).mean()),
    }
    return {"weights": w, "diagnostics": diag}
