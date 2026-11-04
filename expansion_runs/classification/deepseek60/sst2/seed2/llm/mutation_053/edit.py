import numpy as np

def edit(w, f):
    X = np.asarray(f["features"], dtype=np.float64)   # [N,32] pooled features
    y = np.asarray(f["y"], dtype=np.int64)            # [N]
    N, D = X.shape
    t = 2.0 * y - 1.0                                 # targets in {-1,+1}

    # ---- standardize features (folded back into head later) ----
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xs = (X - mu) / sd
    Xa = np.concatenate([Xs, np.ones((N, 1))], axis=1)   # [N, D+1]

    def ridge_fit(Xa_, t_, lam):
        A = Xa_.T @ Xa_ + lam * np.eye(D + 1)
        A[-1, -1] -= lam                              # do not regularize intercept
        return np.linalg.solve(A, Xa_.T @ t_)         # [D+1]

    # ---- 5-fold CV to choose ridge strength ----
    rng = np.random.default_rng(0)
    idx = rng.permutation(N)
    folds = np.array_split(idx, 5)
    lams = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
    best_lam, best_acc = 1.0, -1.0
    cv_table = {}
    for lam in lams:
        accs = []
        for i in range(5):
            val = folds[i]
            tr = np.concatenate([folds[j] for j in range(5) if j != i])
            ws = ridge_fit(Xa[tr], t[tr], lam)
            pred = np.sign(Xa[val] @ ws)
            accs.append(float(np.mean(pred == t[val])))
        a = float(np.mean(accs))
        cv_table[lam] = a
        if a > best_acc:
            best_acc, best_lam = a, lam

    # ---- refit on the full pool with the selected lambda ----
    ws = ridge_fit(Xa, t, best_lam)                   # [D+1]
    g = ws[:D] / sd                                   # direction in raw-feature space
    c = ws[D] - (mu / sd) @ ws[:D]                    # intercept in raw-feature space

    # write into head as a rank-1 (symmetric) readout: logit1 - logit0 = X@g + c
    W = np.zeros((2, D), dtype=np.float64)
    W[1] = g / 2.0
    W[0] = -g / 2.0
    b = np.array([-c / 2.0, c / 2.0], dtype=np.float64)

    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

    # ---- diagnostics ----
    in_sample = float(np.mean(np.sign(Xa @ ws) == t))
    diag = {
        "cv_accuracy_by_lambda": {str(k): round(v, 5) for k, v in cv_table.items()},
        "selected_lambda": best_lam,
        "cv_accuracy_selected": round(best_acc, 5),
        "in_sample_accuracy": round(in_sample, 5),
        "direction_norm": float(np.linalg.norm(g)),
        "intercept": float(c),
        "head_weight_row0_norm": float(np.linalg.norm(W[0])),
    }
    return {"weights": w, "diagnostics": diag}
