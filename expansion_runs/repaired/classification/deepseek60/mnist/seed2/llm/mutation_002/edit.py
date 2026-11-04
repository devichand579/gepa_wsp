import numpy as np

def edit(w, f):
    H = f["features"].astype(np.float64)   # (2000, 128)
    y = f["y"].astype(np.int64)            # (2000,)
    N, D = H.shape
    C = 10

    # Current fc2 readout accuracy on the full feedback pool
    W_cur = w["fc2.weight"].astype(np.float64)
    b_cur = w["fc2.bias"].astype(np.float64)
    logits_cur = H @ W_cur.T + b_cur
    acc_cur = np.mean(np.argmax(logits_cur, axis=1) == y)

    # ---------- closed-form candidates ----------
    def fit_ridge(H_tr, y_tr, alpha):
        Ha = np.hstack([H_tr, np.ones((len(H_tr), 1))])
        Y = np.zeros((len(H_tr), C))
        Y[np.arange(len(H_tr)), y_tr] = 1.0
        A = Ha.T @ Ha + alpha * np.eye(D + 1)
        B = Ha.T @ Y
        W = np.linalg.solve(A, B)          # (D+1, C)
        return W[:D, :].T, W[D, :]         # (C, D), (C,)

    def fit_lda(H_tr, y_tr, alpha):
        means = np.zeros((C, D))
        priors = np.zeros(C)
        for c in range(C):
            idx = (y_tr == c)
            if idx.sum() == 0:
                continue
            means[c] = H_tr[idx].mean(axis=0)
            priors[c] = idx.mean()
        Hc = H_tr.copy()
        for c in range(C):
            idx = (y_tr == c)
            Hc[idx] -= means[c]
        cov = (Hc.T @ Hc) / max(1, len(y_tr) - C)
        cov_reg = cov + alpha * np.eye(D)
        inv_cov = np.linalg.inv(cov_reg)
        weight = (inv_cov @ means.T).T
        bias = -0.5 * np.sum(means * (inv_cov @ means.T).T, axis=1) + np.log(priors + 1e-12)
        return weight, bias

    def acc(H, y, W, b):
        return np.mean(np.argmax(H @ W.T + b, axis=1) == y)

    # ---------- cross-validation for model selection ----------
    rng = np.random.RandomState(42)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_folds = 5
    folds = np.array_split(idx, n_folds)

    def cv_score(method, alpha):
        scores = []
        for i in range(n_folds):
            val = folds[i]
            tr = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            if method == "ridge":
                W, b = fit_ridge(H[tr], y[tr], alpha)
            else:
                W, b = fit_lda(H[tr], y[tr], alpha)
            scores.append(acc(H[val], y[val], W, b))
        return float(np.mean(scores))

    alphas_ridge = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    alphas_lda   = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

    best = {"method": None, "alpha": None, "cv": -1.0}
    for a in alphas_ridge:
        s = cv_score("ridge", a)
        if s > best["cv"]:
            best = {"method": "ridge", "alpha": a, "cv": s}
    for a in alphas_lda:
        s = cv_score("lda", a)
        if s > best["cv"]:
            best = {"method": "lda", "alpha": a, "cv": s}

    # ---------- final fit on all feedback data ----------
    if best["method"] == "ridge":
        W_new, b_new = fit_ridge(H, y, best["alpha"])
    else:
        W_new, b_new = fit_lda(H, y, best["alpha"])

    acc_new = acc(H, y, W_new, b_new)

    # Use the new readout only if it is not worse than the current one.
    # (Cross-validation already guards against overfitting, but this is a final safety check.)
    if acc_new >= acc_cur - 1e-6:
        w["fc2.weight"] = W_new.astype(np.float32)
        w["fc2.bias"]   = b_new.astype(np.float32)
        chosen = "refit"
    else:
        chosen = "current"

    # Clip to safe finite float32 range
    for k in ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]:
        w[k] = np.clip(w[k].astype(np.float32), -1000.0, 1000.0)

    diagnostics = {
        "chosen": chosen,
        "method": best["method"],
        "alpha": best["alpha"],
        "cv_score": best["cv"],
        "acc_current_full": float(acc_cur),
        "acc_new_full": float(acc_new),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
    }
    return {"weights": w, "diagnostics": diagnostics}
