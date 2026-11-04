def edit(w, f):
    import numpy as np

    # ---------- load feedback ----------
    F = f["features"].astype(np.float64)   # (2000, 128)
    y = f["y"].astype(np.int64)            # (2000,)
    n, d = F.shape
    classes = np.unique(y)
    C = len(classes)

    # ---------- standardize features ----------
    mu = F.mean(axis=0)
    sd = F.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Fs = (F - mu) / sd

    # ---------- helpers ----------
    def fit_lda(X, yy, lam):
        mu_c = np.zeros((C, d))
        for i, c in enumerate(classes):
            mu_c[i] = X[yy == c].mean(axis=0)
        Sw = np.zeros((d, d))
        for i, c in enumerate(classes):
            Xc = X[yy == c] - mu_c[i]
            Sw += Xc.T @ Xc
        Sw += lam * np.eye(d)
        Sw_inv_mu = np.linalg.solve(Sw, mu_c.T)  # (d, C)
        W = Sw_inv_mu.T                           # (C, d)
        b = -0.5 * np.sum(mu_c.T * Sw_inv_mu, axis=0)
        priors = np.array([np.mean(yy == c) for c in classes])
        b += np.log(priors + 1e-12)
        return W, b

    def fit_ridge(X, yy, lam):
        Y = np.zeros((n, C))
        for i, c in enumerate(classes):
            Y[yy == c, i] = 1.0
        Xa = np.hstack([X, np.ones((n, 1))])
        da = d + 1
        A = Xa.T @ Xa + lam * np.eye(da)
        A[-1, -1] -= lam
        Wa = np.linalg.solve(A, Xa.T @ Y)
        W = Wa[:-1].T
        b = Wa[-1]
        return W, b

    # ---------- stratified 5-fold CV ----------
    rng = np.random.RandomState(42)
    n_folds = 5
    fold_parts = []
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        fold_parts.append(np.array_split(idx, n_folds))

    lambdas = np.concatenate([
        np.logspace(-4, 2, 25),
        np.array([0.0])
    ])
    best_score = -1.0
    best_cfg = None

    for method in ["lda", "ridge"]:
        for lam in lambdas:
            accs = []
            for k in range(n_folds):
                val_idx = np.concatenate([fold_parts[c][k] for c in range(C)])
                tr_idx = np.setdiff1d(np.arange(n), val_idx)
                Xtr, ytr = Fs[tr_idx], y[tr_idx]
                Xva, yva = Fs[val_idx], y[val_idx]

                if method == "lda":
                    W, b = fit_lda(Xtr, ytr, lam)
                else:
                    W, b = fit_ridge(Xtr, ytr, lam)

                logits = Xva @ W.T + b
                pred = np.argmax(logits, axis=1)
                accs.append(np.mean(pred == yva))
            score = float(np.mean(accs))
            if score > best_score:
                best_score = score
                best_cfg = (method, lam)

    method, lam = best_cfg

    # ---------- refit on full pool ----------
    if method == "lda":
        W_std, b_std = fit_lda(Fs, y, lam)
    else:
        W_std, b_std = fit_ridge(Fs, y, lam)

    # Convert back to raw-feature weights:
    # logits = (Fs) @ W_std.T + b_std
    #        = ((F - mu) / sd) @ W_std.T + b_std
    #        = F @ (W_std / sd).T + (b_std - (W_std / sd) @ mu)
    W_raw = W_std / sd[None, :]
    b_raw = b_std - W_raw @ mu

    # ---------- write into w ----------
    w["fc2.weight"] = W_raw.astype(np.float32)
    w["fc2.bias"] = b_raw.astype(np.float32)

    # keep fc1 unchanged
    w["fc1.weight"] = w["fc1.weight"].astype(np.float32)
    w["fc1.bias"] = w["fc1.bias"].astype(np.float32)

    # ---------- diagnostics ----------
    logits_full = F @ W_raw.T + b_raw
    pred_full = np.argmax(logits_full, axis=1)
    train_acc = float(np.mean(pred_full == y))
    # cross-entropy on full pool
    logits_full_shift = logits_full - logits_full.max(axis=1, keepdims=True)
    probs_full = np.exp(logits_full_shift)
    probs_full /= probs_full.sum(axis=1, keepdims=True)
    ce = float(-np.mean(np.log(probs_full[np.arange(n), y] + 1e-12)))

    diag = {
        "method": method,
        "lambda": float(lam),
        "cv_score": float(best_score),
        "train_acc": train_acc,
        "train_ce": ce,
        "fc2_weight_norm": float(np.linalg.norm(W_raw)),
        "fc2_bias_norm": float(np.linalg.norm(b_raw)),
    }
    return {"weights": w, "diagnostics": diag}
