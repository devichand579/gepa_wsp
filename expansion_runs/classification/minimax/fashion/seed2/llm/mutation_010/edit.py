def edit(w, f):
    import numpy as np

    # ---------------- data ----------------
    F = np.asarray(f["features"], dtype=np.float64)      # (N, 1568) current hidden features
    y = np.asarray(f["y"]).astype(np.int64)              # (N,)
    N, P = F.shape
    C = 10
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    mu = F.mean(axis=0)                                  # feature means
    Fc = F - mu
    ybar = Y.mean(axis=0)                                # class priors (uniform here)
    Yc = Y - ybar

    # ---------------- feature selection (F-statistic: between/within class variance) ----
    classes = np.unique(y)
    M = np.zeros((len(classes), P))
    SW = np.zeros(P)
    for i, c in enumerate(classes):
        Xi = Fc[y == c]
        M[i] = Xi.mean(axis=0)
        SW += Xi.var(axis=0)
    SW /= len(classes)
    SB = M.var(axis=0)
    score = SB / (SW + 1e-9)
    score[~np.isfinite(score)] = 0.0
    order = np.argsort(-score)

    # ---------------- closed-form ridge + LOOCV model selection ----------------
    # For centered data: G = Xc^T Xc = V diag(lam) V^T
    #   ridge coef   B = V diag(1/(lam+a)) V^T Xc^T Yc
    #   hat diag     h_ii = sum_k (x_i^T v_k)^2 / (lam_k + a)
    #   LOOCV resid e_i/(1-h_ii)   (closed form, no training loop)
    K_grid = [100, 200, 400, 800, P]
    a_grid = [1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]

    best = None  # (loo_err, K, alpha, idx)
    results = []
    for K in K_grid:
        idx = np.sort(order[:K])
        X = Fc[:, idx]
        G = X.T @ X
        G = 0.5 * (G + G.T)
        evals, evecs = np.linalg.eigh(G)
        evals = np.clip(evals, 0.0, None)
        XV = X @ evecs                       # (N,K) projections
        XtY = X.T @ Yc                       # (K,C)
        VtXtY = evecs.T @ XtY                # (K,C)
        for a in a_grid:
            d = evals + a
            coef = evecs @ (VtXtY / d[:, None])          # (K,C)
            pred = X @ coef                              # (N,C)
            h = np.sum((XV ** 2) / d[None, :], axis=1)    # (N,)
            denom = np.clip(1.0 - h, 1e-6, None)
            loo_pred = ybar[None, :] + (pred - Yc) / denom[:, None]
            loo_acc = float(np.mean(np.argmax(loo_pred, axis=1) == y))
            train_acc = float(np.mean(np.argmax(ybar[None, :] + pred, axis=1) == y))
            results.append({"K": int(K), "alpha": float(a),
                            "loo_acc": loo_acc, "train_acc": train_acc})
            if best is None or loo_acc > best[0]:
                best = (loo_acc, int(K), float(a), idx)

    loo_acc, K, alpha, idx = best
    X = Fc[:, idx]
    G = X.T @ X
    G = 0.5 * (G + G.T)
    evals, evecs = np.linalg.eigh(G)
    evals = np.clip(evals, 0.0, None)
    coef = evecs @ ((evecs.T @ (X.T @ Yc)) / (evals + alpha)[:, None])   # (K,C)

    # ---------------- write head weights (same names/shapes) ----------------
    HW = np.zeros((C, P), dtype=np.float64)
    HW[:, idx] = coef.T
    HB = ybar - mu @ HW.T                                  # (C,)

    # keep logits in a sane range (argmax invariant, cross-entropy friendlier)
    logits = F @ HW.T + HB[None, :]
    spread = float(np.mean(np.max(logits, axis=1) - np.min(logits, axis=1)))
    if spread > 1e-6:
        s = min(1.0, 6.0 / spread)
        HW = HW * s
        HB = HB * s

    w["head.weight"] = np.asarray(HW, dtype=np.float32)
    w["head.bias"] = np.asarray(HB, dtype=np.float32)

    # ---------------- diagnostics ----------------
    pred = np.argmax(F @ HW.T + HB[None, :], axis=1)
    train_acc = float(np.mean(pred == y))
    per_class = {}
    for c in classes:
        m = (y == c)
        per_class[int(c)] = float(np.mean(pred[m] == c))
    pred_hist = np.bincount(pred, minlength=C).tolist()

    w["head.weight"] = np.clip(np.nan_to_num(w["head.weight"], nan=0.0, posinf=0.0, neginf=0.0), -1000.0, 1000.0)
    w["head.bias"] = np.clip(np.nan_to_num(w["head.bias"], nan=0.0, posinf=0.0, neginf=0.0), -1000.0, 1000.0)

    diagnostics = {
        "method": "closed-form ridge head on cached features (feature-selected, LOOCV-chosen K and lambda)",
        "K_selected": int(K),
        "ridge_lambda": float(alpha),
        "loo_acc_estimate": float(loo_acc),
        "train_acc_after_edit": train_acc,
        "per_class_train_acc": per_class,
        "pred_class_histogram": pred_hist,
        "head_weight_abs_max": float(np.max(np.abs(w["head.weight"]))),
        "head_weight_rms": float(np.sqrt(np.mean(w["head.weight"].astype(np.float64) ** 2))),
        "head_bias": [float(v) for v in np.asarray(w["head.bias"], dtype=np.float64)],
        "logit_spread_mean": spread,
        "cv_table": results[:40],
        "feature_score_top20": [float(v) for v in score[order[:20]]],
        "conv_layers_unchanged": True
    }
    return {"weights": w, "diagnostics": diagnostics}
