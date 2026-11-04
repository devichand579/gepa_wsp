def edit(w, f):
    import numpy as np

    x = np.asarray(f["x"]); y = np.asarray(f["y"]).astype(np.int64)
    N, L = x.shape
    V = w["embed.weight"].shape[0]
    D = w["embed.weight"].shape[1]

    # ---------- design matrices from the permitted pool ----------
    mask = (x != 0)
    rows = np.repeat(np.arange(N), mask.sum(1))
    cols = x[mask].ravel()
    Xb = np.zeros((N, V), dtype=np.float64)
    Xb[rows, cols] = 1.0                      # binary presence
    Xc = np.zeros((N, V), dtype=np.float64)
    np.add.at(Xc, (rows, cols), 1.0)          # counts

    y1 = (y == 1)
    N1 = float(y1.sum()); N0 = float(N - y1.sum())
    n1 = Xb[y1].sum(0); n0 = Xb[~y1].sum(0)

    # ---------- closed-form candidate 1: smoothed NB log-odds ----------
    a = 1.0
    r_nb = np.log((n1 + a) / (N1 + 2 * a)) - np.log((n0 + a) / (N0 + 2 * a))
    r_nb[0] = 0.0                             # pad token is neutral

    # ---------- closed-form candidate 2: ridge normal equations ----------
    t = (2.0 * y - 1.0)
    XtX = Xc.T @ Xc
    XtY = Xc.T @ t
    d = np.arange(V)
    best = None
    for lam in (10.0, 30.0, 100.0, 300.0, 1000.0):
        A = XtX.copy()
        A[d, d] += lam
        try:
            wr = np.linalg.solve(A, XtY)
        except Exception:
            continue
        sc = Xc @ wr
        acc = float(np.mean((sc > 0).astype(np.int64) == y))
        if best is None or acc > best[0]:
            best = (acc, wr)
    r_ridge = best[1] if best is not None else np.zeros(V)
    r_ridge[0] = 0.0

    # ---------- pick the better candidate by 2-fold closed-form CV ----------
    rng = np.random.RandomState(0)
    perm = rng.permutation(N)
    folds = [perm[:N // 2], perm[N // 2:]]

    def nb_scores(idx, r):
        return Xc[idx] @ r

    def ridge_fit(idx, lam):
        Xf = Xc[idx]; tf = t[idx]
        A = Xf.T @ Xf
        A[d, d] += lam
        try:
            return np.linalg.solve(A, Xf.T @ tf)
        except Exception:
            return np.zeros(V)

    def cv_acc(score_fn):
        accs = []
        for k in range(2):
            te = folds[k]; tr = folds[1 - k]
            s = score_fn(tr, te)
            accs.append(float(np.mean((s > 0).astype(np.int64) == y[te])))
        return float(np.mean(accs))

    cv_nb = cv_acc(lambda tr, te: nb_scores(te, r_nb))
    cv_ridge = 0.0
    best_lam = 100.0
    for lam in (10.0, 30.0, 100.0, 300.0, 1000.0):
        acc = cv_acc(lambda tr, te, lam=lam: Xc[te] @ ridge_fit(tr, lam))
        if acc > cv_ridge:
            cv_ridge = acc; best_lam = lam

    use_ridge = cv_ridge > cv_nb + 0.002
    if use_ridge:
        r = ridge_fit(np.arange(N), best_lam)
        r[0] = 0.0
        method = "ridge"
    else:
        r = r_nb.copy()
        method = "nb_logodds"

    # ---------- shrink the <unk> token toward neutral (rare-word noise) ----------
    r[1] = 0.25 * r[1]

    # ---------- scale / clip the per-token score ----------
    s = np.clip(r, -3.0, 3.0)
    R = 3.0
    c = np.sqrt(np.maximum(R * R - s * s, 1e-6))

    # ---------- build constant-norm, zero-mean embedding rows ----------
    u = np.zeros(D); u[0] = 1.0
    v = np.zeros(D); v[1] = 1.0
    u -= u.mean(); v -= v.mean()
    u /= np.linalg.norm(u); v -= v.mean(); v /= np.linalg.norm(v)
    # orthogonalise v against u
    v -= (v @ u) * u
    v -= v.mean(); v /= np.linalg.norm(v)

    E = np.zeros((V, D), dtype=np.float64)
    E[:] = s[:, None] * u[None, :] + c[:, None] * v[None, :]
    E[0] = 0.0                                  # pad row -> exactly zero

    # ---------- write weights ----------
    w["embed.weight"][:] = E.astype(np.float32)
    w["pos.weight"][:] = 0.0

    for b in ("block0", "block1"):
        for k in ("qkv.weight", "qkv.bias", "proj.weight", "proj.bias",
                  "ff1.weight", "ff1.bias", "ff2.weight", "ff2.bias"):
            w[f"{b}.{k}"][:] = 0.0
        w[f"{b}.norm1.weight"][:] = 1.0
        w[f"{b}.norm1.bias"][:] = 0.0
        w[f"{b}.norm2.weight"][:] = 1.0
        w[f"{b}.norm2.bias"][:] = 0.0

    w["norm.weight"][:] = 1.0
    w["norm.bias"][:] = 0.0

    # pooled dim0 = (sqrt(D)/R) * mean_t s_t ;  logit1-logit0 = alpha*pooled0 + beta
    alpha = float(R / np.sqrt(D)) * 4.0
    hw = np.zeros((2, D), dtype=np.float32)
    hw[1, 0] = alpha
    hb = np.zeros(2, dtype=np.float32)
    w["head.weight"][:] = hw
    w["head.bias"][:] = hb

    # ---------- diagnostics (pool, in-sample) ----------
    sc = Xc @ s
    pred = (sc > 0).astype(np.int64)
    diag = {
        "method": method,
        "cv_nb": round(cv_nb, 4),
        "cv_ridge": round(cv_ridge, 4),
        "ridge_lam": best_lam,
        "pool_acc": round(float(np.mean(pred == y)), 4),
        "pool_acc_nb": round(float(np.mean(((Xc @ r_nb) > 0).astype(np.int64) == y)), 4),
        "pool_acc_ridge": round(float(np.mean(((Xc @ r_ridge) > 0).astype(np.int64) == y)), 4),
        "score_mean": round(float(sc.mean()), 4),
        "score_std": round(float(sc.std()), 4),
        "unk_rate": round(float(np.mean(x == 1)), 4),
        "mean_len": round(float(mask.sum(1).mean()), 2),
        "r_absmax": round(float(np.abs(r).max()), 3),
        "r_top": [round(float(v), 3) for v in np.sort(r)[-12:][::-1]],
        "r_bot": [round(float(v), 3) for v in np.sort(r)[:12]],
        "alpha": round(alpha, 4),
    }
    return {"weights": w, "diagnostics": diag}
