import numpy as np

def edit(w, f):
    X = np.asarray(f["x"], dtype=np.int64)      # [N,64] token ids, 0=pad
    y = np.asarray(f["y"], dtype=np.int64)      # [N]
    N, L = X.shape
    V = w["embed.weight"].shape[0]

    # ---- Build length-normalized unigram count design matrix ----
    mask = (X != 0).astype(np.float64)          # [N,64]
    T = mask.sum(axis=1)                         # [N]
    T[T == 0] = 1.0
    # counts per vocab word
    Xc = np.zeros((N, V), dtype=np.float64)
    for i in range(N):
        ids = X[i][mask[i] > 0]
        if ids.size:
            np.add.at(Xc[i], ids, 1.0)
    Xc /= T[:, None]                             # mean-pooled features

    t = 2.0 * y.astype(np.float64) - 1.0         # ±1 targets
    XtX = Xc.T @ Xc
    Xty = Xc.T @ t
    n = float(N)

    # ---- Choose ridge alpha by held-out CV (closed form per alpha) ----
    rng = np.random.RandomState(0)
    perm = rng.permutation(N)
    nfold = 5
    folds = np.array_split(perm, nfold)

    def fit_scores(Xtr, ttr, alpha):
        A = Xtr.T @ Xtr + alpha * np.eye(V)
        b = Xtr.T @ ttr
        return np.linalg.solve(A, b)

    best_alpha, best_cv = None, -1.0
    for alpha in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
        accs = []
        for k in range(nfold):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(nfold) if j != k])
            s = fit_scores(Xc[tr], t[tr], alpha)
            pred = (Xc[te] @ s) > 0
            accs.append(np.mean(pred == (t[te] > 0)))
        cv = float(np.mean(accs))
        if cv > best_cv:
            best_cv, best_alpha = cv, alpha

    # ---- Final fit on all data ----
    s = fit_scores(Xc, t, best_alpha)            # per-word score s(v)
    bias = 0.0
    # intercept via residual mean (ridge shrinks it; recompute simple bias)
    raw = Xc @ s
    # choose bias to maximize accuracy on the ±1 margin
    cand = np.sort(raw)
    # threshold that best separates t>0 vs t<0
    order = np.argsort(raw)
    ts = t[order]
    cum = np.cumsum(ts)
    # predict +1 when raw > thr  -> correct count = (#pos with raw>thr) + (#neg with raw<=thr)
    pos_total = np.sum(t > 0)
    neg_total = np.sum(t < 0)
    # for threshold between raw[order[i]] and raw[order[i+1]]
    best_thr, best_acc = 0.0, -1.0
    # evaluate thresholds at midpoints
    for i in range(N - 1):
        thr = 0.5 * (raw[order[i]] + raw[order[i + 1]])
        # positives above thr
        npos_above = pos_total - np.sum(ts[:i + 1] > 0)
        nneg_below = np.sum(ts[:i + 1] < 0)
        acc = (npos_above + nneg_below) / N
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    bias = -best_thr

    # ---- Encode s(v) into embed dim 0 with zero-mean/unit-variance rows ----
    c = max(4.0, float(np.max(np.abs(s))) / 3.9 + 1e-6)
    a = s / c                                    # |a| < 4
    E = np.zeros((V, 32), dtype=np.float64)
    E[:, 0] = a
    E[:, 1] = -a
    rem = 32.0 - 2.0 * a * a
    rem = np.clip(rem, 0.0, None)
    half = np.sqrt(rem / 2.0)
    E[:, 2] = half
    E[:, 3] = -half
    # row 0 (PAD) must stay zero-safe: give it a valid zero-mean unit-var vector
    E[0, :] = 0.0
    E[0, 0] = 0.0
    E[0, 1] = 0.0
    E[0, 2] = np.sqrt(32.0 / 2.0)
    E[0, 3] = -np.sqrt(32.0 / 2.0)

    w["embed.weight"] = E.astype(np.float32)
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)

    # ---- Head: read dim 0 with gain c, plus bias ----
    hw = np.zeros_like(w["head.weight"], dtype=np.float32)
    hb = np.zeros_like(w["head.bias"], dtype=np.float32)
    # logit1 - logit0 = c * pooled[0] + (hb1 - hb0)
    hw[1, 0] = float(c)
    hw[0, 0] = 0.0
    hb[1] = float(bias) / 2.0
    hb[0] = -float(bias) / 2.0
    w["head.weight"] = hw
    w["head.bias"] = hb

    # ---- Keep all other modules zero / identity ----
    for k in list(w.keys()):
        if k in ("embed.weight", "pos.weight", "head.weight", "head.bias"):
            continue
        arr = np.asarray(w[k])
        if arr.ndim == 1 and arr.shape[0] in (32, 64, 96) and "norm" in k and "weight" in k:
            w[k] = np.ones_like(arr, dtype=np.float32)
        elif arr.ndim == 1 and "norm" in k and "bias" in k:
            w[k] = np.zeros_like(arr, dtype=np.float32)
        else:
            w[k] = np.zeros_like(arr, dtype=np.float32)

    # final norm must be identity scale
    w["norm.weight"] = np.ones_like(w["norm.weight"], dtype=np.float32)
    w["norm.bias"] = np.zeros_like(w["norm.bias"], dtype=np.float32)

    # ---- Sanity diagnostics ----
    pred = (Xc @ s + bias) > 0
    train_acc = float(np.mean(pred == (t > 0)))
    diag = {
        "best_alpha": float(best_alpha),
        "cv_acc": float(best_cv),
        "train_acc_linear": train_acc,
        "s_absmax": float(np.max(np.abs(s))),
        "s_std": float(np.std(s)),
        "c": float(c),
        "bias": float(bias),
        "embed_nonzero": int(np.count_nonzero(w["embed.weight"])),
    }
    return {"weights": w, "diagnostics": diag}
