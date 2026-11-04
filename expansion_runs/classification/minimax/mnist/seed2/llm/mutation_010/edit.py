def edit(w, f):
    import numpy as np

    # ---------------- data ----------------
    X = np.asarray(f["x"], dtype=np.float64).reshape(len(f["x"]), -1)
    y = np.asarray(f["y"]).astype(np.int64).ravel()
    N, D = X.shape
    C = 10
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    W1 = np.asarray(w["fc1.weight"], dtype=np.float64)
    b1 = np.asarray(w["fc1.bias"], dtype=np.float64)
    W2 = np.asarray(w["fc2.weight"], dtype=np.float64)
    b2 = np.asarray(w["fc2.bias"], dtype=np.float64)

    Horig = np.asarray(f["features"], dtype=np.float64)          # current hidden features
    Hrecomp = np.maximum(0.0, X @ W1.T + b1)                    # sanity check
    recomp_err = float(np.max(np.abs(Horig - Hrecomp)))

    # ---------------- helpers (closed form only) ----------------
    def ridge_all(Xd, Yd, lams):
        """Closed-form ridge for every lambda at once via one eigendecomposition.
        Xd: (n,d) design (already centered), Yd: (n,k) targets (centered).
        Returns list of (d,k) solutions."""
        n, d = Xd.shape
        G = Xd.T @ Xd
        s, V = np.linalg.eigh(G)
        VtY = V.T @ (Xd.T @ Yd)
        out = []
        for lam in lams:
            out.append(V @ (VtY / (s[:, None] + lam)))
        return out

    def cv_acc(Xd, Yd, lams, folds=5):
        """5-fold CV accuracy of the closed-form ridge for each lambda (no training loop)."""
        n = Xd.shape[0]
        rs = np.random.RandomState(0)
        perm = rs.permutation(n)
        fold_id = np.zeros(n, dtype=np.int64)
        for k in range(folds):
            fold_id[perm[k::folds]] = k
        accs = np.zeros(len(lams), dtype=np.float64)
        for k in range(folds):
            tr = fold_id != k
            te = ~tr
            Xtr = Xd[tr]; Ytr = Yd[tr]
            mu = Xtr.mean(axis=0)
            Ym = Ytr.mean(axis=0)
            Xc = Xtr - mu
            Yc = Ytr - Ym
            sols = ridge_all(Xc, Yc, lams)
            for li, Wl in enumerate(sols):
                pred = (Xd[te] - mu) @ Wl + Ym
                accs[li] += float((pred.argmax(axis=1) == y[te]).sum())
        return accs / n

    def fit_design(Xd, lams):
        """Return (best_lambda, W(10,d), b(10), cv_acc_curve) using centered ridge."""
        mu = Xd.mean(axis=0)
        Ym = Y.mean(axis=0)
        Xc = Xd - mu
        Yc = Y - Ym
        accs = cv_acc(Xd, Y, lams)
        li = int(np.argmax(accs))
        Wl = ridge_all(Xc, Yc, [lams[li]])[0]
        b = Ym - mu @ Wl
        return lams[li], Wl, b, accs

    # ---------------- pixel-space ridge directions (closed form, all lambdas at once) ----------------
    xbar = X.mean(axis=0)
    Xc = X - xbar
    G = Xc.T @ Xc
    sp, Vp = np.linalg.eigh(G)
    VtYX = Vp.T @ (Xc.T @ Y)                       # (784,10)
    lams_pix = np.array([0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0])
    Wpix = []
    for lam in lams_pix:
        Wpix.append(Vp @ (VtYX / (sp[:, None] + lam)))   # (784,10) centered-pixel directions
    # pick pixel lambda by CV accuracy of the resulting linear classifier
    def pix_cv_acc(Wp):
        pred = Xc @ Wp
        return float((pred.argmax(axis=1) == y).mean())
    # honest CV for pixel lambda
    n = N
    rs = np.random.RandomState(1)
    perm = rs.permutation(n)
    fold_id = np.zeros(n, dtype=np.int64)
    for k in range(5):
        fold_id[perm[k::5]] = k
    pix_acc = np.zeros(len(lams_pix))
    for k in range(5):
        tr = fold_id != k; te = ~tr
        Xtr = X[tr]; Ytr = Y[tr]
        m = Xtr.mean(axis=0); Ym = Ytr.mean(axis=0)
        Xtrc = Xtr - m; Ytrc = Ytr - Ym
        Gtr = Xtrc.T @ Xtrc
        st, Vt = np.linalg.eigh(Gtr)
        VtY = Vt.T @ (Xtrc.T @ Ytrc)
        for li, lam in enumerate(lams_pix):
            Wl = Vt @ (VtY / (st[:, None] + lam))
            pred = (X[te] - m) @ Wl + Ym
            pix_acc[li] += float((pred.argmax(axis=1) == y[te]).sum())
    pix_acc /= n
    li_pix = int(np.argmax(pix_acc))
    lam_pix = float(lams_pix[li_pix])
    Wp = Vp @ (VtYX / (sp[:, None] + lam_pix))     # (784,10) directions, centered-pixel
    bpix = Y.mean(axis=0) - xbar @ Wp              # (10,)
    pix_lin_acc = float(((X @ Wp + bpix).argmax(axis=1) == y).mean())

    # ---------------- candidate designs ----------------
    lams_out = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0])

    # discriminative power of each existing hidden unit (between-class F statistic)
    cls_mean = np.zeros((C, Horig.shape[1]))
    for c in range(C):
        cls_mean[c] = Horig[y == c].mean(axis=0)
    gm = Horig.mean(axis=0)
    between = ((cls_mean - gm) ** 2).mean(axis=0)
    within = np.zeros(Horig.shape[1])
    for c in range(C):
        Hc = Horig[y == c]
        within += ((Hc - cls_mean[c]) ** 2).mean(axis=0) * (np.mean(y == c))
    Fstat = between / (within + 1e-8)
    order = np.argsort(-Fstat)
    keep108 = np.sort(order[:108])          # drop 20 least discriminative hidden units

    # pair features from pixel directions (exact linear scores via relu(s)-relu(-s))
    s_pix = X @ Wp + bpix                  # (N,10)
    pair_feat = np.concatenate([np.maximum(s_pix, 0.0), np.maximum(-s_pix, 0.0)], axis=1)  # (N,20)

    # random ReLU features (fixed seed) for an ELM-style design
    rs2 = np.random.RandomState(7)
    Wr = rs2.randn(D, 108) / np.sqrt(D)
    br = rs2.randn(108) * 0.5
    rand_feat = np.maximum(0.0, X @ Wr + br)

    designs = {
        "hidden_only": (Horig, None),
        "hidden_plus_pixel": (np.concatenate([pair_feat, Horig[:, keep108]], axis=1), "pair"),
        "hidden_plus_rand": (np.concatenate([pair_feat, rand_feat], axis=1), "pair"),
        "rand_plus_hidden": (np.concatenate([Horig[:, keep108], rand_feat], axis=1), "none"),
    }

    results = {}
    for name, (Xd, kind) in designs.items():
        lam, Wl, b, accs = fit_design(Xd, lams_out)
        tr_acc = float(((Xd @ Wl + b).argmax(axis=1) == y).mean())
        results[name] = dict(lam=lam, W=Wl, b=b, cv=float(accs.max()), train=tr_acc, kind=kind)

    best_name = max(results, key=lambda k: (results[k]["cv"], results[k]["train"]))
    best = results[best_name]
    Wl = best["W"]; bl = best["b"]

    # ---------------- assemble new weights ----------------
    if best_name == "hidden_only":
        newW1 = W1.copy(); newb1 = b1.copy()
        newW2 = Wl.T.copy(); newb2 = bl.copy()
    else:
        # rows 0..19 : +- pixel ridge directions (exact linear scores)
        newW1 = np.zeros((128, D), dtype=np.float64)
        newb1 = np.zeros(128, dtype=np.float64)
        newW1[0:10] = Wp.T
        newW1[10:20] = -Wp.T
        if best_name == "hidden_plus_pixel":
            newW1[20:128] = W1[keep108]
            newb1[20:128] = b1[keep108]
        else:
            newW1[20:128] = Wr.T
            newb1[20:128] = br
        newW2 = Wl.T.copy(); newb2 = bl.copy()

    # numerical hygiene
    newW1 = np.clip(newW1, -1000.0, 1000.0)
    newb1 = np.clip(newb1, -1000.0, 1000.0)
    newW2 = np.clip(newW2, -1000.0, 1000.0)
    newb2 = np.clip(newb2, -1000.0, 1000.0)

    w["fc1.weight"] = newW1.astype(np.float32)
    w["fc1.bias"] = newb1.astype(np.float32)
    w["fc2.weight"] = newW2.astype(np.float32)
    w["fc2.bias"] = newb2.astype(np.float32)

    # ---------------- diagnostics ----------------
    Hn = np.maximum(0.0, X @ newW1.T + newb1)
    Ln = Hn @ newW2.T + newb2
    acc_new = float((Ln.argmax(axis=1) == y).mean())
    logits_old = np.asarray(f["logits"], dtype=np.float64)
    acc_old = float((logits_old.argmax(axis=1) == y).mean())
    probs_old = np.asarray(f["probs"], dtype=np.float64)
    p_old = np.clip(probs_old[np.arange(N), y], 1e-12, 1.0)
    ce_old = float(-np.log(p_old).mean())
    z = Ln - Ln.max(axis=1, keepdims=True)
    ez = np.exp(z); pz = ez / ez.sum(axis=1, keepdims=True)
    ce_new = float(-np.log(np.clip(pz[np.arange(N), y], 1e-12, 1.0)).mean())

    diag = {
        "design": best_name,
        "lambda_out": float(best["lam"]),
        "cv_acc": float(best["cv"]),
        "train_acc_new": acc_new,
        "train_acc_old": acc_old,
        "ce_new": ce_new,
        "ce_old": ce_old,
        "lambda_pixel": lam_pix,
        "pixel_linear_acc": pix_lin_acc,
        "pixel_cv_acc": float(pix_acc.max()),
        "hidden_recompute_max_err": recomp_err,
        "candidates": {k: {"cv": float(v["cv"]), "train": float(v["train"]),
                           "lam": float(v["lam"])} for k, v in results.items()},
        "kept_hidden_units": int(len(keep108)),
        "fc1_absmax": float(np.abs(newW1).max()),
        "fc2_absmax": float(np.abs(newW2).max()),
        "feature_zero_frac_new": float((Hn == 0).mean()),
    }
    return {"weights": w, "diagnostics": diag}
