def edit(w, f):
    import numpy as np

    # ---------------- forward helpers (numpy only) ----------------
    def conv3x3(x, W, b):
        N, C, H, Wd = x.shape
        O = W.shape[0]
        xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
        v = np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(2, 3))
        v = v.transpose(0, 2, 3, 1, 4, 5).reshape(N * H * Wd, C * 9)
        out = v @ W.reshape(O, C * 9).T + b
        return out.reshape(N, H, Wd, O).transpose(0, 3, 1, 2)

    def maxpool2(x):
        N, C, H, Wd = x.shape
        return x.reshape(N, C, H // 2, 2, Wd // 2, 2).max(axis=(3, 5))

    def conv1_pool(w, x, chunk=200):
        cw1 = w["conv1.weight"]; cb1 = w["conv1.bias"]
        N = x.shape[0]; outs = []
        for i in range(0, N, chunk):
            a1 = conv3x3(x[i:i + chunk], cw1, cb1)
            p1 = maxpool2(np.maximum(a1, 0.0))
            outs.append(p1)
        return np.concatenate(outs, axis=0)

    def conv2_feat(w2, b2, p1, chunk=200):
        N = p1.shape[0]; outs = []
        for i in range(0, N, chunk):
            a2 = conv3x3(p1[i:i + chunk], w2, b2)
            f2 = maxpool2(np.maximum(a2, 0.0))
            outs.append(f2.reshape(f2.shape[0], -1))
        return np.concatenate(outs, axis=0)

    def softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    # ---------------- data ----------------
    x = np.asarray(f["x"], dtype=np.float32)
    y = np.asarray(f["y"]).astype(np.int64)
    N = x.shape[0]
    K = 10
    Y1 = np.zeros((N, K), dtype=np.float64)
    Y1[np.arange(N), y] = 1.0

    diag = {}
    w2_orig = np.array(w["conv2.weight"], dtype=np.float32)
    b2_orig = np.array(w["conv2.bias"], dtype=np.float32)

    # ---------------- conv1 pooled map (fixed) ----------------
    P1 = conv1_pool(w, x)                       # [N,16,14,14]
    diag["p1_mean"] = float(P1.mean())
    diag["p1_std"] = float(P1.std())

    # sanity: reproduce cached features with the ORIGINAL conv2
    F_orig = conv2_feat(w2_orig, b2_orig, P1)
    F_cached = np.asarray(f["features"], dtype=np.float32)
    if F_cached.shape == F_orig.shape:
        diag["feat_repro_maxabs"] = float(np.abs(F_orig - F_cached).max())
    else:
        diag["feat_repro_maxabs"] = -1.0
    use_cached = diag["feat_repro_maxabs"] < 1e-3
    if use_cached:
        F_orig = F_cached.astype(np.float64)

    # ---------------- closed-form head candidates ----------------
    def ridge_fit(X, Yt, lam):
        mu = X.mean(axis=0)
        Xc = X - mu
        G = Xc.T @ Xc
        G.flat[:: G.shape[0] + 1] += lam
        Wt = np.linalg.solve(G, Xc.T @ Yt)          # [d,K]
        b = Yt.mean(axis=0) - mu @ Wt
        return Wt, b

    def lda_fit(X, Yt, alpha):
        mu = X.mean(axis=0)
        Xc = X - mu
        d = X.shape[1]
        S = np.zeros((d, d))
        for c in range(K):
            Xc_c = X[Yt[:, c] > 0.5] - mu
            S += Xc_c.T @ Xc_c
        S /= max(N - K, 1)
        S.flat[:: d + 1] += alpha * (np.trace(S) / d + 1e-12)
        M = np.stack([X[Yt[:, c] > 0.5].mean(axis=0) - mu for c in range(K)], axis=1)
        Wt = np.linalg.solve(S, M)                  # [d,K]
        b = Yt.mean(axis=0) - mu @ Wt
        return Wt, b

    def acc_of(X, Wt, b):
        return float((np.argmax(X @ Wt + b, axis=1) == y).mean())

    def cv_score(X, fit_fn, params, folds=5):
        idx = np.arange(N)
        rs = np.random.RandomState(0)
        rs.shuffle(idx)
        parts = np.array_split(idx, folds)
        accs = []
        for k in range(folds):
            va = parts[k]
            tr = np.concatenate([parts[j] for j in range(folds) if j != k])
            Wt, b = fit_fn(X[tr], Y1[tr], params)
            accs.append(float((np.argmax(X[va] @ Wt + b, axis=1) == y[va]).mean()))
        return float(np.mean(accs)), accs

    lam_grid = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
    best = {"cv": -1.0}
    for lam in lam_grid:
        cv, _ = cv_score(F_orig, ridge_fit, lam)
        if cv > best["cv"]:
            best = {"cv": cv, "lam": lam, "kind": "ridge"}
    for al in [0.01, 0.03, 0.1, 0.3, 1.0]:
        cv, _ = cv_score(F_orig, lda_fit, al)
        if cv > best["cv"]:
            best = {"cv": cv, "alpha": al, "kind": "lda"}
    diag["head_only_best_cv"] = float(best["cv"])
    diag["head_only_kind"] = best["kind"]
    diag["head_only_param"] = float(best.get("lam", best.get("alpha")))

    # ---------------- discriminative conv2 bank (closed form) ----------------
    # target maps: soft class occupancy at 14x14 derived from the image itself
    xs = x[:, 0].reshape(N, 28, 28).astype(np.float64)
    small = xs.reshape(N, 14, 2, 14, 2).mean(axis=(2, 4))          # [N,14,14]
    mmax = small.reshape(N, -1).max(axis=1, keepdims=True) + 1e-6
    small = small / mmax[:, :, None]                               # in [0,1]
    occ = small.reshape(N, -1)                                     # [N,196]

    # patch matrix of P1 (16x3x3 -> 144 taps), built in chunks
    def patch_matrix(P1, rows):
        # rows: indices into N
        out = np.empty((len(rows), 196, 144), dtype=np.float32)
        Pp = np.pad(P1[rows], ((0, 0), (0, 0), (1, 1), (1, 1)))
        v = np.lib.stride_tricks.sliding_window_view(Pp, (3, 3), axis=(2, 3))
        v = v.transpose(0, 2, 3, 1, 4, 5).reshape(len(rows), 196, 144)
        out[:] = v
        return out

    def build_detectors(rows, lam=1.0):
        """one-vs-rest local least squares detectors -> 10 filters (16,3,3)"""
        A = patch_matrix(P1, rows).reshape(-1, 144).astype(np.float64)
        G = A.T @ A
        G.flat[:: 145] += lam
        Ws = np.zeros((K, 16, 3, 3), dtype=np.float64)
        for c in range(K):
            t = (Y1[rows, c][:, None] * occ[rows]).reshape(-1)
            rhs = A.T @ t
            sol = np.linalg.solve(G, rhs)
            Ws[c] = sol.reshape(16, 3, 3)
        return Ws

    def build_bank(rows):
        """32 channels: 16 original box-blur + 16 discriminative detectors"""
        det = build_detectors(rows)
        W2 = np.zeros((32, 16, 3, 3), dtype=np.float64)
        W2[:16] = w2_orig[:, :, 0, 0].reshape(16, 16, 1, 1)[:, :, 0, 0].reshape(16, 16, 1, 1) * 0  # placeholder
        # original conv2 channels (keep exactly)
        W2[:16] = w2_orig[:16].astype(np.float64)
        # discriminative detectors: 10 class detectors + 6 extra (confusable classes)
        extra = [0, 2, 4, 6, 0, 6]
        W2[16:26] = det
        W2[26:32] = det[extra]
        return W2

    # out-of-fold evaluation of the new bank (filters rebuilt per fold -> no leakage)
    rs = np.random.RandomState(0)
    order = np.arange(N); rs.shuffle(order)
    parts = np.array_split(order, 5)
    oof_new = np.zeros((N, K))
    oof_new_ok = True
    for k in range(5):
        va = parts[k]
        tr = np.concatenate([parts[j] for j in range(5) if j != k])
        W2k = build_bank(tr)
        Ftr = conv2_feat(W2k.astype(np.float32), b2_orig, P1[tr])
        Fva = conv2_feat(W2k.astype(np.float32), b2_orig, P1[va])
        # inner ridge lambda selection on the training part (fixed grid, closed form)
        bestl = lam_grid[0]; bestc = -1.0
        for lam in lam_grid:
            Wt, b = ridge_fit(Ftr.astype(np.float64), Y1[tr], lam)
            c = float((np.argmax(Ftr @ Wt + b, axis=1) == y[tr]).mean())
            if c > bestc:
                bestc = c; bestl = lam
        Wt, b = ridge_fit(Ftr.astype(np.float64), Y1[tr], bestl)
        oof_new[va] = Fva.astype(np.float64) @ Wt + b
    cv_new = float((np.argmax(oof_new, axis=1) == y).mean())
    diag["new_bank_oof_acc"] = cv_new

    # ---------------- choose final solution ----------------
    diag["current_pool_acc"] = float((np.argmax(np.asarray(f["logits"]), axis=1) == y).mean())

    use_new = cv_new > best["cv"] + 0.002
    if use_new:
        W2f = build_bank(np.arange(N)).astype(np.float32)
        Fnew = conv2_feat(W2f, b2_orig, P1).astype(np.float64)
        # final lambda: pick by CV on the new bank (closed form, no leakage in weights)
        bl = lam_grid[0]; bc = -1.0
        for lam in lam_grid:
            cv, _ = cv_score(Fnew, ridge_fit, lam)
            if cv > bc:
                bc = cv; bl = lam
        Wt, b = ridge_fit(Fnew, Y1, bl)
        diag["chosen"] = "new_bank"
        diag["chosen_cv"] = float(bc)
        diag["chosen_lam"] = float(bl)
    else:
        W2f = w2_orig
        Ffin = F_orig
        if best["kind"] == "ridge":
            Wt, b = ridge_fit(Ffin, Y1, best["lam"])
        else:
            Wt, b = lda_fit(Ffin, Y1, best["alpha"])
        diag["chosen"] = "head_only_" + best["kind"]
        diag["chosen_cv"] = float(best["cv"])

    # ---------------- write out ----------------
    HW = np.asarray(w["head.weight"], dtype=np.float32)
    HB = np.asarray(w["head.bias"], dtype=np.float32)
    HW[:] = Wt.T.astype(np.float32)
    HB[:] = b.astype(np.float32)
    np.clip(HW, -1000.0, 1000.0, out=HW)
    np.clip(HB, -1000.0, 1000.0, out=HB)
    w["head.weight"] = HW
    w["head.bias"] = HB
    w["conv2.weight"] = np.clip(np.asarray(W2f, dtype=np.float32), -1000.0, 1000.0)
    w["conv2.bias"] = np.clip(np.asarray(b2_orig, dtype=np.float32), -1000.0, 1000.0)

    # ---------------- diagnostics ----------------
    Ffin = conv2_feat(w["conv2.weight"], w["conv2.bias"], P1).astype(np.float64)
    logits_new = Ffin @ Wt + b
    pred = np.argmax(logits_new, axis=1)
    diag["final_pool_acc"] = float((pred == y).mean())
    diag["final_pool_ce"] = float(-np.log(np.maximum(softmax(logits_new)[np.arange(N), y], 1e-12)).mean())
    diag["logit_std"] = float(logits_new.std())
    per_class = {}
    for c in range(K):
        m = (y == c)
        per_class[str(c)] = float((pred[m] == c).mean()) if m.sum() else 0.0
    diag["per_class_acc"] = per_class
    diag["head_w_std"] = float(HW.std())
    diag["head_w_maxabs"] = float(np.abs(HW).max())
    diag["conv2_w_maxabs"] = float(np.abs(w["conv2.weight"]).max())
    diag["finite_ok"] = bool(np.isfinite(HW).all() and np.isfinite(HB).all()
                             and np.isfinite(w["conv2.weight"]).all())

    return {"weights": w, "diagnostics": diag}
