def edit(w, f):
    import numpy as np

    # ---------------- helpers ----------------
    def conv2d(x, W, b):
        # x: [N,Cin,H,W], W: [Cout,Cin,kh,kw] -> [N,Cout,H,W], cross-correlation, pad=1
        N, Cin, H, Wd = x.shape
        Cout, _, kh, kw = W.shape
        xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
        # im2col
        cols = np.empty((N, Cin * kh * kw, H * Wd), dtype=np.float32)
        idx = 0
        for c in range(Cin):
            for i in range(kh):
                for j in range(kw):
                    cols[:, idx, :] = xp[:, c, i:i + H, j:j + Wd].reshape(N, H * Wd)
                    idx += 1
        Wm = W.reshape(Cout, Cin * kh * kw).T          # [Cin*kh*kw, Cout]
        out = np.einsum('nkc,co->no', cols, Wm, optimize=True).reshape(N, H, Wd, Cout)
        out = np.transpose(out, (0, 3, 1, 2)) + b.reshape(1, Cout, 1, 1)
        return out.astype(np.float32)

    def maxpool2(x):
        N, C, H, Wd = x.shape
        x = x.reshape(N, C, H // 2, 2, Wd // 2, 2)
        return x.max(axis=(3, 5))

    def forward_feats(x, c1w, c1b, c2w, c2b):
        h = np.maximum(conv2d(x, c1w, c1b), 0)
        h = maxpool2(h)
        h = np.maximum(conv2d(h, c2w, c2b), 0)
        h = maxpool2(h)
        return h.reshape(h.shape[0], -1)   # channel, row, col

    def softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def nll(logits, y):
        p = softmax(logits)
        return float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean())

    # ---------------- read params ----------------
    c1w = np.array(w["conv1.weight"], dtype=np.float32)
    c1b = np.array(w["conv1.bias"], dtype=np.float32)
    c2w = np.array(w["conv2.weight"], dtype=np.float32)
    c2b = np.array(w["conv2.bias"], dtype=np.float32)
    hw = np.array(w["head.weight"], dtype=np.float32)
    hb = np.array(w["head.bias"], dtype=np.float32)

    X = np.asarray(f["x"], dtype=np.float32)
    Y = np.asarray(f["y"]).astype(np.int64)
    N = X.shape[0]
    Yh = np.zeros((N, 10), dtype=np.float32)
    Yh[np.arange(N), Y] = 1.0

    # ---------------- verify forward pass against cached arrays ----------------
    Fb = np.asarray(f["features"], dtype=np.float32)
    Fb_batch = np.asarray(f["layer.head.input0"], dtype=np.float32)
    c1o_batch = np.asarray(f["layer.conv1.output"], dtype=np.float32)
    c2o_batch = np.asarray(f["layer.conv2.output"], dtype=np.float32)

    def conv1_out(x):
        return conv2d(x, c1w, c1b)

    def conv2_out(h1):
        return conv2d(h1, c2w, c2b)

    chk = {}
    # batch check (conv1 pre-relu, conv2 pre-relu, features)
    bidx = np.asarray(f["batch.ids"]).astype(np.int64)
    bx = np.asarray(f["batch.x"], dtype=np.float32)
    o1 = conv1_out(bx)
    chk["conv1_err"] = float(np.abs(o1 - c1o_batch).max())
    h1 = maxpool2(np.maximum(o1, 0))
    o2 = conv2_out(h1)
    chk["conv2_err"] = float(np.abs(o2 - c2o_batch).max())
    fb = forward_feats(bx, c1w, c1b, c2w, c2b)
    chk["feat_batch_err"] = float(np.abs(fb - Fb_batch).max())
    # full-pool feature check
    F = forward_feats(X, c1w, c1b, c2w, c2b)
    chk["feat_pool_err"] = float(np.abs(F - Fb).max())
    # logits check with current head
    Lb = F @ hw.T + hb
    chk["logit_err"] = float(np.abs(Lb - np.asarray(f["logits"], dtype=np.float32)).max())

    # ---------------- closed-form ridge machinery ----------------
    def ridge_fit(A, T, lam):
        # A: [n,d] design (already centered/augmented as desired), T: [n,k]
        d = A.shape[1]
        G = A.T @ A
        G.flat[:: d + 1] += lam
        try:
            Wt = np.linalg.solve(G, A.T @ T)
        except np.linalg.LinAlgError:
            Wt = np.linalg.lstsq(G, A.T @ T, rcond=None)[0]
        return Wt

    # standardize features (mean/std) for conditioning
    mu = F.mean(axis=0)
    sd = F.std(axis=0)
    sd[sd < 1e-6] = 1.0
    Z = (F - mu) / sd

    # ---------------- candidate 1: shrinkage LDA (class moments + shrunk cov) ----------------
    def lda_scores(Z, y, shrink):
        classes = np.arange(10)
        M = np.zeros((10, Z.shape[1]), dtype=np.float64)
        for c in classes:
            M[c] = Z[y == c].mean(axis=0)
        gm = Z.mean(axis=0)
        S = np.zeros((Z.shape[1], Z.shape[1]), dtype=np.float64)
        for c in classes:
            D = Z[y == c] - M[c]
            S += D.T @ D
        S /= max(N - 10, 1)
        d = Z.shape[1]
        tr = np.trace(S) / d
        S_s = (1 - shrink) * S + shrink * tr * np.eye(d)
        P = np.linalg.inv(S_s)
        Wc = M - gm
        return (Z - gm) @ P @ Wc.T, Wc, P

    # ---------------- candidate 2: ridge to one-hot ----------------
    def ridge_scores(Z, y, lam):
        T = np.zeros((N, 10))
        T[np.arange(N), y] = 1.0
        Wt = ridge_fit(Z, T, lam)
        return Z @ Wt, Wt

    # ---------------- K-fold closed-form CV to compare candidates ----------------
    rng = np.random.RandomState(0)
    perm = rng.permutation(N)
    folds = np.array_split(perm, 5)

    def cv_score(score_fn):
        accs = []
        for k in range(5):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(5) if j != k])
            sc_tr, sc_te = score_fn(tr, te)
            pred = sc_te.argmax(axis=1)
            accs.append(float((pred == Y[te]).mean()))
        return float(np.mean(accs)), accs

    def make_lda(shrink):
        def fn(tr, te):
            Ztr = Z[tr]
            ytr = Y[tr]
            M = np.zeros((10, Z.shape[1]))
            for c in range(10):
                M[c] = Ztr[ytr == c].mean(axis=0)
            gm = Ztr.mean(axis=0)
            S = np.zeros((Z.shape[1], Z.shape[1]))
            for c in range(10):
                D = Ztr[ytr == c] - M[c]
                S += D.T @ D
            S /= max(len(tr) - 10, 1)
            d = Z.shape[1]
            tr_ = np.trace(S) / d
            S_s = (1 - shrink) * S + shrink * tr_ * np.eye(d)
            P = np.linalg.inv(S_s)
            Wc = M - gm
            return (Ztr - gm) @ P @ Wc.T, (Z[te] - gm) @ P @ Wc.T
        return fn

    def make_ridge(lam):
        def fn(tr, te):
            T = np.zeros((len(tr), 10))
            T[np.arange(len(tr)), Y[tr]] = 1.0
            Wt = ridge_fit(Z[tr], T, lam)
            return Z[tr] @ Wt, Z[te] @ Wt
        return fn

    diag = {"forward_checks": chk}

    cands = {}
    for sh in (0.05, 0.1, 0.2, 0.35, 0.5, 0.7):
        cands["lda_s%.2f" % sh] = make_lda(sh)
    for lam in (1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0):
        cands["ridge_l%g" % lam] = make_ridge(lam)

    cvres = {}
    for name, fn in cands.items():
        a, per = cv_score(fn)
        cvres[name] = a
    diag["cv_acc"] = cvres

    # ---------------- baseline: current head, evaluated on pool ----------------
    base_pred = Lb.argmax(axis=1)
    diag["current_head_pool_acc"] = float((base_pred == Y).mean())
    diag["current_head_pool_ce"] = nll(Lb, Y)
    # current head under 5-fold CV (refit on train folds with the same construction is not
    # possible exactly, so we only report in-sample)
    diag["current_head_in_sample_ce"] = nll(Lb, Y)

    best_name = max(cvres, key=cvres.get)
    diag["best_candidate"] = best_name
    diag["best_cv_acc"] = cvres[best_name]

    # ---------------- build final head from the best candidate ----------------
    if best_name.startswith("lda"):
        sh = float(best_name.split("s")[1])
        sc, Wc, P = lda_scores(Z, Y, sh)
        # closed-form affine recalibration of the discriminant scores:
        # fit logits = A @ sc + b by least squares to one-hot targets
        A = np.concatenate([sc, np.ones((N, 1))], axis=1)
        Wt = ridge_fit(A, Yh, 1e-6)
        new_logits = A @ Wt
        # temperature-style sharpening is already absorbed by the affine fit
        new_head_w = (Wt[:10].T @ (P @ Wc.T)) / sd
        new_head_b = Wt[:10].T @ (-mu / sd) + Wt[10]
    else:
        lam = float(best_name.split("l")[1])
        sc, Wt = ridge_scores(Z, Y, lam)
        A = np.concatenate([sc, np.ones((N, 1))], axis=1)
        W2 = ridge_fit(A, Yh, 1e-6)
        new_logits = A @ W2
        new_head_w = (Wt.T @ W2[:10].T) / sd
        new_head_b = Wt.T @ (-mu / sd) + W2[10]

    # sanity: recompute logits from the new head weights on the same features
    chk_logits = F @ new_head_w.T + new_head_b
    diag["rebuild_logit_err"] = float(np.abs(chk_logits - new_logits).max())
    diag["new_head_pool_acc"] = float((chk_logits.argmax(axis=1) == Y).mean())
    diag["new_head_pool_ce"] = nll(chk_logits, Y)

    # ---------------- final safety: keep the better of (current head, new head) ----------------
    # compare on the pool; also require finite and bounded weights
    ok = (np.all(np.isfinite(new_head_w)) and np.all(np.isfinite(new_head_b)) and
          np.abs(new_head_w).max() <= 1000 and np.abs(new_head_b).max() <= 1000)
    diag["weights_ok"] = bool(ok)
    if not ok or diag["new_head_pool_acc"] < diag["current_head_pool_acc"] - 1e-9:
        # fall back: keep current head but sharpen it (temperature < 1) to cut CE
        # closed-form choice of the sharpening factor: pick the scale that minimises
        # the pooled NLL among a small vectorised candidate set
        scales = np.exp(np.linspace(np.log(0.25), np.log(40.0), 60))
        best = None
        for s in scales:
            v = nll(Lb * s, Y)
            if best is None or v < best[1]:
                best = (float(s), v)
        s = best[0]
        new_head_w = hw * s
        new_head_b = hb * s
        diag["fallback"] = True
        diag["fallback_scale"] = s
        diag["fallback_ce"] = best[1]
    else:
        diag["fallback"] = False

    w["head.weight"] = np.clip(new_head_w, -1000, 1000).astype(np.float32)
    w["head.bias"] = np.clip(new_head_b, -1000, 1000).astype(np.float32)

    # conv layers untouched
    w["conv1.weight"] = c1w
    w["conv1.bias"] = c1b
    w["conv2.weight"] = c2w
    w["conv2.bias"] = c2b

    diag["head_w_std"] = float(w["head.weight"].std())
    diag["head_w_absmax"] = float(np.abs(w["head.weight"]).max())
    diag["head_b"] = [float(v) for v in w["head.bias"]]
    return {"weights": w, "diagnostics": diag}
