import numpy as np

def edit(w, f):
    # ---------------- data ----------------
    X = np.asarray(f["x"], dtype=np.float64).reshape(len(f["x"]), -1)
    y = np.asarray(f["y"], dtype=np.int64)
    N, D = X.shape
    K = 10
    Y = np.zeros((N, K)); Y[np.arange(N), y] = 1.0

    W1 = np.asarray(w["fc1.weight"], dtype=np.float64)
    b1 = np.asarray(w["fc1.bias"], dtype=np.float64)
    W2 = np.asarray(w["fc2.weight"], dtype=np.float64)
    b2 = np.asarray(w["fc2.bias"], dtype=np.float64)

    # ---------------- current model (for error analysis) ----------------
    cur_logits = np.maximum(X @ W1.T + b1, 0.0) @ W2.T + b2
    cur_pred = cur_logits.argmax(1)
    wrong = cur_pred != y
    n_wrong = int(wrong.sum())

    # ---------------- class statistics ----------------
    mu = np.zeros((K, D)); cnt = np.zeros(K)
    for c in range(K):
        Xc = X[y == c]
        cnt[c] = len(Xc)
        mu[c] = Xc.mean(0) if len(Xc) else X.mean(0)
    mu_all = X.mean(0)
    Sw = np.zeros((D, D))
    for c in range(K):
        Xc = X[y == c] - mu[c]
        Sw += Xc.T @ Xc
    Sw /= max(N, 1)
    tr = np.trace(Sw) / D
    Sw = Sw + 0.05 * tr * np.eye(D)

    evals, U = np.linalg.eigh(Sw)
    evals = np.maximum(evals, 1e-8)
    # M = U diag(evals^-1/2)  ->  z = (x-mu_all) @ M ;  d.M = d^T Sw^-1 d
    M = U * (1.0 / np.sqrt(evals))[None, :]
    Z = (X - mu_all) @ M
    Zw = Z[wrong] if n_wrong > 30 else Z

    mu_w = np.zeros((K, D))
    for c in range(K):
        sel = (y == c) & wrong
        mu_w[c] = Z[sel].mean(0) if sel.sum() >= 3 else Z[y == c].mean(0)

    # ---------------- direction bank (whitened space) ----------------
    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else v

    # PCA of whitened data (total scatter)
    Zc = Z - Z.mean(0)
    Czz = (Zc.T @ Zc) / max(N, 1)
    evp, Vp = np.linalg.eigh(Czz)
    order = np.argsort(evp)[::-1]
    pca_dirs = [unit(Vp[:, i]) for i in order[:40]]

    # one-vs-rest LDA directions
    ovr = []
    for c in range(K):
        others = [j for j in range(K) if j != c]
        d = (mu[c] - mu[others].mean(0)) @ M
        ovr.append(unit(d))

    # pairwise LDA directions (sorted by whitened separation)
    pairs = []
    for a in range(K):
        for b in range(a + 1, K):
            d = (mu[a] - mu[b]) @ M
            sep = np.linalg.norm(d)
            pairs.append((sep, a, b, unit(d)))
    pairs.sort(key=lambda t: -t[0])
    pair_dirs = [p[3] for p in pairs]

    # error-focused directions: wrong-class mean vs correct-class mean
    err_dirs = []
    for c in range(K):
        d = mu_w[c] - Z[y == c].mean(0)
        err_dirs.append(unit(d))
    # hard-pair directions: class means of the most confused pairs
    hard_pairs = []
    for a in range(K):
        for b in range(K):
            if a >= b: continue
            sel = (y == a) & wrong & (cur_pred == b)
            if sel.sum() >= 3:
                hard_pairs.append((sel.sum(), unit(mu[a] - mu[b])))
    hard_pairs.sort(key=lambda t: -t[0])
    hard_dirs = [h[1] for h in hard_pairs[:12]]

    def build(n_ovr, n_pair, n_pca, n_err, n_hard, K_h):
        """returns (fc1 rows, fc2 col scaling) for the chosen composition"""
        dirs = []
        dirs += ovr[:n_ovr]
        dirs += pair_dirs[:n_pair]
        dirs += pca_dirs[:n_pca]
        dirs += err_dirs[:n_err]
        dirs += hard_dirs[:n_hard]
        dirs = dirs[:128 // (1 + K_h)]
        rows = []; scales = []
        for d in dirs:
            p = Z @ d
            lo = p.min(); hi = p.max()
            # affine (always-on) unit
            t0 = lo - 0.5 * (hi - lo + 1e-6) - 1.0
            rows.append((d, t0, 0))          # 0 = affine
            for k in range(K_h):
                q = 0.25 + 0.5 * k / max(K_h - 1, 1)
                t = lo + q * (hi - lo)
                rows.append((d, t, 1))      # 1 = hinge
        A = np.zeros((N, len(rows)))
        for j, (d, t, kind) in enumerate(rows):
            p = Z @ d
            if kind == 0:
                A[:, j] = p - t
            else:
                A[:, j] = np.maximum(p - t, 0.0)
        return A, rows

    # ---------------- ridge solve helpers (closed form) ----------------
    def ridge_fit(A, yv, lam):
        G = A.T @ A
        G.flat[:: G.shape[0] + 1] += lam
        B = A.T @ yv
        return np.linalg.solve(G, B)

    def loo_select(A, yv, lams):
        n = A.shape[0]
        G = A.T @ A
        B = A.T @ yv
        evals_, Q = np.linalg.eigh(G)
        Bt = Q.T @ B
        best = None
        for lam in lams:
            d = evals_ + lam
            H = (A @ Q) / d[None, :]                 # hat matrix rows
            h = np.einsum('ij,ij->i', H, A)
            alpha = Q @ (Bt / d[:, None])
            pred = A @ alpha
            resid = yv - pred
            denom = np.maximum(1.0 - h, 1e-6)
            loo = resid / denom[:, None]
            mse = float(np.mean(loo ** 2))
            acc = float(np.mean((pred.argmax(1) == yv.argmax(1))))
            if best is None or mse < best[1]:
                best = (lam, mse, acc, alpha)
        return best

    def cv_acc(A, yv, lam, folds=5):
        n = A.shape[0]
        idx = np.arange(n)
        rs = np.random.RandomState(0)
        rs.shuffle(idx)
        accs = []
        for k in range(folds):
            te = idx[k::folds]; tr = np.setdiff1d(idx, te)
            Wf = ridge_fit(A[tr], yv[tr], lam)
            pr = A[te] @ Wf
            accs.append(float(np.mean(pr.argmax(1) == yv[te].argmax(1))))
        return float(np.mean(accs))

    lams = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]

    # ---------------- candidate compositions ----------------
    cands = [
        # n_ovr, n_pair, n_pca, n_err, n_hard, K_h
        (10, 45, 19, 0, 0, 1),
        (10, 30, 26, 0, 0, 1),
        (10, 20, 30, 0, 0, 1),
        (10, 45, 0, 0, 0, 1),
        (10, 30, 10, 10, 0, 1),
        (10, 24, 16, 10, 6, 1),
        (10, 30, 26, 0, 0, 2),
        (10, 20, 20, 0, 0, 2),
        (10, 16, 10, 0, 0, 3),
        (10, 10, 10, 0, 0, 4),
        (10, 45, 19, 0, 0, 0),
    ]

    best = None
    for comp in cands:
        A, rows = build(*comp)
        if A.shape[1] != 128:
            continue
        muA = A.mean(0); sdA = A.std(0)
        sdA = np.where(sdA < 1e-8, 1.0, sdA)
        As = (A - muA) / sdA
        lam, mse, acc, alpha = loo_select(As, Y, lams)
        cacc = cv_acc(As, Y, lam, folds=5)
        key = (cacc, acc, -mse)
        if best is None or key > best[0]:
            best = (key, comp, As, muA, sdA, alpha, lam, mse, acc, cacc, rows)

    key, comp, As, muA, sdA, alpha, lam, mse, acc, cacc, rows = best

    # ---------------- assemble final weights ----------------
    n_rows = len(rows)
    W1n = np.zeros((n_rows, D))
    b1n = np.zeros(n_rows)
    for j, (d, t, kind) in enumerate(rows):
        wo = (d * (1.0 / np.sqrt(evals))) @ U.T
        # p = d . z = (d/sqrt(evals)) . U^T (x - mu_all)  =>  w = wo, offset = -wo.mu_all
        off = -wo @ mu_all
        if kind == 0:
            W1n[j] = wo
            b1n[j] = off - t
        else:
            W1n[j] = wo
            b1n[j] = off - t
    # standardize: feature = (A - muA)/sdA ; A = W1n x + b1n (affine) or relu(W1n x + b1n)
    # fold: W1n' = W1n/sdA ; b1n' = (b1n - muA)/sdA  (for affine rows, exact)
    # for hinge rows the standardization must be applied to the *pre-activation*:
    #   relu(W1n x + b1n) = sdA * relu(W1n x + b1n)/sdA -> we instead scale the row and
    #   keep the same threshold: relu(W1n' x + b1n')*sdA with W1n'=W1n/sdA, b1n'=(b1n-muA)/sdA
    #   is NOT equal in general, so instead scale rows and shift bias by the mean of the
    #   pre-activation: use exact per-unit affine transform on the pre-activation:
    #   g = W1n x + b1n ; want (relu(g)-muA)/sdA = relu((g-muA)/sdA)*? -> only exact for affine.
    # Safe route: keep hinge rows unscaled and absorb scale into fc2 (exact for relu too):
    #   feature_j = (relu(g_j) - muA_j)/sdA_j = relu(g_j)/sdA_j - muA_j/sdA_j
    #   => fc2 column scale 1/sdA_j and fc2 bias absorbs -muA_j/sdA_j * sum of that column.
    # We therefore use: fc1 unchanged (raw pre-activation), and fc2 = alpha^T diag(1/sdA),
    # with bias correction for the constant offsets.
    W2n = alpha.T / sdA[None, :]                      # (10,128)
    const = (muA / sdA)                               # per-unit constant removed
    b2n = b2 - W2n @ const

    # ---------------- sanity: recompute forward pass ----------------
    H = np.maximum(X @ W1n.T + b1n, 0.0)
    L = H @ W2n.T + b2n
    pred = L.argmax(1)
    acc_new = float(np.mean(pred == y))
    p = np.exp(L - L.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    ce = float(-np.mean(np.log(np.maximum(p[np.arange(N), y], 1e-12))))

    def fin(a):
        return np.clip(np.asarray(a, dtype=np.float32), -1000.0, 1000.0)

    out = {
        "fc1.weight": fin(W1n),
        "fc1.bias": fin(b1n),
        "fc2.weight": fin(W2n),
        "fc2.bias": fin(b2n),
    }
    diag = {
        "composition": list(comp),
        "ridge_lambda": float(lam),
        "loo_mse": float(mse),
        "loo_acc": float(acc),
        "cv5_acc": float(cacc),
        "train_acc_after": acc_new,
        "train_ce_after": ce,
        "current_acc": float(np.mean(cur_pred == y)),
        "n_wrong_current": n_wrong,
        "fc1_w_abs_max": float(np.max(np.abs(W1n))),
        "fc2_w_abs_max": float(np.max(np.abs(W2n))),
        "fc1_b_min_max": [float(b1n.min()), float(b1n.max())],
        "fc2_b": [float(v) for v in b2n],
        "n_units": int(n_rows),
        "feature_scale_min_max": [float(sdA.min()), float(sdA.max())],
    }
    return {"weights": out, "diagnostics": diag}
