def edit(w, f):
    import numpy as np

    # ---------------- data ----------------
    X = np.asarray(f["x"], dtype=np.float64).reshape(len(f["x"]), -1)
    N, D = X.shape
    y = np.asarray(f["y"]).astype(np.int64)
    K = 10
    Y = np.zeros((N, K), dtype=np.float64)
    Y[np.arange(N), y] = 1.0
    ybar = Y.mean(0)

    Xm = X.mean(0)
    Xc = X - Xm
    XtX = Xc.T @ Xc
    XtY = Xc.T @ (Y - ybar)

    # eigendecomposition of the (centred) Gram matrix -> cheap ridge family
    evals, evecs = np.linalg.eigh(XtX)
    evals = np.maximum(evals, 0.0)
    VT_XtY = evecs.T @ XtY

    def ridge_W(lam):
        return evecs @ (VT_XtY / (evals + lam)[:, None])

    def ridge_scores(W):
        return (X - Xm) @ W + ybar

    # ---------------- CV over ridge lambda (closed form, no optimisation loop) ---
    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    folds = np.array_split(perm, 5)

    lam_grid = [1e0, 3e0, 1e1, 3e1, 1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5]
    cv_lin = {}
    for lam in lam_grid:
        Wl = ridge_W(lam)
        accs = []
        for k in range(5):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(5) if j != k])
            # refit on the training part only (closed form)
            Xtr = X[tr] - X[tr].mean(0)
            ytr = Y[tr].mean(0)
            A = Xtr.T @ Xtr
            B = Xtr.T @ (Y[tr] - ytr)
            Wk = np.linalg.solve(A + lam * np.eye(D), B)
            sc = (X[te] - X[tr].mean(0)) @ Wk + ytr
            accs.append(float((sc.argmax(1) == y[te]).mean()))
        cv_lin[lam] = float(np.mean(accs))

    lam_star = max(cv_lin, key=lambda L: cv_lin[L])
    W_star = ridge_W(lam_star)
    s_star = ridge_scores(W_star)              # (N,K) linear class scores
    acc_lin_full = float((s_star.argmax(1) == y).mean())

    # ---------------- feature bank (128 hidden units) ----------------
    # unit spec: (direction[784], bias, symmetric?)  symmetric -> +/- pair
    specs = []

    # (1) +/- ridge one-vs-rest directions: reproduces the linear classifier exactly
    for c in range(K):
        d = W_star[:, c].copy()
        b = float(ybar[c] - Xm @ d)
        specs.append((d, b, True))

    # (2) +/- LDA directions (shared-covariance whitened class-mean differences)
    mu = np.zeros((K, D))
    for c in range(K):
        mu[c] = X[y == c].mean(0)
    gm = mu - Xm
    lam_lda = 30.0
    Sig = XtX / N + lam_lda * np.eye(D)
    Ld = np.linalg.solve(Sig, gm.T).T          # (K,D)
    for c in range(K):
        d = Ld[c] / (np.linalg.norm(Ld[c]) + 1e-12)
        proj = X @ d
        specs.append((d, float(-np.median(proj)), True))

    # (3) +/- class-mean directions
    for c in range(K):
        d = gm[c].copy()
        nrm = np.linalg.norm(d)
        if nrm < 1e-9:
            continue
        d = d / nrm
        proj = X @ d
        specs.append((d, float(-np.median(proj)), True))

    # (4) matched-filter units: relu(x.mu_c - 0.5||mu_c||^2)
    for c in range(K):
        d = mu[c].copy()
        nrm2 = float(d @ d)
        specs.append((d, float(-0.5 * nrm2), False))

    # (5) +/- top PCA directions (generic shape, rectified -> nonlinearity)
    order = np.argsort(evals)[::-1]
    for k in range(24):
        d = evecs[:, order[k]].copy()
        proj = X @ d
        specs.append((d, float(-np.median(proj)), True))

    # (6) +/- ridge directions at a second (sharper) regularisation
    W2 = ridge_W(max(lam_star * 0.1, 1.0))
    for c in range(K):
        d = W2[:, c].copy()
        nrm = np.linalg.norm(d)
        if nrm < 1e-12:
            continue
        d = d / nrm
        proj = X @ d
        specs.append((d, float(-np.median(proj)), True))

    # pad / trim to exactly 128 units
    while len(specs) < 64:
        d = evecs[:, order[len(specs) % len(order)]].copy()
        proj = X @ d
        specs.append((d, float(-np.median(proj)), True))
    specs = specs[:64]

    Dm = np.zeros((64, D))
    Bm = np.zeros(64)
    for j, (d, b, sym) in enumerate(specs):
        Dm[j] = d
        Bm[j] = b

    # build the 128 hidden units: symmetric pairs (d,b) and (-d,-b)
    W1 = np.zeros((128, D))
    B1 = np.zeros(128)
    for j in range(64):
        W1[2 * j] = Dm[j]
        B1[2 * j] = Bm[j]
        W1[2 * j + 1] = -Dm[j]
        B1[2 * j + 1] = -Bm[j]

    H = np.maximum(X @ W1.T + B1, 0.0)          # (N,128) hidden features

    # ---------------- closed-form ridge readout with CV over lambda -------------
    Hm = H.mean(0)
    Hc = H - Hm
    HtH = Hc.T @ Hc
    HtY = Hc.T @ (Y - ybar)
    evH, eVH = np.linalg.eigh(HtH)
    evH = np.maximum(evH, 0.0)
    VT_HtY = eVH.T @ HtY

    lam2_grid = [1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0]
    cv2 = {}
    for lam2 in lam2_grid:
        accs = []
        for k in range(5):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(5) if j != k])
            Htr = H[tr]
            mh = Htr.mean(0)
            Hc2 = Htr - mh
            ytr = Y[tr].mean(0)
            A = Hc2.T @ Hc2
            B = Hc2.T @ (Y[tr] - ytr)
            Wk = np.linalg.solve(A + lam2 * np.eye(128), B)
            sc = (H[te] - mh) @ Wk + ytr
            accs.append(float((sc.argmax(1) == y[te]).mean()))
        cv2[lam2] = float(np.mean(accs))

    lam2_star = max(cv2, key=lambda L: cv2[L])
    W2r = eVH @ (VT_HtY / (evH + lam2_star)[:, None])
    b2 = ybar - Hm @ W2r

    # ---------------- write weights ----------------
    w["fc1.weight"] = np.clip(W1.astype(np.float32), -1000, 1000)
    w["fc1.bias"] = np.clip(B1.astype(np.float32), -1000, 1000)
    w["fc2.weight"] = np.clip(W2r.T.astype(np.float32), -1000, 1000)
    w["fc2.bias"] = np.clip(b2.astype(np.float32), -1000, 1000)

    # ---------------- diagnostics ----------------
    sc_full = H @ W2r + b2
    acc_full = float((sc_full.argmax(1) == y).mean())
    per_class = []
    for c in range(K):
        m = (y == c)
        per_class.append(float((sc_full[m].argmax(1) == y[m]).mean()))
    # how well the +/- ridge pairs alone reproduce the linear classifier
    pair_only = H[:, :2 * K] @ np.vstack(
        [np.eye(K), -np.eye(K)]) + np.repeat(ybar[None, :], 2, axis=0).reshape(2, K).T
    acc_pairs = float((pair_only.argmax(1) == y).mean())

    diag = {
        "cv_linear_ridge_acc": {str(k): round(v, 4) for k, v in cv_lin.items()},
        "lam_star_linear": float(lam_star),
        "cv_readout_acc": {str(k): round(v, 4) for k, v in cv2.items()},
        "lam_star_readout": float(lam2_star),
        "cv_acc_final": round(cv2[lam2_star], 4),
        "insample_acc_final": round(acc_full, 4),
        "insample_acc_linear_ridge": round(acc_lin_full, 4),
        "insample_acc_pair_reconstruction": round(acc_pairs, 4),
        "per_class_acc": [round(v, 4) for v in per_class],
        "fc1_weight_std": float(W1.std()),
        "fc1_bias_absmax": float(np.abs(B1).max()),
        "fc2_weight_std": float(W2r.std()),
        "fc2_weight_absmax": float(np.abs(W2r).max()),
        "hidden_active_frac": float((H > 0).mean()),
        "logit_std": float(sc_full.std()),
        "n_units": 128,
    }
    return {"weights": w, "diagnostics": diag}
