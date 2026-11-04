import numpy as np

def edit(w, f):
    # ---------------- data ----------------
    X = f["x"].reshape(-1, 784).astype(np.float64)      # [N,784]
    y = f["y"].astype(np.int64)
    N, D = X.shape
    K = 10

    # ---------------- class means / within-class covariance ----------------
    Xc = X - X.mean(0, keepdims=True)
    Sw = np.zeros((D, D))
    Mc = np.zeros((K, D))
    for c in range(K):
        m = y == c
        Xc_c = X[m] - X[m].mean(0, keepdims=True)
        Sw += Xc_c.T @ Xc_c
        Mc[c] = X[m].mean(0)
    Sw /= max(N - K, 1)
    tr = np.trace(Sw) / D

    # ---------------- whitened supervised projection (LDA directions) --------
    # shrink covariance for stability
    lam_shrink = 0.30
    Sw_r = (1.0 - lam_shrink) * Sw + lam_shrink * tr * np.eye(D)
    # solve Sw_r^{-1} M^T  (M: class means, centered)
    M = Mc - Mc.mean(0, keepdims=True)
    try:
        L = np.linalg.cholesky(Sw_r)
        W = np.linalg.solve(L.T, np.linalg.solve(L, M.T)).T   # [K,784]
    except np.linalg.LinAlgError:
        W = M @ np.linalg.pinv(Sw_r)

    # ---------------- build hidden units (fc1) ----------------------------
    # layout: 2 units/class for exact linear score (LDA), then hinge units
    # for convex PL quadratic features along extra directions.
    units = []      # list of (weight[784], bias)
    for c in range(K):
        units.append((W[c].copy(), 0.0))          # relu(+s_c)
        units.append((-W[c].copy(), 0.0))         # relu(-s_c)
    n_lin = len(units)                            # 20

    # extra directions: whitened random + supervised residual directions
    rng = np.random.default_rng(0)
    Wn = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-9)
    # directions: random in whitened space (mix of class-mean dirs and random)
    dirs = []
    # supervised combos (differences of class means) -> discriminative nonlinear probes
    for _ in range(18):
        a, b = rng.integers(0, K, 2)
        d = Wn[a] - Wn[b]
        dirs.append(d)
    # random directions in whitened metric: use random rows of Sw^{-1/2}-ish
    # (approximate whitening via Cholesky of Sw_r: draw z ~ N(0,I), map to L^{-T})
    L = np.linalg.cholesky(Sw_r)
    for _ in range(14):
        z = rng.normal(size=D)
        z = z / np.linalg.norm(z)
        v = np.linalg.solve(L.T, z)
        dirs.append(v)
    # normalize each direction in the whitened metric for stable hinge ranges
    dirs = np.array(dirs)
    dirs = dirs / np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-9)

    # hinge thresholds per direction (convex PL approximation of s^2 on [-R,R])
    R = 2.5
    K_h = 4
    hinge_units = []
    for v in dirs:
        s = X @ v
        lo, hi = np.percentile(s, 1.0), np.percentile(s, 99.0)
        lo = min(lo, -1e-6); hi = max(hi, 1e-6)
        ts = np.linspace(lo, hi, K_h + 1)[1:-1]
        for t in ts:
            hinge_units.append((v.copy(), -float(t)))
    units.extend(hinge_units)
    H = len(units)                                # 20 + 32*3 = 116

    # ---------------- assemble fc1 ----------------------------------------
    W1 = np.zeros((H, D))
    b1 = np.zeros(H)
    for j, (wv, bv) in enumerate(units):
        W1[j] = wv
        b1[j] = bv

    # ---------------- compute features with the FINAL fc1 ------------------
    A = X @ W1.T + b1
    Phi = np.maximum(A, 0.0)                       # [N,H]

    # ---------------- closed-form ridge to one-hot targets ----------------
    # class-balanced targets: each class contributes equally (counts are equal here)
    Y = np.zeros((N, K))
    Y[np.arange(N), y] = 1.0
    cnt = np.bincount(y, minlength=K).astype(np.float64)
    Y = Y * (N / (K * np.maximum(cnt, 1)))[:, None]

    # standardize columns (helps conditioning); keep scales to unstandardize later
    mu = Phi.mean(0)
    sd = Phi.std(0)
    sd[sd < 1e-8] = 1.0
    Phi_s = (Phi - mu) / sd

    # ridge with a few candidate lambdas -> pick by closed-form GCV-free proxy:
    # use leave-one-out via hat-matrix (closed form) on the ridge solution.
    n, p = Phi_s.shape
    G = Phi_s.T @ Phi_s
    c = Phi_s.T @ Y
    trG = np.trace(G)
    best = None
    for lam in [1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]:
        Areg = G + lam * np.eye(p)
        try:
            Ai = np.linalg.inv(Areg)
        except np.linalg.LinAlgError:
            continue
        B = Ai @ c
        Hhat = Phi_s @ Ai @ Phi_s.T
        # closed-form LOO residuals for ridge (PRESS)
        d = 1.0 - np.diag(Hhat)
        d = np.where(np.abs(d) < 1e-9, 1e-9, d)
        pred = Phi_s @ B
        resid = Y - pred
        loo = (resid / d[:, None]) ** 2
        err = float(loo.sum())
        if best is None or err < best[0]:
            best = (err, lam, B, Ai)
    err, lam, B, Ai = best

    W2_s = B.T                                     # [K,p] standardized space
    W2 = (W2_s / sd).T                            # back to raw feature space
    b2 = Y.mean(0) - W2_s.T @ mu                   # bias from standardization
    # note: b2 = mean(Y) - W2_s @ mu  (per class)

    # ---------------- diagnostics ----------------------------------------
    pred = Phi @ W2.T + b2
    predc = pred.argmax(1)
    acc = float((predc == y).mean())
    per_class = [float((predc[y == c] == c).mean()) for c in range(K)]
    # in-sample (train) accuracy
    tr_acc = acc
    # held-out-ish estimate via LOO classification accuracy (closed form)
    Hhat = Phi_s @ Ai @ Phi_s.T
    d = 1.0 - np.diag(Hhat)
    d = np.where(np.abs(d) < 1e-9, 1e-9, d)
    loo_pred = (Phi_s @ B - (Y - Phi_s @ B) / d[:, None]).argmax(1)
    loo_acc = float((loo_pred == y).mean())
    # logit scale diagnostics
    lg = Phi @ W2.T + b2
    ce = float(np.mean(np.log(np.exp(lg - lg.max(1, keepdims=True)).sum(1)) +
                         (lg.max(1) - lg[np.arange(N), y])))
    diag = {
        "train_acc": round(acc, 4),
        "loo_acc": round(loo_acc, 4),
        "ridge_lambda": lam,
        "n_units": H,
        "n_linear_units": n_lin,
        "n_hinge_units": len(hinge_units),
        "per_class_acc": [round(v, 3) for v in per_class],
        "train_ce": round(ce, 4),
        "logit_std": round(float(lg.std()), 4),
        "feat_mean": round(float(Phi.mean()), 4),
        "feat_std": round(float(Phi.std()), 4),
        "active_frac": round(float((Phi > 0).mean()), 4),
        "fc1_abs_max": round(float(np.abs(W1).max()), 4),
        "fc2_abs_max": round(float(np.abs(W2).max()), 4),
        "sw_trace_per_dim": round(float(tr), 5),
        "lda_row_norm": [round(float(v), 3) for v in np.linalg.norm(W, axis=1)],
    }

    # ---------------- write back (float32, finite, |.|<=1000) --------------
    def clean(a):
        a = np.asarray(a, dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(a, -1000.0, 1000.0)

    w["fc1.weight"] = clean(W1)
    w["fc1.bias"] = clean(b1)
    w["fc2.weight"] = clean(W2)
    w["fc2.bias"] = clean(b2)
    return {"weights": w, "diagnostics": diag}
