import numpy as np

def edit(w, f):
    # ---------------- data ----------------
    X = np.asarray(f["x"], dtype=np.float64).reshape(-1, 784)
    y = np.asarray(f["y"], dtype=np.int64).ravel()
    N, D = X.shape
    C = 10
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # ---------------- helpers ----------------
    def _ridge_fit(H, T, lam):
        """closed-form ridge with intercept. H:(n,p) T:(n,k) -> W:(p,k), b:(k,)"""
        n = H.shape[0]
        Hm = H.mean(axis=0)
        Tm = T.mean(axis=0)
        Hc = H - Hm
        Tc = T - Tm
        A = Hc.T @ Hc
        A.flat[:: A.shape[0] + 1] += lam
        W = np.linalg.solve(A, Hc.T @ Tc)
        b = Tm - Hm @ W
        return W, b

    def _press_select(H, T, lams):
        """closed-form leave-one-out selection of ridge lambda via SVD (no training loop)."""
        n, p = H.shape
        Hm = H.mean(axis=0)
        Hc = H - Hm
        Tm = T.mean(axis=0)
        Tc = T - Tm
        U, s, Vt = np.linalg.svd(Hc, full_matrices=False)
        UtT = U.T @ Tc
        best = (np.inf, lams[0])
        for lam in lams:
            d = s ** 2
            filt = d / (d + lam)
            Hd = (U * filt) @ U.T
            h = np.clip(np.diag(Hd), 0.0, 0.999)
            R = Tc - Hd @ Tc
            r = R / (1.0 - h)[:, None]
            err = float(np.mean(r ** 2))
            if err < best[0]:
                best = (err, lam)
        return best[1]

    def _softmax(Z):
        Z = Z - Z.max(axis=1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    rng = np.random.default_rng(12345)

    # ---------------- (A) linear raw-pixel ridge (closed form) ----------------
    lam_lin = 1.0 * N
    Wl, bl = _ridge_fit(X, Y, lam_lin)
    S_lin = X @ Wl + bl

    # ---------------- (B) nonlinear random-feature model ----------------
    # whiten inputs (shrinkage covariance) so random projections have comparable SNR
    mu = X.mean(axis=0)
    Xc = X - mu
    S = (Xc.T @ Xc) / N
    tr = float(np.trace(S)) / D
    Sreg = S + (0.10 * tr) * np.eye(D)
    evals, evecs = np.linalg.eigh(Sreg)
    evals = np.maximum(evals, 1e-8)
    Cinvs = (evecs / evals) @ evecs.T          # (D,D)
    Xw = Xc @ Cinvs                            # whitened inputs
    Z = Xw / np.sqrt(D)                        # unit-variance scale

    # class-informed thresholds: each unit is a soft detector for one class
    M = 128
    cls = np.arange(M) % C
    R = rng.standard_normal((M, D))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    P = Z @ R.T                                # (N,M) random projections
    thr = np.zeros(M)
    for c in range(C):
        idx = np.where(cls == c)[0]
        if idx.size == 0:
            continue
        vals = P[y == c][:, idx]
        q = 0.60 if (c % 2 == 0) else 0.40
        thr[idx] = np.quantile(vals, q, axis=0)
    H = np.maximum(P - thr[None, :], 0.0)     # ReLU features (fc1 output)

    # ---------------- (C) fc2: closed-form ridge on features ----------------
    lams = [1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    lam_nl = _press_select(H, Y, lams)
    W2, b2 = _ridge_fit(H, Y, lam_nl)
    S_nl = H @ W2 + b2

    # ---------------- (D) blend linear + nonlinear (fixed weight) ----------------
    alpha = 0.5
    S_bl = alpha * S_nl + (1.0 - alpha) * S_lin

    # ---------------- diagnostics ----------------
    def _acc(S):
        return float(np.mean(np.argmax(S, axis=1) == y))

    def _ce(S):
        Pm = _softmax(S)
        return float(-np.mean(np.log(np.clip(Pm[np.arange(N), y], 1e-12, None))))

    # simple 5-fold out-of-fold check of the blended predictor (closed-form refits only)
    idx = np.arange(N)
    folds = np.array_split(rng.permutation(idx), 5)
    oof = np.zeros((N, C))
    for fo in folds:
        trn = np.setdiff1d(idx, fo)
        Wl2, bl2 = _ridge_fit(X[trn], Y[trn], lam_lin)
        Sl = X[fo] @ Wl2 + bl2
        Htr = np.maximum(Z[trn] @ R.T - thr[None, :], 0.0)
        W22, b22 = _ridge_fit(Htr, Y[trn], lam_nl)
        Sn = H[fo] @ W22 + b22
        oof[fo] = alpha * Sn + (1.0 - alpha) * Sl

    diag = {
        "lam_linear": float(lam_lin),
        "lam_nonlinear": float(lam_nl),
        "acc_train_linear": _acc(S_lin),
        "acc_train_nonlinear": _acc(S_nl),
        "acc_train_blend": _acc(S_bl),
        "ce_train_blend": _ce(S_bl),
        "acc_oof_blend": float(np.mean(np.argmax(oof, axis=1) == y)),
        "ce_oof_blend": float(-np.mean(np.log(np.clip(
            _softmax(oof)[np.arange(N), y], 1e-12, None)))),
        "feat_mean": float(H.mean()),
        "feat_std": float(H.std()),
        "feat_frac_zero": float((H <= 0).mean()),
        "fc1_absmax": float(np.abs(R @ Cinvs).max()),
        "fc2_absmax": float(np.abs(W2).max()),
    }

    # ---------------- write weights ----------------
    W1 = R @ Cinvs                            # (128,784)
    b1 = thr - (R @ Cinvs) @ mu               # pre-act = W1 x + b1 = P - thr
    w["fc1.weight"] = W1.astype(np.float32)
    w["fc1.bias"] = b1.astype(np.float32)
    w["fc2.weight"] = W2.astype(np.float32)
    w["fc2.bias"] = b2.astype(np.float32)

    return {"weights": w, "diagnostics": diag}
