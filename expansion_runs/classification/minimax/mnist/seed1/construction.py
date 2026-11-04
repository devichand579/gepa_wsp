def edit(w, f):
    import numpy as np

    X = np.asarray(f["x"], dtype=np.float64).reshape(len(f["x"]), -1)   # (N,784)
    y = np.asarray(f["y"]).astype(np.int64).ravel()
    N, D = X.shape
    C = 10
    Y = np.zeros((N, C))
    Y[np.arange(N), y] = 1.0

    mu = X.mean(axis=0)
    Xc = X - mu

    # ---------------- class moments ----------------
    cls = []
    for c in range(C):
        Xc_c = Xc[y == c]
        cls.append(Xc_c.mean(axis=0))
    M = np.stack(cls, axis=1)                      # (784,10) class means (centered)

    # ---------------- within-class covariance (shrinkage) ----------------
    Sw = np.zeros((D, D))
    for c in range(C):
        Z = Xc[y == c] - M[:, c]
        Sw += Z.T @ Z
    Sw /= max(N - C, 1)
    tr = np.trace(Sw) / D
    Sw += 0.30 * tr * np.eye(D)

    # ---------------- whitened LDA directions ----------------
    # solve Sw @ W = M  (regularized, symmetric positive definite)
    evals, evecs = np.linalg.eigh(Sw)
    evals = np.maximum(evals, 1e-8)
    W_half = evecs @ np.diag(evals ** -0.5) @ evecs.T
    B = W_half @ M                                # (784,10) whitened class means
    # discriminant directions: class c vs mean of others
    Bc = B - B.mean(axis=1, keepdims=True)
    # normalize each direction to unit norm
    nrm = np.linalg.norm(Bc, axis=0, keepdims=True) + 1e-12
    Bc = Bc / nrm

    # ---------------- build feature bank (128 units) ----------------
    # group A: exact linear LDA score per class  -> 2 units/class (relu(s), relu(-s))
    #   s_c(x) = (x-mu) . Bc[:,c]  (unit-norm whitened contrast)
    # group B: thresholded (spline) expansions of the same contrasts
    # group C: whitened random directions (nonlinear diversity)
    rng = np.random.default_rng(0)

    rows = []      # list of (784,) weight rows
    biases = []    # list of float biases
    tags = []

    # --- Group A: 10 classes x 2 units = 20
    for c in range(C):
        d = Bc[:, c]
        rows.append(d.copy()); biases.append(0.0); tags.append(("A+", c))
        rows.append(-d.copy()); biases.append(0.0); tags.append(("A-", c))

    # --- Group B: thresholded expansions of contrasts (10 classes x 8 thresholds = 80)
    # thresholds on the standardized contrast score
    Zs = Xc @ Bc                     # (N,10) contrast scores
    sd = Zs.std(axis=0) + 1e-9
    for c in range(C):
        d = Bc[:, c]
        for q in range(8):
            t = (-1.5 + 3.0 * q / 7.0) * sd[c]
            rows.append(d.copy()); biases.append(-t); tags.append(("B", c))

    # --- Group C: whitened random directions (28 units) for nonlinear capacity
    n_rand = 28
    R = rng.standard_normal((D, n_rand))
    R = W_half @ R
    R /= (np.linalg.norm(R, axis=0, keepdims=True) + 1e-12)
    Zr = Xc @ R
    sdr = Zr.std(axis=0) + 1e-9
    for j in range(n_rand):
        d = R[:, j]
        rows.append(d.copy()); biases.append(-0.5 * sdr[j]); tags.append(("C", j))

    W1 = np.stack(rows, axis=0)                    # (128,784)
    b1 = np.asarray(biases, dtype=np.float64)       # (128,)
    H = np.maximum(Xc @ W1.T + b1, 0.0)            # (N,128) hidden features

    # ---------------- closed-form ridge with LOO-CV lambda ----------------
    Hm = H.mean(axis=0)
    Hc = H - Hm
    Ym = Y.mean(axis=0)
    Yc = Y - Ym

    U, s, Vt = np.linalg.svd(Hc, full_matrices=False)
    UtY = U.T @ Yc
    s2 = s ** 2

    def ridge_solve(lam):
        d = s2 / (s2 + lam)
        A = Vt.T @ (d[:, None] * UtY)             # (128,10)
        return A, d

    # closed-form LOO error for ridge (hat-matrix formula)
    def loo_mse(lam):
        A, d = ridge_solve(lam)
        resid = Yc - Hc @ A
        hdiag = (d ** 2).sum()
        denom = np.maximum(1.0 - hdiag, 1e-6)
        return float((resid ** 2).sum() / (denom ** 2).sum()), A

    lams = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0,
                     100.0, 300.0, 1000.0, 3000.0, 10000.0])
    best = None
    for lam in lams:
        err, A = loo_mse(lam)
        if best is None or err < best[0]:
            best = (err, lam, A)
    loo_err, lam, A = best

    # ---------------- assemble fc2 (with bias) ----------------
    W2 = A.T                                     # (10,128)
    b2 = Ym - Hm @ W2.T                           # (10,)

    # ---------------- verify + diagnostics ----------------
    logits = H @ W2.T + b2
    pred = logits.argmax(axis=1)
    acc = float((pred == y).mean())
    # softmax cross-entropy
    z = logits - logits.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=1))
    ce = float((lse - z[np.arange(N), y]).mean())

    # also report accuracy of the pure linear (group A only) sub-model
    ga = [i for i, t in enumerate(tags) if t[0] in ("A+", "A-")]
    Ha = H[:, ga]
    Aa = np.linalg.lstsq(Ha, Y, rcond=None)[0]
    acc_lin = float((Ha @ Aa).argmax(axis=1).__eq__(y).mean())

    # ---------------- write out ----------------
    out = dict(w)
    out["fc1.weight"] = W1.astype(np.float32)
    out["fc1.bias"] = b1.astype(np.float32)
    out["fc2.weight"] = W2.astype(np.float32)
    out["fc2.bias"] = b2.astype(np.float32)

    # safety: finite, bounded
    for k in ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"):
        arr = out[k]
        arr = np.nan_to_num(arr, nan=0.0, posinf=100.0, neginf=-100.0)
        out[k] = np.clip(arr, -1000.0, 1000.0).astype(np.float32)

    diag = {
        "hypothesis": "closed-form ridge readout over supervised (whitened LDA-contrast) + random ReLU features",
        "lambda": float(lam),
        "loo_mse": float(loo_err),
        "train_acc": acc,
        "train_ce": ce,
        "linear_only_acc": acc_lin,
        "n_features": int(W1.shape[0]),
        "fc1_w_absmax": float(np.abs(out["fc1.weight"]).max()),
        "fc2_w_absmax": float(np.abs(out["fc2.weight"]).max()),
        "feat_mean": float(H.mean()),
        "feat_frac_zero": float((H == 0).mean()),
        "logit_std": float(logits.std()),
    }
    return {"weights": out, "diagnostics": diag}
