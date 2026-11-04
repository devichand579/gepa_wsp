import numpy as np

def edit(w, f):
    # ---------------- data ----------------
    X = f["x"].reshape(-1, 784).astype(np.float64)
    y = np.asarray(f["y"]).astype(np.int64)
    N, D = X.shape
    C = 10
    Xm = X.mean(0)
    Xc = X - Xm
    Y = np.zeros((N, C))
    Y[np.arange(N), y] = 1.0

    # ---------------- within-class scatter ----------------
    Sw = np.zeros((D, D))
    for c in range(C):
        Xc_c = Xc[y == c]
        Sw += Xc_c.T @ Xc_c
    Sw /= max(N - C, 1)
    Sw += 1e-3 * np.trace(Sw) / D * np.eye(D)

    evals, evecs = np.linalg.eigh(Sw)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    keep = evals > max(evals.max(), 1e-12) * 1e-7
    evals, evecs = evals[keep], evecs[:, keep]
    Vw = evecs
    Z = Xc @ Vw / np.sqrt(evals)          # within-class whitened coords

    # ---------------- class means in whitened space ----------------
    mu = np.zeros((C, Z.shape[1]))
    for c in range(C):
        mu[c] = Z[y == c].mean(0)
    gmean = mu.mean(0)
    # regularised LDA direction for class c
    lam = 1e-2 * (np.trace(np.cov(Z, rowvar=False)) / Z.shape[1] + 1e-12)
    Wc = np.zeros((C, Z.shape[1]))
    for c in range(C):
        d = mu[c] - gmean
        Wc[c] = np.linalg.solve(np.cov(Z, rowvar=False) + lam * np.eye(Z.shape[1]), d)
    # scale so scores are comparable across classes
    s_raw = Z @ Wc.T
    sc = s_raw.std(0)
    sc[sc < 1e-9] = 1.0
    Wc = Wc / sc
    b_c = -gmean @ (Wc.T) / 1.0
    b_c = np.array([b_c[c] for c in range(C)])
    # recompute scores with per-class intercepts
    S = Z @ Wc.T + b_c[None, :]
    # recentre each class score to zero mean, unit std (keeps ranking)
    S = (S - S.mean(0)) / S.std(0)
    b_c = -S.mean(0) * 0.0
    # final: s_c(x) = (Z @ Wc.T) + off_c  with off chosen so mean score = 0
    off = -S.mean(0)
    S = Z @ Wc.T + off[None, :]
    # score scale per class (unit std)
    S = S / S.std(0)
    off = off / 1.0

    # map class-score weights back to pixel space:  s_c = xw_c + b_c
    # Z = Xc @ Vw / sqrt(evals)  ->  Z @ Wc.T = Xc @ (Vw diag(1/sqrt(evals)) Wc.T)
    M = Vw @ (Wc.T / np.sqrt(evals)[:, None])      # [D, C]
    M = M / S.std(0)[None, :]                      # normalise per class
    b_lin = off - Xm @ M                           # s_c = x @ M[:,c] + b_lin[c]

    # ---------------- hidden units ----------------
    n_lin = 40
    n_det = 88
    units = []          # (w [D], b)
    lin_idx = []

    # (a) linear copies of class scores (always active) + extra LDA directions
    # extra LDA subspace directions: between-class scatter in whitened space
    Bc = np.cov(Z[y == c].mean(0) - gmean for c in []) if False else None
    mu_d = np.array([Z[y == c].mean(0) for c in range(C)])
    Sb = np.cov(mu_d.T)
    Sb = Sb + 1e-6 * np.eye(Z.shape[1])
    ew, ev = np.linalg.eigh(Sb)
    ev, ew = ev[::-1], ew[:, ::-1]
    # directions in whitened space -> pixel space
    for j in range(n_lin):
        if j < C:
            d = Wc[j]
        else:
            k = j - C
            if k < ew.shape[1]:
                d = ew[:, k]
            else:
                d = np.zeros(Z.shape[1])
                d[k % Z.shape[1]] = 1.0
        wv = Vw @ (d / np.sqrt(evals))
        wv = wv / (np.linalg.norm(wv) + 1e-12)
        units.append((wv, 0.0))
        lin_idx.append(len(units) - 1)

    # (b) rectified class-score detectors: relu(s_c - theta_ck)
    #     thresholds chosen from the empirical quantiles of s_c
    n_thr = 3
    for c in range(C):
        sc_vals = S[:, c]
        qs = np.quantile(sc_vals, [0.15, 0.45, 0.75])
        for th in qs:
            wv = M[:, c].copy()
            wv = wv / (np.linalg.norm(wv) + 1e-12)
            units.append((wv, -th))

    # (c) per-class prototype matched filters at two thresholds
    for c in range(C):
        d = mu[c] - gmean
        d = d / (np.linalg.norm(d) + 1e-12)
        wv = Vw @ (d / np.sqrt(evals))
        wv = wv / (np.linalg.norm(wv) + 1e-12)
        vals = Z @ d
        for th in np.quantile(vals, [0.30, 0.70]):
            units.append((wv, -th))

    # (d) competition units: relu(s_d - s_c) for informative pairs
    #     use pairs (c,d) where c,d differ and both are "hard" classes
    acc_cur = (np.asarray(f["logits"]).argmax(1) == y)
    conf = np.zeros((C, C))
    for c in range(C):
        for d in range(C):
            if c != d:
                conf[c, d] = np.sum((y == c) & (np.asarray(f["logits"]).argmax(1) == d))
    pairs = []
    for c in range(C):
        for d in range(C):
            if c != d and conf[c, d] > 0:
                pairs.append((c, d))
    pairs.sort(key=lambda p: -conf[p[0], p[1]])
    n_comp = n_det - (C * n_thr) - (C * 2)
    for (c, d) in pairs[:max(n_comp, 0)]:
        wv = M[:, c] - M[:, d]
        nrm = np.linalg.norm(wv)
        if nrm < 1e-9:
            continue
        wv = wv / nrm
        units.append((wv, 0.0))

    # trim / pad to exactly 128
    H = 128
    if len(units) > H:
        units = units[:H]
    while len(units) < H:
        units.append((np.zeros(D), 0.0))

    W1 = np.zeros((H, D))
    b1 = np.zeros(H)
    for i, (wv, bv) in enumerate(units):
        W1[i] = wv
        b1[i] = bv
    # make the designated linear units always active (large positive bias)
    scale = np.abs(Xc @ W1[lin_idx].T).max(0) if len(lin_idx) else np.array([1.0])
    for k, i in enumerate(lin_idx):
        b1[i] = 1.0 + 2.0 * scale[k]

    # ---------------- forward (features) ----------------
    def feats(W1, b1):
        A = X @ W1.T + b1
        return np.maximum(A, 0.0)

    Phi = feats(W1, b1)

    # ---------------- ridge readout with closed-form LOO ----------------
    G = Phi.T @ Phi
    Cmat = Phi.T @ Y
    Nf = Phi.shape[0]
    tr = np.trace(G) / G.shape[0]
    lams = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
    best = None
    for lam in lams:
        R = G + lam * tr * np.eye(G.shape[0])
        try:
            Ri = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            continue
        B = Ri @ Cmat
        # LOO via hat matrix diagonal
        Hd = Phi @ Ri @ Phi.T
        dg = np.clip(np.diag(Hd), 0.0, 0.999)
        err = 0.0
        for i in range(Nf):
            r = Y[i] - Phi[i] @ B
            r2 = r - (Phi[i] @ Ri @ (Phi[i] @ B - r)) / (1.0 - dg[i])
            err += float(r2 @ r2)
        if best is None or err < best[0]:
            best = (err, lam, B, Ri)
    err, lam, B, Ri = best

    W2 = B.T.copy()            # [10, H]
    b2 = (Y.mean(0) - Phi.mean(0) @ B.T).copy()

    # ---------------- temperature scaling (argmax invariant) ----------------
    def nll(W2_, b2_, s):
        L = s * (Phi @ W2_.T + b2_)
        L = L - L.max(1, keepdims=True)
        Z_ = np.exp(L)
        P = Z_ / Z_.sum(1, keepdims=True)
        return -np.mean(np.log(np.maximum(P[np.arange(Nf), y], 1e-12)))

    grid = np.exp(np.linspace(np.log(0.5), np.log(40.0), 60))
    vals = [nll(W2, b2, s) for s in grid]
    s_best = float(grid[int(np.argmin(vals))])

    # ---------------- write out ----------------
    w["fc1.weight"] = W1.astype(np.float32)
    w["fc1.bias"] = b1.astype(np.float32)
    w["fc2.weight"] = (W2 * s_best).astype(np.float32)
    w["fc2.bias"] = (b2 * s_best).astype(np.float32)

    # ---------------- diagnostics ----------------
    def forward_np(W1, b1, W2, b2, Xa):
        Hh = np.maximum(Xa @ W1.T + b1, 0.0)
        return Hh @ W2.T + b2

    Ltr = forward_np(W1, b1, W2, b2, X)
    pred = Ltr.argmax(1)
    acc_new = float(np.mean(pred == y))
    Ls = s_best * Ltr
    Ls = Ls - Ls.max(1, keepdims=True)
    P = np.exp(Ls); P = P / P.sum(1, keepdims=True)
    ce_new = float(-np.mean(np.log(np.maximum(P[np.arange(N), y], 1e-12))))
    per_class = [float(np.mean(pred[y == c] == c)) for c in range(C)]
    old_acc = float(np.mean(np.asarray(f["logits"]).argmax(1) == y))
    old_ce = float(np.mean(-np.log(np.maximum(np.asarray(f["probs"])[np.arange(N), y], 1e-12))))

    # batch check
    ids = np.asarray(f["batch.ids"]).astype(np.int64)
    Xb = f["batch.x"].reshape(-1, 784).astype(np.float64)
    yb = np.asarray(f["batch.y"]).astype(np.int64)
    Lb = forward_np(W1, b1, W2, b2, Xb)
    pb = Lb.argmax(1)
    Lbs = s_best * Lb; Lbs = Lbs - Lbs.max(1, keepdims=True)
    Pb = np.exp(Lbs); Pb = Pb / Pb.sum(1, keepdims=True)
    ce_b = float(-np.mean(np.log(np.maximum(Pb[np.arange(len(ids)), yb], 1e-12))))

    diag = {
        "hypothesis": "LDA class-score competition features (relu(s_c-theta), relu(s_d-s_c)) + "
                      "prototype matched filters + always-on affine class-score units; "
                      "fc2 ridge by closed-form LOO; global logit temperature for calibration",
        "old_acc": old_acc, "new_acc": acc_new,
        "old_ce": old_ce, "new_ce": ce_new,
        "ridge_lambda": float(lam), "loo_mse": float(err),
        "temperature": s_best,
        "per_class_acc": per_class,
        "batch_acc": float(np.mean(pb == yb)), "batch_ce": ce_b,
        "n_units": H, "n_lin_units": len(lin_idx),
        "fc1_w_abs_max": float(np.abs(W1).max()),
        "fc2_w_abs_max": float(np.abs(W2 * s_best).max()),
        "fc1_b_min": float(b1.min()), "fc1_b_max": float(b1.max()),
        "feat_zero_frac": float(np.mean(Phi == 0.0)),
        "feat_mean": float(Phi.mean()),
        "n_comp_units": max(n_comp, 0),
    }
    return {"weights": w, "diagnostics": diag}
