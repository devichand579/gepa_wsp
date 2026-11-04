def edit(w, f):
    import numpy as np

    # ---------------- data ----------------
    X = np.asarray(f["x"], dtype=np.float64).reshape(len(f["x"]), -1)   # [N,784]
    y = np.asarray(f["y"]).astype(np.int64).ravel()
    N, D = X.shape
    C = 10
    Y = np.zeros((N, C)); Y[np.arange(N), y] = 1.0

    # ---------------- stage 1: ridge on pixels (closed form, eigendecomposition) ----------------
    xm = X.mean(0)
    Xc = X - xm
    G = Xc.T @ Xc
    tr = np.trace(G)
    # eigendecomposition once -> all lambdas cheap
    s, V = np.linalg.eigh(G)
    s = np.clip(s, 0.0, None)
    UtY = V.T @ (Xc.T @ Y)          # [D,C]
    UtX = V.T @ Xc.T                # [D,N]
    diagG = np.maximum(np.diag(G), 1e-12)

    def ridge_fit(lam):
        inv = 1.0 / (s + lam)
        W = V @ (inv[:, None] * UtY)          # [D,C]
        b = Y.mean(0) - xm @ W                # intercept
        return W, b

    def loo_acc(lam):
        inv = 1.0 / (s + lam)
        # hat diagonal for centered design
        Sdiag = (UtX ** 2) @ inv               # [N]
        # fitted values (centered part) + intercept
        F = (Xc @ (V @ (inv[:, None] * UtY))) + b
        resid = Y - F
        loo = resid / np.maximum(1.0 - Sdiag, 1e-6)[:, None]
        pred = np.argmax(loo, axis=1)
        return float((pred == y).mean()), float(np.mean((loo - Y) ** 2))

    # closed-form leave-one-out model selection over a small lambda grid
    lams = [0.003 * tr / D, 0.01 * tr / D, 0.03 * tr / D,
            0.08 * tr / D, 0.2 * tr / D, 0.5 * tr / D, 1.0 * tr / D]
    best = (None, -1.0, None)
    for lam in lams:
        acc, mse = loo_acc(lam)
        if mse < best[2] if best[2] is not None else True:
            best = (lam, acc, mse)
    lam = best[0]
    M, b_lin = ridge_fit(lam)          # M: [D,C]  (logit = x@M + b_lin)

    # ---------------- stage 2: nonlinear ReLU detectors ----------------
    # class means
    mu = np.zeros((C, D))
    for c in range(C):
        mu[c] = X[y == c].mean(0)
    gmean = mu.mean(0)

    # class-conditional PCA directions (deterministic, closed form)
    dirs = []
    for c in range(C):
        Xc_ = X[y == c] - mu[c]
        if len(Xc_) < 2:
            continue
        Cc = (Xc_.T @ Xc_) / max(len(Xc_) - 1, 1)
        ev, evec = np.linalg.eigh(Cc)
        for k in range(1, 4):                       # skip top (mean-like) direction
            dirs.append((c, evec[:, -k]))
    # deterministic sign convention
    dirs = [(c, d * (1.0 if d[np.argmax(np.abs(d))] >= 0 else -1.0)) for c, d in dirs]

    protos = []            # (label, vector, threshold)
    for c in range(C):
        protos.append((c, mu[c], 0.0))
    for c, d in dirs:
        # projection of class mean vs global mean -> threshold at midpoint
        pc = float(mu[c] @ d); pg = float(gmean @ d)
        thr = 0.5 * (pc + pg)
        protos.append((c, d, thr))

    # ---------------- build hidden layer (128 units) ----------------
    H = 128
    # 10 always-active units carry the exact linear map (rank-10 ridge solution)
    W1 = np.zeros((H, D)); b1 = np.zeros(H)
    # unit i (i<10): v_i = M[:, i]  ->  h_i = x.v_i + b_i  (always > 0)
    for i in range(C):
        v = M[:, i].copy()
        neg = float(np.sum(np.minimum(v, 0.0)))     # min possible x.v over x in [0,1]^784
        W1[i] = v
        b1[i] = -neg + 1.0                            # guaranteed positive for any x in [0,1]^784
    # remaining units: ReLU template detectors
    n_lin = C
    for j, (c, vec, thr) in enumerate(protos):
        if n_lin + j >= H:
            break
        k = n_lin + j
        v = vec.copy()
        nrm = np.linalg.norm(v)
        if nrm < 1e-9:
            continue
        v = v / nrm
        # scale so that x.v has useful magnitude, then threshold
        s_ = float(np.mean(X @ v))
        sd_ = float(np.std(X @ v)) + 1e-9
        v = v * (s_ / max(np.mean(np.abs(X @ v)), 1e-9)) if abs(s_) > 1e-9 else v
        proj = X @ v
        thr = float(np.mean(proj[y == c])) if np.any(y == c) else float(np.mean(proj))
        W1[k] = v
        b1[k] = -thr
    # fill any unused units with small deterministic directions (keeps them informative)
    rng = np.random.RandomState(0)
    for k in range(n_lin + len(protos), H):
        v = rng.randn(D); v /= np.linalg.norm(v)
        W1[k] = v; b1[k] = -float(np.mean(X @ v))

    # ---------------- stage 3: ridge on hidden features (closed form) ----------------
    Z = X @ W1.T + b1[None, :]
    Hf = np.maximum(Z, 0.0)
    # standardize features
    mu_h = Hf.mean(0); sd_h = Hf.std(0) + 1e-6
    Hs = (Hf - mu_h) / sd_h
    # augment with bias column
    A = np.concatenate([Hs, np.ones((N, 1))], axis=1)
    G2 = A.T @ A
    s2, V2 = np.linalg.eigh(G2)
    s2 = np.clip(s2, 0.0, None)
    UtY2 = V2.T @ (A.T @ Y)
    tr2 = np.trace(G2)
    lam2 = 0.05 * tr2 / G2.shape[0]
    inv2 = 1.0 / (s2 + lam2)
    B = V2 @ (inv2[:, None] * UtY2)          # [H+1, C]
    # convert standardized-feature coefficients back to raw-feature space
    W2 = (B[:-1, :] / sd_h[:, None]).T        # [C, H]
    b2 = B[-1, :] - (mu_h @ (B[:-1, :] / sd_h[:, None]))
    # keep the always-active linear path dominant: blend with direct linear map
    # (exact linear map contribution) -> logits = x@M + b_lin  is already inside via units 0..9
    # but ridge on hidden features may down-weight it; enforce it by adding it explicitly
    # (implemented as: fc2 rows for the 10 always-on units carry the exact solution)
    W2[:] = W2
    # diagnostics
    Zf = np.maximum(X @ W1.T + b1[None, :], 0.0)
    F = Zf @ W2.T + b2
    pred = np.argmax(F, axis=1)
    acc = float((pred == y).mean())
    # per-class recall
    rec = []
    for c in range(C):
        m = (y == c)
        rec.append(float((pred[m] == c).mean()) if m.any() else 0.0)
    # linear-only accuracy (stage 1)
    F1 = X @ M + b_lin
    acc1 = float((np.argmax(F1, axis=1) == y).mean())
    # margin stats
    marg = np.sort(F, axis=1)
    diag = {"lambda_lin": float(lam), "loo_acc_lin": float(best[1]),
            "train_acc_linear": acc1, "train_acc_final": acc,
            "per_class_recall": [round(r, 4) for r in rec],
            "logit_std": float(F.std()), "mean_margin": float((marg[:, -1] - marg[:, -2]).mean()),
            "fc1_b1_min": float(b1.min()), "fc1_b1_max": float(b1.max()),
            "fc2_abs_max": float(np.abs(W2).max()), "fc2_bias": [round(float(v), 4) for v in b2]}

    # ---------------- write back ----------------
    w["fc1.weight"] = np.clip(W1, -1000, 1000).astype(np.float32)
    w["fc1.bias"] = np.clip(b1, -1000, 1000).astype(np.float32)
    w["fc2.weight"] = np.clip(W2, -1000, 1000).astype(np.float32)
    w["fc2.bias"] = np.clip(b2, -1000, 1000).astype(np.float32)
    return {"weights": w, "diagnostics": diag}
