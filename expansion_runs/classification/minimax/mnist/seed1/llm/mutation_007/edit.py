import numpy as np

def edit(w, f):
    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    X = np.asarray(f["x"], dtype=np.float64).reshape(-1, 784)
    y = np.asarray(f["y"]).astype(np.int64).ravel()
    N, D = X.shape
    C = 10
    Y1 = np.zeros((N, C), dtype=np.float64)
    Y1[np.arange(N), y] = 1.0

    mu_x = X.mean(axis=0)
    Xc = X - mu_x

    # ------------------------------------------------------------------
    # Whitened PCA subspace (closed form eigendecomposition)
    # ------------------------------------------------------------------
    Cov = (Xc.T @ Xc) / N
    evals, evecs = np.linalg.eigh(Cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    d = 200
    d = min(d, D)
    lam = evals[:d]
    V = evecs[:, :d]
    shrink = 1e-2
    scale = 1.0 / np.sqrt(lam + shrink * lam.mean() + 1e-12)
    Z = (Xc @ V) * scale                      # N x d whitened scores
    Zs = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)

    # ------------------------------------------------------------------
    # Class templates (permitted class moments)
    # ------------------------------------------------------------------
    mus = np.zeros((C, D), dtype=np.float64)
    for c in range(C):
        m = (y == c)
        if m.any():
            mus[c] = X[m].mean(axis=0)
    mu_n = mus / (np.linalg.norm(mus, axis=1, keepdims=True) + 1e-12)
    Tsim = Zs @ (mu_n @ V).T                   # N x C cosine-like similarity
    Tsim = (Tsim - Tsim.mean(0)) / (Tsim.std(0) + 1e-12)

    # ------------------------------------------------------------------
    # Candidate hidden-layer constructions
    # ------------------------------------------------------------------
    rng = np.random.default_rng(0)

    def build_random(n_units, seed):
        r = np.random.default_rng(seed)
        A = r.standard_normal((n_units, d))
        A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        # random bias thresholds spanning a range of activation rates
        b = r.uniform(-1.2, 1.2, size=n_units)
        return A, b

    def build_template(n_units):
        # rectified class-similarity features: thresholds per class
        rows = []
        n_cls = min(C, n_units)
        per = max(1, n_units // n_cls)
        ths = [-1.5, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        for c in range(n_cls):
            for k in range(per):
                t = ths[k % len(ths)]
                rows.append((c, t))
        rows = rows[:n_units]
        W = np.zeros((len(rows), C), dtype=np.float64)
        b = np.zeros(len(rows), dtype=np.float64)
        for i, (c, t) in enumerate(rows):
            W[i, c] = 1.0
            b[i] = -t
        return W, b

    def combine(n_rand, n_tmpl, seed):
        parts = []
        if n_rand > 0:
            A, bb = build_random(n_rand, seed)
            parts.append((A, bb, "rand"))
        if n_tmpl > 0:
            Wt, bt = build_template(n_tmpl)
            parts.append((Wt, bt, "tmpl"))
        W1 = np.zeros((128, D), dtype=np.float64)
        b1 = np.zeros(128, dtype=np.float64)
        idx = 0
        for (M, bb, kind) in parts:
            for i in range(M.shape[0]):
                if kind == "rand":
                    W1[idx] = M[i] * scale @ V.T
                    b1[idx] = bb[i]
                else:
                    W1[idx] = M[i] @ mus
                    b1[idx] = bb[i]
                idx += 1
        return W1, b1

    def forward(W1, b1, Xd):
        H = Xd @ W1.T + b1
        np.maximum(H, 0.0, out=H)
        return H

    def ridge_fit(H, T, lam):
        G = H.T @ H
        dH = G.shape[0]
        scale_g = np.trace(G) / max(dH, 1)
        A = G + lam * scale_g * np.eye(dH)
        B = H.T @ T
        Wout = np.linalg.solve(A, B)
        b0 = T.mean(axis=0) - (H.mean(axis=0) @ Wout)
        return Wout, b0

    def acc_on(H, T, y, Wout, b0):
        S = H @ Wout + b0
        return float((S.argmax(1) == y).mean())

    # ------------------------------------------------------------------
    # Honest split-based selection among closed-form constructions
    # ------------------------------------------------------------------
    idx = np.arange(N)
    rng2 = np.random.default_rng(7)
    rng2.shuffle(idx)
    n_tr = 1500
    tr, te = idx[:n_tr], idx[n_tr:]

    cands = []
    for (nr, nt, seed) in [(128, 0, 1), (96, 32, 2), (64, 64, 3), (32, 96, 4),
                            (112, 16, 5), (80, 48, 6), (48, 80, 7), (128, 0, 11),
                            (96, 32, 12), (64, 64, 13)]:
        cands.append((nr, nt, seed))

    best = None
    for (nr, nt, seed) in cands:
        W1, b1 = combine(nr, nt, seed)
        Htr = forward(W1, b1, X[tr])
        Hte = forward(W1, b1, X[te])
        # small grid of ridge strengths, selected on the *training* split only
        for lam in [1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
            Wout, b0 = ridge_fit(Htr, Y1[tr], lam)
            a_tr = acc_on(Htr, Y1[tr], y[tr], Wout, b0)
            if best is None or a_tr > best[0]:
                best = (a_tr, nr, nt, seed, lam)
    _, nr, nt, seed, lam = best

    # ------------------------------------------------------------------
    # Final closed-form fit on the whole permitted pool
    # ------------------------------------------------------------------
    W1, b1 = combine(nr, nt, seed)
    H = forward(W1, b1, X)

    # analytic GCV over a small ridge grid (closed form, no optimizer loop)
    n = N
    best_gcv = None
    for lg in [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1]:
        lam = 10.0 ** lg
        Wout, b0 = ridge_fit(H, Y1, lam)
        S = H @ Wout + b0
        r = Y1 - S
        rss = float((r ** 2).sum())
        # effective dof via hat matrix trace (SVD based, closed form)
        U, sv, Vt = np.linalg.svd(H, full_matrices=False)
        sv2 = sv ** 2
        tr_h = float((sv2 / (sv2 + lam * (np.trace(H.T @ H) / H.shape[1]))).sum())
        denom = (1.0 - tr_h / n) ** 2
        if denom <= 1e-9:
            continue
        gcv = rss / n / denom
        if best_gcv is None or gcv < best_gcv[0]:
            best_gcv = (gcv, lg, Wout, b0, tr_h, rss)

    gcv, lg, Wout, b0, tr_h, rss = best_gcv
    S = H @ Wout + b0
    acc = float((S.argmax(1) == y).mean())

    # ------------------------------------------------------------------
    # Write back
    # ------------------------------------------------------------------
    out = {}
    for k, v in w.items():
        out[k] = np.asarray(v, dtype=np.float32).copy()

    out["fc1.weight"] = np.clip(W1.T.astype(np.float32), -1000.0, 1000.0)  # [784,128] -> transposed below
    out["fc1.weight"] = np.clip(W1.astype(np.float32), -1000.0, 1000.0)    # [128,784]
    out["fc1.bias"] = np.clip(b1.astype(np.float32), -1000.0, 1000.0)      # [128]
    out["fc2.weight"] = np.clip(Wout.T.astype(np.float32), -1000.0, 1000.0) # [10,128]
    out["fc2.bias"] = np.clip(b0.astype(np.float32), -1000.0, 1000.0)       # [10]

    for k in out:
        out[k] = np.nan_to_num(out[k], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # sanity check with the actual architecture
    Xf = np.asarray(f["x"], dtype=np.float32).reshape(-1, 784)
    Hf = np.maximum(Xf @ out["fc1.weight"].T + out["fc1.bias"], 0.0)
    Sf = Hf @ out["fc2.weight"].T + out["fc2.bias"]
    acc_check = float((Sf.argmax(1) == y).mean())

    diagnostics = {
        "selected": {"n_random_units": int(nr), "n_template_units": int(nt),
                     "seed": int(seed), "split_best_train_acc": float(best[0]),
                     "ridge_log10": float(lg)},
        "gcv_ridge_log10": float(lg),
        "gcv_value": float(gcv),
        "train_acc_in_sample": float(acc),
        "train_acc_recomputed": float(acc_check),
        "effective_dof": float(tr_h),
        "rss": float(rss),
        "n_samples": int(N),
        "pca_dims": int(d),
        "fc1_weight_abs_max": float(np.abs(out["fc1.weight"]).max()),
        "fc2_weight_abs_max": float(np.abs(out["fc2.weight"]).max()),
        "feature_mean": float(H.mean()),
        "feature_std": float(H.std()),
        "feature_zero_frac": float((H == 0).mean()),
        "logit_std": float(Sf.std()),
    }
    return {"weights": out, "diagnostics": diagnostics}
