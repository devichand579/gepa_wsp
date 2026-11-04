import numpy as np

def edit(w, f):
    # ---------------- data ----------------
    X = np.asarray(f["x"], dtype=np.float64).reshape(-1, 784)
    y = np.asarray(f["y"]).astype(np.int64)
    N, D = X.shape
    K = 10
    Y = np.zeros((N, K))
    Y[np.arange(N), y] = 1.0

    fc1w = np.asarray(w["fc1.weight"], dtype=np.float64)
    fc1b = np.asarray(w["fc1.bias"], dtype=np.float64)
    fc2w = np.asarray(w["fc2.weight"], dtype=np.float64)
    fc2b = np.asarray(w["fc2.bias"], dtype=np.float64)

    # ---------------- class statistics ----------------
    mu = np.zeros((K, D))
    for c in range(K):
        m = (y == c)
        mu[c] = X[m].mean(axis=0)
    Sw = np.zeros((D, D))
    for c in range(K):
        Z = X[y == c] - mu[c]
        Sw += Z.T @ Z
    Sw /= max(N - K, 1)
    St = np.cov(X, rowvar=False)
    St = np.atleast_2d(St)

    def top_dirs(C, k):
        C = 0.5 * (C + C.T)
        ev, V = np.linalg.eigh(C)
        idx = np.argsort(ev)[::-1][:k]
        return V[:, idx], np.maximum(ev[idx], 1e-8)

    # ---------------- feature builders ----------------
    def hinge_pairs(dirs, thresh):
        """dirs: (m,D) unit-norm directions; thresh: (m,) thresholds.
        returns fc1 weight (2m,D), bias (2m,) and feature matrix (N,2m)."""
        m = dirs.shape[0]
        W = np.zeros((2 * m, D))
        b = np.zeros(2 * m)
        W[0::2] = dirs
        W[1::2] = -dirs
        b[0::2] = -thresh
        b[1::2] = thresh
        Z = X @ W.T + b
        H = np.maximum(Z, 0.0)
        return W, b, H

    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else v

    def quant_thresh(s, q):
        return float(np.quantile(s, q))

    def build_pairwise():
        # pairwise LDA directions mu_c - mu_c' with a hinge at the pair boundary
        dirs, ths = [], []
        for c in range(K):
            for c2 in range(c + 1, K):
                d = unit(mu[c] - mu[c2])
                sc = X[y == c] @ d
                sc2 = X[y == c2] @ d
                t = 0.5 * (sc.mean() + sc2.mean())
                dirs.append(d)
                ths.append(t)
        Dm = np.array(dirs)
        Tm = np.array(ths)
        W, b, H = hinge_pairs(Dm, Tm)
        # fill remaining capacity with class-template hinges
        used = W.shape[0]
        rem = 128 - used
        extra_d, extra_t = [], []
        for c in range(K):
            d = unit(mu[c])
            s = X @ d
            extra_d.append(d)
            extra_t.append(quant_thresh(s, 0.5))
            if len(extra_d) * 2 + used >= 128:
                break
        if extra_d:
            W2, b2, H2 = hinge_pairs(np.array(extra_d), np.array(extra_t))
            W = np.concatenate([W, W2], 0)
            b = np.concatenate([b, b2], 0)
            H = np.concatenate([H, H2], 1)
        if W.shape[0] < 128:  # pad with PCA hinge pairs
            Vp, _ = top_dirs(St, 64)
            need = (128 - W.shape[0]) // 2
            d = Vp[:, :need].T
            s = X @ d.T
            t = np.quantile(s, 0.5, axis=0)
            W2, b2, H2 = hinge_pairs(d, t)
            W = np.concatenate([W, W2], 0)
            b = np.concatenate([b, b2], 0)
            H = np.concatenate([H, H2], 1)
        return W, b, H

    def build_template_spline():
        # class templates, several thresholds each -> spline in template score
        dirs, ths = [], []
        for c in range(K):
            d = unit(mu[c])
            s = X @ d
            for q in (0.2, 0.4, 0.6, 0.8):
                dirs.append(d)
                ths.append(quant_thresh(s, q))
        Dm = np.array(dirs)
        Tm = np.array(ths)
        W, b, H = hinge_pairs(Dm, Tm)
        used = W.shape[0]
        if used < 128:
            Vp, _ = top_dirs(St, 64)
            need = (128 - used) // 2
            d = Vp[:, :need].T
            s = X @ d.T
            t = np.quantile(s, 0.5, axis=0)
            W2, b2, H2 = hinge_pairs(d, t)
            W = np.concatenate([W, W2], 0)
            b = np.concatenate([b, b2], 0)
            H = np.concatenate([H, H2], 1)
        return W, b, H

    def build_pca_hinge():
        Vp, _ = top_dirs(St, 64)
        d = Vp.T
        s = X @ d.T
        t = np.quantile(s, 0.5, axis=0)
        W, b, H = hinge_pairs(d, t)
        return W, b, H

    def build_current():
        H = np.asarray(f["features"], dtype=np.float64)
        return fc1w.copy(), fc1b.copy(), H

    # ---------------- closed-form ridge + LOO lambda ----------------
    def ridge_fit(H, lam):
        A = H.T @ H + lam * np.eye(H.shape[1])
        B = H.T @ Y
        Wt = np.linalg.solve(A, B)
        b = Y.mean(axis=0) - H.mean(axis=0) @ Wt
        return Wt, b

    def loo_mse(H, lam):
        A = H.T @ H + lam * np.eye(H.shape[1])
        B = H.T @ Y
        Ainv = np.linalg.inv(A)
        Wt = Ainv @ B
        Hinv = np.linalg.inv(H.T @ H + 1e-8 * np.eye(H.shape[1]))
        # hat matrix diagonal
        Hd = np.einsum('ij,jk,ik->i', H, Hinv, H)
        h = 1.0 - Hd
        h = np.maximum(h, 1e-6)
        R = Y - H @ Wt
        loo = (R / h[:, None]) ** 2
        return float(loo.mean()), Wt

    def cv_acc(H, lam, folds=5):
        idx = np.arange(N)
        acc = 0.0
        for k in range(folds):
            te = idx[idx % folds == k]
            tr = idx[idx % folds != k]
            Wt, b = ridge_fit(H[tr], lam)
            pred = H[te] @ Wt + b
            acc += float((pred.argmax(1) == y[te]).sum())
        return acc / N

    def pick_lambda(H):
        lams = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
        best = (None, -1.0)
        for lam in lams:
            a = cv_acc(H, lam)
            if a > best[1]:
                best = (lam, a)
        return best[0], best[1]

    # ---------------- candidate comparison (closed-form fits only) ----------------
    builders = [build_current, build_pairwise, build_template_spline, build_pca_hinge]
    results = []
    for bi, bf in enumerate(builders):
        try:
            Wc, bc, H = bf()
            if H.shape[1] != 128:
                continue
            lam, acc = pick_lambda(H)
            results.append((acc, bi, Wc, bc, H, lam))
        except Exception:
            continue

    if not results:
        return w

    results.sort(key=lambda t: -t[0])
    best_acc, best_i, Wc, bc, H, lam = results[0]

    # final closed-form fit on the whole pool with the selected lambda
    Wt, b = ridge_fit(H, lam)

    # ---------------- write out ----------------
    out = {k: np.array(v, dtype=np.float32, copy=True) for k, v in w.items()}
    out["fc1.weight"] = np.clip(Wc, -1000.0, 1000.0).astype(np.float32)
    out["fc1.bias"] = np.clip(bc, -1000.0, 1000.0).astype(np.float32)
    out["fc2.weight"] = np.clip(Wt.T, -1000.0, 1000.0).astype(np.float32)
    out["fc2.bias"] = np.clip(b, -1000.0, 1000.0).astype(np.float32)

    # diagnostics
    pred = (H @ Wt + b).argmax(1)
    acc_in = float((pred == y).mean())
    ce = float(-np.log(np.clip((H @ Wt + b)[np.arange(N), y], 1e-12, None)).mean())
    cand = []
    for a, i, _, _, _, l in sorted(results, key=lambda t: -t[0]):
        cand.append({"builder": i, "cv_acc": round(float(a), 4), "lam": float(l)})
    diagnostics = {
        "chosen_builder": int(best_i),
        "chosen_cv_acc": round(float(best_acc), 4),
        "chosen_lambda": float(lam),
        "train_acc_after": round(acc_in, 4),
        "train_ce_after": round(ce, 4),
        "candidates": cand,
        "fc1_w_abs_max": float(np.abs(Wc).max()),
        "fc1_b_min_max": [float(bc.min()), float(bc.max())],
        "fc2_w_abs_max": float(np.abs(Wt).max()),
        "fc2_b": [round(float(v), 4) for v in b],
        "n_train": int(N),
        "feature_zero_frac": round(float((H == 0).mean()), 4),
    }
    return {"weights": out, "diagnostics": diagnostics}
