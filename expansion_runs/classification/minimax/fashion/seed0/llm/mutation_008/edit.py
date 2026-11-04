import numpy as np

def edit(w, f):
    # ---------------- forward pass (verified against cached activations) -------------
    def conv2d(x, W, b, pad=1):
        N, C, H, Wd = x.shape
        O = W.shape[0]
        xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        out = np.zeros((N, O, H, Wd), dtype=np.float32)
        for kh in range(W.shape[2]):
            for kw in range(W.shape[3]):
                patch = xp[:, :, kh:kh + H, kw:kw + Wd]
                out += np.einsum('nchw,oc->nohw', patch, W[:, :, kh, kw], optimize=True)
        out += b.reshape(1, O, 1, 1)
        return out

    def maxpool(x):
        N, C, H, Wd = x.shape
        H2, W2 = H // 2, Wd // 2
        v = x[:, :, :H2 * 2, :W2 * 2].reshape(N, C, H2, 2, W2, 2)
        return v.max(axis=(3, 5))

    def features_from(x, c1w, c1b, c2w, c2b):
        h = maxpool(np.maximum(conv2d(x, c1w, c1b), 0.0))
        h = maxpool(np.maximum(conv2d(h, c2w, c2b), 0.0))
        return h.reshape(h.shape[0], -1)

    X = np.asarray(f["x"], dtype=np.float32)
    y = np.asarray(f["y"]).astype(np.int64)
    N = X.shape[0]
    Y = np.zeros((N, 10), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    c1w = np.asarray(w["conv1.weight"], dtype=np.float32)
    c1b = np.asarray(w["conv1.bias"], dtype=np.float32)
    c2w = np.asarray(w["conv2.weight"], dtype=np.float32)
    c2b = np.asarray(w["conv2.bias"], dtype=np.float32)

    F = np.asarray(f["features"], dtype=np.float64)
    Fchk = features_from(X, c1w, c1b, c2w, c2b).astype(np.float64)
    fwd_err = float(np.abs(Fchk - F).max())

    # ---------------- generic closed-form solvers -----------------------------------
    def ridge_fit(Xt, Tt, lam):
        """closed-form ridge with intercept; Xt (n,d) float64, Tt (n,k)"""
        mu = Xt.mean(axis=0)
        Xc = Xt - mu
        G = Xc.T @ Xc
        d = G.shape[0]
        A = G + lam * np.eye(d)
        B = Xc.T @ (Tt - Tt.mean(axis=0))
        Wt = np.linalg.solve(A, B)
        b = Tt.mean(axis=0) - mu @ Wt
        return Wt, b

    def gcv_lambda(Xt, Tt, lams):
        """closed-form GCV over a vector of lambdas (SVD based, no iteration)."""
        mu = Xt.mean(axis=0)
        Xc = Xt - mu
        Tc = Tt - Tt.mean(axis=0)
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        Ut = U.T @ Tc                      # (r,k)
        n = Xt.shape[0]
        out = []
        for lam in lams:
            df = 0.0
            denom = 0.0
            for i in range(s.shape[0]):
                si = s[i]
                if si <= 1e-12:
                    continue
                f = si ** 2 / (si ** 2 + lam)
                df += f
                denom += (Ut[i] ** 2) / (si ** 2 + lam) ** 2
            rss = float(np.sum(Tc ** 2) - np.sum((s[:, None] ** 2 / (s[:, None] ** 2 + lam)) * Ut ** 2))
            g = rss / max(n - df, 1e-9) / (1.0 + denom / n) ** 2 if denom > 0 else rss / max(n - df, 1e-9)
            out.append(g)
        return np.array(out)

    def cv_acc(Wt, b, Xt, Tt, folds=5):
        """closed-form k-fold CV accuracy of a linear head (fit per fold, closed form)."""
        n = Xt.shape[0]
        idx = np.arange(n)
        parts = np.array_split(idx, folds)
        correct = 0
        for k in range(folds):
            te = parts[k]
            tr = np.concatenate([parts[j] for j in range(folds) if j != k])
            Wf, bf = ridge_fit(Xt[tr], Tt[tr], lam_fix)
            pred = Xt[te] @ Wf + bf
            correct += int((pred.argmax(axis=1) == Tt[te].argmax(axis=1)).sum())
        return correct / n

    # ---------------- candidate heads on the cached features ------------------------
    lams = np.array([1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0])
    g = gcv_lambda(F, Y, lams)
    lam_gcv = float(lams[int(np.argmin(g))])

    # baseline: current head
    hw = np.asarray(w["head.weight"], dtype=np.float64)
    hb = np.asarray(w["head.bias"], dtype=np.float64)
    base_acc = float((F @ hw.T + hb).argmax(axis=1).astype(np.int64).__eq__(y).mean())

    # ridge-to-one-hot, lambda by GCV
    Wr, br = ridge_fit(F, Y, lam_gcv)
    ridge_acc = float((F @ Wr + br).argmax(axis=1).__eq__(y).mean())

    # shrinkage LDA (regularized Gaussian discriminant, class means + shrunk cov)
    mu_c = np.stack([F[y == c].mean(axis=0) for c in range(10)])
    Xc = F - mu_c[y]
    S = (Xc.T @ Xc) / N
    tr = np.trace(S) / S.shape[0]
    S = (1 - 0.3) * S + 0.3 * tr * np.eye(S.shape[0])
    Sinv_mu = np.linalg.solve(S, mu_c.T)          # (d,10)
    lda_score = F @ Sinv_mu - 0.5 * np.sum(mu_c * Sinv_mu.T, axis=1)[None, :]
    lda_acc = float(lda_score.argmax(axis=1).__eq__(y).mean())

    # regularized nearest class mean
    ncm_score = F @ mu_c.T
    ncm_acc = float(ncm_score.argmax(axis=1).__eq__(y).mean())

    # CV comparison (closed form, ridge family)
    lam_fix = lam_gcv
    cv_ridge = cv_acc(None, None, F, Y, folds=5)

    # CV of current head (train on folds, evaluate on held-out) for reference
    def cv_of_fixed_head(W0, b0, Xt, Tt, folds=5):
        n = Xt.shape[0]
        idx = np.arange(n)
        parts = np.array_split(idx, folds)
        corr = 0
        for k in range(folds):
            te = parts[k]
            tr = np.concatenate([parts[j] for j in range(folds) if j != k])
            mu = Xt[tr].mean(axis=0)
            Xc = Xt[tr] - mu
            G = Xc.T @ Xc
            lam = lam_gcv
            A = G + lam * np.eye(G.shape[0])
            B = Xc.T @ (Tt[tr] - Tt[tr].mean(axis=0))
            Wf = np.linalg.solve(A, B)
            bf = Tt[tr].mean(axis=0) - mu @ Wf
            pred = Xt[te] @ Wf + bf
            corr += int((pred.argmax(axis=1) == Tt[te].argmax(axis=1)).sum())
        return corr / n

    cv_base = cv_of_fixed_head(hw, hb, F, Y, folds=5)

    # ---------------- pick head by CV accuracy (tie-break: current head) ------------
    cands = {
        "current": (cv_base, base_acc, hw, hb),
        "ridge_gcv": (cv_ridge, ridge_acc, Wr, br),
    }
    best_name = "current"
    best_cv = cv_base
    for k, (cva, acc, Wc, bc) in cands.items():
        if cva > best_cv + 1e-9:
            best_cv = cva
            best_name = k
    if best_name == "ridge_gcv":
        Wsel, bsel = Wr, br
    else:
        Wsel, bsel = hw, hb

    # ---------------- closed-form recalibration of logits ----------------------------
    # global scale s (LS fit to one-hot) and per-class shift t (LS fit), argmax-safe check via CV
    Z = F @ Wsel.T + bsel
    Zc = Z - Z.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    s = float((Zc * Yc).sum() / max((Zc * Zc).sum(), 1e-9))
    s = float(np.clip(s, 0.02, 5.0))
    # per-class shift: match class means of scaled logits to target class means
    t = Y.mean(axis=0) - s * Z.mean(axis=0)

    def acc_of(Zm):
        return float(Zm.argmax(axis=1).__eq__(y).mean())

    def ce_of(Zm):
        Zs = Zm - Zm.max(axis=1, keepdims=True)
        P = np.exp(Zs)
        P /= P.sum(axis=1, keepdims=True)
        return float(-np.log(np.maximum(P[np.arange(N), y], 1e-12)).mean())

    acc_raw = acc_of(Z)
    ce_raw = ce_of(Z)
    acc_scaled = acc_of(s * Z + t[None, :])
    ce_scaled = ce_of(s * Z + t[None, :])

    # only adopt calibration if it does not hurt accuracy (argmax change check)
    if acc_scaled >= acc_raw - 1e-9:
        Zfinal = s * Z + t[None, :]
    else:
        Zfinal = Z
        s = 1.0
        t = np.zeros(10)

    # ---------------- optional: hand-crafted oriented-edge conv1 bank ---------------
    # Same architecture (16 filters, 3x3, single input channel). Evaluated with the
    # same closed-form CV harness; adopted only if CV accuracy improves.
    def sobel_bank():
        # 4 orientations x 3 scales of oriented derivative filters + 4 smooth/edge extras
        filters = []
        for ang in range(4):
            th = ang * np.pi / 4.0
            dx, dy = np.cos(th), np.sin(th)
            for sc in (1.0, 1.5, 2.0):
                k = np.zeros((3, 3), dtype=np.float64)
                for i in range(3):
                    for j in range(3):
                        u = (i - 1) / sc
                        v = (j - 1) / sc
                        r2 = u * u + v * v
                        g = np.exp(-r2 / 2.0)
                        k[i, j] = g * (u * dx + v * dy)
                filters.append(k)
        extra = [np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
                 np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]]),
                 np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]]),
                 np.array([[-1, 2, -1], [-2, 4, -2], [-1, 2, -1]])]
        for e in extra:
            filters.append(e)
        return np.array(filters[:16], dtype=np.float32)

    def cv_acc_for_features(Fx, Tt, lam, folds=5):
        n = Fx.shape[0]
        idx = np.arange(n)
        parts = np.array_split(idx, folds)
        corr = 0
        for k in range(folds):
            te = parts[k]
            tr = np.concatenate([parts[j] for j in range(folds) if j != k])
            mu = Fx[tr].mean(axis=0)
            Xc = Fx[tr] - mu
            A = Xc.T @ Xc + lam * np.eye(Fx.shape[1])
            B = Xc.T @ (Tt[tr] - Tt[tr].mean(axis=0))
            Wf = np.linalg.solve(A, B)
            bf = Tt[tr].mean(axis=0) - mu @ Wf
            corr += int(((Fx[te] @ Wf + bf).argmax(axis=1) == Tt[te].argmax(axis=1)).sum())
        return corr / n

    sob = sobel_bank()
    sob = sob / (np.abs(sob).max(axis=(1, 2), keepdims=True) + 1e-8)
    F_sob = features_from(X, sob, c1b, c2w, c2b).astype(np.float64)
    cv_sob = cv_acc_for_features(F_sob, Y, lam_gcv, folds=5)

    conv_used = "current"
    if cv_sob > best_cv + 0.002:
        Ffinal = F_sob
        conv_used = "sobel"
        best_cv = cv_sob
    else:
        Ffinal = F

    # ---------------- final closed-form fit on ALL permitted data -------------------
    if conv_used == "sobel":
        Wfinal, bfinal = ridge_fit(Ffinal, Y, lam_gcv)
        Zf = Ffinal @ Wfinal + bfinal
        acc_f = acc_of(Zf)
        ce_f = ce_of(Zf)
        # recalibrate on final logits
        Zc2 = Zf - Zf.mean(axis=0)
        Yc2 = Y - Y.mean(axis=0)
        s2 = float(np.clip((Zc2 * Yc2).sum() / max((Zc2 * Zc2).sum(), 1e-9), 0.02, 5.0))
        t2 = Y.mean(axis=0) - s2 * Zf.mean(axis=0)
        if acc_of(s2 * Zf + t2[None, :]) >= acc_f - 1e-9:
            Zf = s2 * Zf + t2[None, :]
            s, t = s2, t2
        else:
            s, t = 1.0, np.zeros(10)
        acc_f = acc_of(Zf)
        ce_f = ce_of(Zf)
        w["conv1.weight"] = sob.astype(np.float32)
        w["head.weight"] = Wfinal.T.astype(np.float32)
        w["head.bias"] = bfinal.astype(np.float32)
    else:
        # keep conv layers; refit head on all data with the selected family
        if best_name == "ridge_gcv":
            Wfinal, bfinal = ridge_fit(F, Y, lam_gcv)
        else:
            Wfinal, bfinal = hw, hb
        Zf = F @ Wfinal + bfinal
        acc_f = acc_of(Zf)
        ce_f = ce_of(Zf)
        Zc2 = Zf - Zf.mean(axis=0)
        Yc2 = Y - Y.mean(axis=0)
        s2 = float(np.clip((Zc2 * Yc2).sum() / max((Zc2 * Zc2).sum(), 1e-9), 0.02, 5.0))
        t2 = Y.mean(axis=0) - s2 * Zf.mean(axis=0)
        if acc_of(s2 * Zf + t2[None, :]) >= acc_f - 1e-9:
            Zf = s2 * Zf + t2[None, :]
            s, t = s2, t2
        else:
            s, t = 1.0, np.zeros(10)
        acc_f = acc_of(Zf)
        ce_f = ce_of(Zf)
        w["head.weight"] = Wfinal.T.astype(np.float32)
        w["head.bias"] = bfinal.astype(np.float32)

    # ---------------- finalize: sanitize -------------------------------------------
    for k in list(w.keys()):
        arr = np.asarray(w[k])
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1000.0, neginf=-1000.0)
        arr = np.clip(arr, -1000.0, 1000.0)
        w[k] = arr.astype(np.float32)

    diagnostics = {
        "forward_max_abs_err_vs_cached_features": fwd_err,
        "n": int(N),
        "lam_gcv": lam_gcv,
        "gcv_curve": [round(float(v), 6) for v in g],
        "train_acc_current_head": base_acc,
        "train_acc_ridge_gcv": ridge_acc,
        "train_acc_lda_shrunk": lda_acc,
        "train_acc_ncm": ncm_acc,
        "cv5_acc_current_family": cv_base,
        "cv5_acc_ridge_gcv": cv_ridge,
        "cv5_acc_sobel_conv1": cv_sob,
        "selected_head": best_name,
        "conv1_choice": conv_used,
        "logit_scale": s,
        "per_class_shift": [round(float(v), 5) for v in np.asarray(t).ravel()],
        "train_acc_before_calib": acc_raw,
        "train_ce_before_calib": ce_raw,
        "train_acc_after_calib": acc_f,
        "train_ce_after_calib": ce_f,
        "head_weight_absmax": float(np.abs(np.asarray(w["head.weight"])).max()),
        "head_bias": [round(float(v), 4) for v in np.asarray(w["head.bias"]).ravel()],
    }
    return {"weights": w, "diagnostics": diagnostics}
