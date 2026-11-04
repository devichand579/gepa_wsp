def edit(w, f):
    import numpy as np

    # ---------- helpers ----------
    def conv2d(x, W, b, pad=1):
        # x: (N,C,H,W), W: (O,C,kh,kw) -> (N,O,H,W)
        N, C, H, Wd = x.shape
        O, C2, kh, kw = W.shape
        if pad > 0:
            xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        else:
            xp = x
        Hp, Wp = xp.shape[2], xp.shape[3]
        cols = np.empty((N, C, kh, kw, Hp - kh + 1, Wp - kw + 1), dtype=np.float64)
        for i in range(kh):
            for j in range(kw):
                cols[:, :, i, j, :, :] = xp[:, :, i:i + Hp - kh + 1, j:j + Wp - kw + 1]
        out = np.einsum('ncijkl,ockl->noij', cols, W.astype(np.float64), optimize=True)
        out += b.astype(np.float64).reshape(1, O, 1, 1)
        return out

    def maxpool2(x):
        N, C, H, Wd = x.shape
        H2, W2 = H // 2, Wd // 2
        r = x[:, :, :H2 * 2, :W2 * 2].reshape(N, C, H2, 2, W2, 2)
        return r.max(axis=(3, 5))

    def forward_feats(x, w):
        h = maxpool2(np.maximum(conv2d(x, w["conv1.weight"], w["conv1.bias"]), 0.0))
        h = maxpool2(np.maximum(conv2d(h, w["conv2.weight"], w["conv2.bias"]), 0.0))
        return h.reshape(h.shape[0], -1)

    diag = {}

    # ---------- data ----------
    X = np.asarray(f["x"], dtype=np.float64)
    y = np.asarray(f["y"]).astype(np.int64)
    N = X.shape[0]
    K = 10

    # ---------- recompute features from CURRENT weights (self-consistency) ----------
    feats = forward_feats(X, w)
    cached = np.asarray(f["features"], dtype=np.float64)
    diag["feat_max_abs_diff_vs_cached"] = float(np.max(np.abs(feats - cached)))
    diag["feat_mean_abs_diff_vs_cached"] = float(np.mean(np.abs(feats - cached)))
    diag["feat_rel_diff"] = float(np.mean(np.abs(feats - cached)) / (np.mean(np.abs(cached)) + 1e-12))

    # sanity: cached logits vs head applied to cached features
    hw = np.asarray(w["head.weight"], dtype=np.float64)
    hb = np.asarray(w["head.bias"], dtype=np.float64)
    lg = np.asarray(f["logits"], dtype=np.float64)
    diag["logit_max_abs_diff_vs_cached"] = float(np.max(np.abs(cached @ hw.T + hb - lg)))
    diag["current_train_acc"] = float(np.mean(lg.argmax(1) == y))

    # ---------- closed-form ridge head fit on the feedback pool ----------
    Y = np.zeros((N, K), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    mu = feats.mean(axis=0)
    Xc = feats - mu
    G = Xc.T @ Xc                      # (D,D)
    B = Xc.T @ Y                       # (D,K)
    tr = np.trace(G)
    D = G.shape[0]
    diag["D"] = int(D)
    diag["trace_G"] = float(tr)
    diag["rank_G"] = int(np.linalg.matrix_rank(G, tol=1e-8 * max(tr, 1.0)))

    # eigendecomposition once -> all lambdas cheap (closed form)
    evals, evecs = np.linalg.eigh(G)
    evals = np.clip(evals, 0.0, None)
    Bt = evecs.T @ B                  # (D,K)

    def solve(lam):
        d = evals + lam
        Wt = Bt / d[:, None]
        Wc = evecs @ Wt                 # (D,K) weights on centered features
        b = Y.mean(axis=0) - mu @ Wc
        return Wc, b

    def loo_mse(lam):
        # analytic leave-one-out for ridge (Hutchinson / PRESS)
        d = evals + lam
        Wt = Bt / d[:, None]
        # hat matrix diagonal: h_ii = sum_j x_ij^2 * (evec_ij^2 / d_j)
        U = evecs * (1.0 / np.sqrt(d))       # (D,D)
        Z = Xc @ U                          # (N,D)
        h = np.sum(Z * Z, axis=1)            # (N,)
        h = np.clip(h, 0.0, 0.999999)
        resid = Y - (Xc @ Wt) - Y.mean(axis=0)[None, :]
        r = resid / (1.0 - h)[:, None]
        return float(np.mean(r * r)), h

    lams = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]
    best = None
    for lam in lams:
        Wc, b = solve(lam)
        pred = Xc @ Wc + b
        acc = float(np.mean(pred.argmax(1) == y))
        lm, _ = loo_mse(lam)
        diag.setdefault("lam_scan", []).append([float(lam), round(acc, 4), round(lm, 5)])
        if best is None or lm < best[0]:
            best = (lm, lam, acc)

    lam = best[1]
    Wc, b = solve(lam)
    diag["chosen_lambda"] = float(lam)
    diag["loo_mse"] = float(best[0])
    diag["in_sample_acc"] = float(best[2])

    # ---------- write back ----------
    new_head_w = (Wc.T + 0.0).astype(np.float32)          # (K,D)
    new_head_b = b.astype(np.float32)                     # (K,)

    # keep magnitudes sane and finite
    new_head_w = np.clip(new_head_w, -50.0, 50.0)
    new_head_b = np.clip(new_head_b, -50.0, 50.0)

    w["head.weight"] = np.ascontiguousarray(new_head_w, dtype=np.float32)
    w["head.bias"] = np.ascontiguousarray(new_head_b, dtype=np.float32)

    # ---------- verify final forward pass with the new tensors ----------
    fnew = forward_feats(X, w)
    lnew = fnew @ w["head.weight"].astype(np.float64).T + w["head.bias"].astype(np.float64)
    diag["final_train_acc"] = float(np.mean(lnew.argmax(1) == y))
    per_class = []
    for c in range(K):
        m = (y == c)
        per_class.append(round(float(np.mean(lnew[m].argmax(1) == c)), 4))
    diag["final_per_class_acc"] = per_class
    diag["final_logit_std"] = float(np.std(lnew))
    diag["head_w_absmax"] = float(np.max(np.abs(w["head.weight"])))
    diag["head_b_absmax"] = float(np.max(np.abs(w["head.bias"])))
    diag["all_finite"] = bool(all(np.all(np.isfinite(v)) for v in w.values()))

    return {"weights": w, "diagnostics": diag}
