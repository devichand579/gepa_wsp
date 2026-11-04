import numpy as np

def edit(w, f):
    # ------------------------------------------------------------------
    # 0) helpers: numpy forward pass (only to verify head refit end-to-end)
    # ------------------------------------------------------------------
    def conv2d(x, W, b):
        # x [N,C,H,W], W [O,C,3,3], b [O]
        N, C, H, Wd = x.shape
        O = W.shape[0]
        xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
        cols = np.empty((N, C * 9, H * Wd), dtype=np.float64)
        for i in range(9):
            r = i // 3
            c = i % 3
            cols[:, :, i::9] = xp[:, :, r:r + H, c:c + Wd].reshape(N, C, H * Wd)
        out = np.einsum('oc,ncp->nop', W.reshape(O, -1), cols) + b[None, :, None]
        return out.reshape(N, O, H, Wd)

    def maxpool2d(x):
        N, C, H, Wd = x.shape
        x = x.reshape(N, C, H // 2, 2, Wd // 2, 2)
        return x.max(axis=(3, 5))

    def forward(x, cw1, cb1, cw2, cb2, hw, hb):
        h = maxpool2d(np.maximum(conv2d(x, cw1, cb1), 0.0))
        h = maxpool2d(np.maximum(conv2d(h, cw2, cb2), 0.0))
        feat = h.reshape(x.shape[0], -1)
        return feat @ hw.T + hb, feat

    diag = {}

    # ------------------------------------------------------------------
    # 1) data
    # ------------------------------------------------------------------
    X = np.asarray(f["features"], dtype=np.float64)          # [N,1568]
    y = np.asarray(f["y"]).astype(np.int64).ravel()          # [N]
    N, D = X.shape
    K = 10
    Y = np.zeros((N, K))
    Y[np.arange(N), y] = 1.0

    Wc = w["conv1.weight"].astype(np.float64)
    bc = w["conv1.bias"].astype(np.float64)
    W2 = w["conv2.weight"].astype(np.float64)
    b2 = w["conv2.bias"].astype(np.float64)
    Hc = w["head.weight"].astype(np.float64)
    Bc = w["head.bias"].astype(np.float64)

    # sanity: cached features must match a numpy forward pass
    try:
        xs = np.asarray(f["x"], dtype=np.float64)
        lg_chk, ft_chk = forward(xs, Wc, bc, W2, b2, Hc, Bc)
        feat_err = float(np.max(np.abs(ft_chk - X)))
        logit_err = float(np.max(np.abs(lg_chk - np.asarray(f["logits"], dtype=np.float64))))
    except Exception:
        feat_err = -1.0
        logit_err = -1.0
    diag["feature_recompute_max_abs_err"] = feat_err
    diag["logit_recompute_max_abs_err"] = logit_err

    # ------------------------------------------------------------------
    # 2) current head quality (pool) + class stats
    # ------------------------------------------------------------------
    def acc_of(Wm, bm):
        return float((np.argmax(X @ Wm.T + bm, axis=1) == y).mean())

    acc_cur = acc_of(Hc, Bc)
    diag["pool_acc_current_head"] = acc_cur
    diag["pool_acc_current_head_per_class"] = [
        float((np.argmax(X @ Hc.T + Bc, axis=1)[y == c] == c).mean()) for c in range(K)
    ]

    # ------------------------------------------------------------------
    # 3) closed-form ridge family with a prior mean = current head
    #    min_W  (1/N)||Xc W - Yc||^2 + a||W - Wp||^2   (+ intercept)
    #    solved from one eigendecomposition of the centered Gram matrix
    # ------------------------------------------------------------------
    xm = X.mean(axis=0)
    Xc = X - xm
    ym = Y.mean(axis=0)
    Yc = Y - ym

    keep = Xc.std(axis=0) > 1e-9
    if keep.sum() < 8:
        keep = np.ones(D, dtype=bool)
    Xk = Xc[:, keep]
    G = (Xk.T @ Xk) / N
    g, V = np.linalg.eigh(G)
    g = np.maximum(g, 0.0)
    C = (Xk.T @ Yc) / N

    # prior (current head) expressed on the kept columns, centered
    Wp_full = Hc.copy()
    Wp_full = Wp_full - Wp_full.mean(axis=1, keepdims=True)   # remove row means
    Wp = Wp_full[:, keep]

    # hat-matrix diagonal pieces:  U = Xk V diag(1/sqrt(g))
    sq = np.sqrt(np.maximum(g, 1e-12))
    U = (Xk @ V) / sq[None, :]
    U2 = U * U

    def solve(a):
        """closed-form ridge (prior mean Wp) for regularization a."""
        d = 1.0 / (g + a)
        Wk_new = V @ (d[:, None] * (V.T @ (C + a * Wp)))
        b_new = ym - xm @ Wk_new.T
        return Wk_new, b_new

    def loo_stats(a):
        """closed-form leave-one-out diagnostics for the ridge solution."""
        d = 1.0 / (g + a)
        Wk_new, b_new = solve(a)
        # full-fit residuals on kept columns
        pred_k = Xk @ Wk_new
        r = Yc - pred_k
        h = (U2 @ (g * d)) / N
        h = np.clip(h, 0.0, 0.999)
        rloo = r / (1.0 - h)[:, None]
        # LOO prediction for sample i (intercept from full fit, centered form)
        pl = ym[None, :] + rloo
        loo_acc = float((np.argmax(pl, axis=1) == y).mean())
        return loo_acc, Wk_new, b_new

    # candidate regularization strengths (relative to mean eigenvalue scale)
    scale = float(g.mean()) if g.mean() > 0 else 1.0
    alphas = [scale * m for m in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)]

    cands = []
    for a in alphas:
        try:
            la, Wk_new, b_new = loo_stats(a)
            Wfull = np.zeros((K, D))
            Wfull[:, keep] = Wk_new
            pa = acc_of(Wfull, b_new)
            cands.append((a, la, pa, Wfull, b_new))
        except Exception:
            pass

    # also a pure OLS-ish solution (no prior) for reference
    best = None
    for (a, la, pa, Wfull, b_new) in cands:
        # selection: best closed-form LOO accuracy; tie-break by pool accuracy
        key = (round(la, 6), pa)
        if best is None or key > best[0]:
            best = (key, a, la, pa, Wfull, b_new)

    if best is None:                      # fallback: keep current head
        return {"weights": w, "diagnostics": diag}

    _, a_best, loo_best, pool_best, W_new, b_new = best

    # ------------------------------------------------------------------
    # 4) safety: only accept the refit if it is not clearly worse on the pool
    # ------------------------------------------------------------------
    if pool_best < acc_cur - 0.02:
        # blend toward the current head (closed-form convex combination)
        lam = 0.5
        W_mix = lam * W_new + (1.0 - lam) * Hc
        b_mix = lam * b_new + (1.0 - lam) * Bc
        if acc_of(W_mix, b_mix) >= acc_cur:
            W_new, b_new = W_mix, b_mix
            pool_best = acc_of(W_mix, b_mix)
        else:
            W_new, b_new = Hc.copy(), Bc.copy()
            pool_best = acc_cur
            diag["edit_applied"] = "kept_current_head"
    else:
        diag["edit_applied"] = "ridge_head_refit"

    # ------------------------------------------------------------------
    # 5) verify end-to-end with the numpy forward pass on all pool images
    # ------------------------------------------------------------------
    try:
        lg_new, ft_new = forward(xs, Wc, bc, W2, b2, W_new, b_new)
        acc_new = float((np.argmax(lg_new, axis=1) == y).mean())
        diag["pool_acc_after_edit_forward_pass"] = acc_new
        diag["pool_acc_after_edit_per_class"] = [
            float((np.argmax(lg_new, axis=1)[y == c] == c).mean()) for c in range(K)
        ]
        lg_cur, _ = forward(xs, Wc, bc, W2, b2, Hc, Bc)
        diag["pool_acc_before_edit_forward_pass"] = float((np.argmax(lg_cur, axis=1) == y).mean())
        # logit margin statistics (confidence) before/after
        def margin(lg):
            s = np.sort(lg, axis=1)
            return float(np.mean(s[:, -1] - s[:, -2]))
        diag["mean_logit_margin_before"] = margin(lg_cur)
        diag["mean_logit_margin_after"] = margin(lg_new)
    except Exception:
        diag["pool_acc_after_edit_forward_pass"] = -1.0

    # ------------------------------------------------------------------
    # 6) write back (preserve names/shapes/dtype, finite, bounded)
    # ------------------------------------------------------------------
    w["head.weight"] = np.clip(W_new, -1000.0, 1000.0).astype(np.float32)
    w["head.bias"] = np.clip(b_new, -1000.0, 1000.0).astype(np.float32)

    diag["alpha_ridge"] = float(a_best)
    diag["alpha_over_mean_eig"] = float(a_best / scale) if scale > 0 else -1.0
    diag["loo_acc_ridge"] = float(loo_best)
    diag["pool_acc_ridge"] = float(pool_best)
    diag["n_features_kept"] = int(keep.sum())
    diag["loo_curve"] = [[float(a / scale), float(l), float(p)] for (a, l, p, _, _) in cands]
    diag["head_weight_abs_max_before"] = float(np.max(np.abs(Hc)))
    diag["head_weight_abs_max_after"] = float(np.max(np.abs(W_new)))
    diag["head_weight_l2_before"] = float(np.linalg.norm(Hc))
    diag["head_weight_l2_after"] = float(np.linalg.norm(W_new))
    diag["conv_unchanged"] = True

    return {"weights": w, "diagnostics": diag}
