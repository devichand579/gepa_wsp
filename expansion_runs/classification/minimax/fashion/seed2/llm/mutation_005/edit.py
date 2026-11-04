def edit(w, f):
    # ---------------- data ----------------
    X = np.asarray(f["features"], dtype=np.float64)      # (N, 1568) cached final features
    y = np.asarray(f["y"]).astype(np.int64).ravel()       # (N,)
    N, D = X.shape
    C = 10
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    def center(Xs, Ys):
        mu_s = Xs.mean(axis=0)
        ybar_s = Ys.mean(axis=0)
        return Xs - mu_s, Ys - ybar_s, mu_s, ybar_s

    def ridge_family(Xc_, Yc_, lams):
        """closed-form ridge solutions for several lambdas at once.
        W[l] = (Xc^T Xc + lam_l I)^-1 Xc^T Yc   (via one symmetric eigendecomposition)"""
        G = Xc_.T @ Xc_
        B = Xc_.T @ Yc_
        ev, V = np.linalg.eigh(G)
        ev = np.maximum(ev, 0.0)
        VTB = V.T @ B
        inv = 1.0 / (ev[:, None] + lams[None, :])          # (D, L)
        Ws = np.einsum('ij,jk,lk->lik', V, VTB, inv)        # (L, C, D)
        return Ws

    # ---------------- one deterministic split for model selection ----------------
    rng = np.random.default_rng(12345)
    perm = rng.permutation(N)
    ntr = int(0.8 * N)
    tr, va = perm[:ntr], perm[ntr:]

    Xtr_c, Ytr_c, mu_tr, ybar_tr = center(X[tr], Y[tr])
    Xva_c = X[va] - mu_tr
    y_va = y[va]

    var_scale = float(np.mean(np.diag(Xtr_c.T @ Xtr_c)))   # mean feature variance
    alphas = np.array([0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    lams = alphas * max(var_scale, 1e-12)

    Wtr = ridge_family(Xtr_c, Ytr_c, lams)                  # (L, C, D)
    btr = ybar_tr[None, :] + Wtr @ mu_tr                    # (L, C)

    logits_va = np.einsum('nd,lcd->lnc', Xva_c, Wtr) + btr[:, None, :]   # (L, Nv, C)
    pred_va = logits_va.argmax(axis=2)
    acc_va = (pred_va == y_va[None, :]).mean(axis=1)        # (L,)

    # log-softmax CE for tie-breaking / scale choice
    def logsoftmax(z):
        m = z.max(axis=-1, keepdims=True)
        e = np.exp(z - m)
        return z - m - np.log(e.sum(axis=-1, keepdims=True))

    ls_va = logsoftmax(logits_va)
    ce_va = -ls_va[:, np.arange(len(y_va)), y_va].mean(axis=1)

    best_acc = acc_va.max()
    tol = acc_va >= (best_acc - 0.002)
    ce_masked = np.where(tol, ce_va, np.inf)
    li = int(np.argmin(ce_masked))
    lam_star = float(lams[li])

    # ---------------- single positive logit scale (argmax unchanged) ----------------
    scales = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0])
    z = logits_va[li]                                       # (Nv, C)
    ls = logsoftmax(scales[:, None, None] * z[None, :, :])
    ce_s = -ls[:, np.arange(len(y_va)), y_va].mean(axis=1)
    s_star = float(scales[int(np.argmin(ce_s))])

    # ---------------- final closed-form refit on the ENTIRE pool ----------------
    Xc, Yc, mu, ybar = center(X, Y)
    W_full = ridge_family(Xc, Yc, np.array([lam_star]))[0]   # (C, D)
    b_full = ybar + W_full @ mu                             # (C,)

    W_final = s_star * W_full
    b_final = s_star * b_full

    # ---------------- write back (head only; conv stack untouched) ----------------
    w["head.weight"] = np.ascontiguousarray(W_final.T.astype(np.float32))  # (10,1568)
    w["head.bias"] = np.ascontiguousarray(b_final.astype(np.float32))      # (10,)

    # ---------------- diagnostics ----------------
    logits_tr = X @ W_final.T + b_final
    pred_tr = logits_tr.argmax(axis=1)
    acc_tr = float((pred_tr == y).mean())
    per_class = []
    for c in range(C):
        m = (y == c)
        per_class.append(float((pred_tr[m] == c).mean()) if m.sum() else 0.0)
    ce_tr = float(-logsoftmax(logits_tr)[np.arange(N), y].mean())
    pred_hist = np.bincount(pred_tr, minlength=C).tolist()

    diag = {
        "note": "head rebuilt by closed-form ridge LS on cached features; conv1/conv2 unchanged",
        "lam_star": lam_star,
        "alpha_star": float(lam_star / max(var_scale, 1e-12)),
        "logit_scale": s_star,
        "holdout_acc": float(acc_va[li]),
        "holdout_acc_all_lams": [float(a) for a in acc_va],
        "holdout_ce_all_lams": [float(c) for c in ce_va],
        "alphas": [float(a) for a in alphas],
        "pool_acc_insample": acc_tr,
        "pool_ce_insample": ce_tr,
        "per_class_acc_insample": per_class,
        "pred_hist_insample": pred_hist,
        "logit_mean": float(logits_tr.mean()),
        "logit_std": float(logits_tr.std()),
        "head_w_abs_max": float(np.abs(W_final).max()),
        "head_w_std": float(W_final.std()),
        "feat_mean": float(X.mean()),
        "feat_std": float(X.std()),
        "feat_zero_frac": float((X == 0).mean()),
        "top_feat_per_class": [int(np.argmax(np.abs(W_final[c]))) for c in range(C)],
    }
    return {"weights": w, "diagnostics": diag}
