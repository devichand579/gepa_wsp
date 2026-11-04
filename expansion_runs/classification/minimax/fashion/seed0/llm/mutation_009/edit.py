import numpy as np

def edit(w, f):
    # ---------------- helpers: forward pass (numpy) ----------------
    def conv2d(x, W, b):
        # x [N,C,H,Wi], W [O,C,kh,kw] -> [N,O,H,Wi]
        N, C, H, Wi = x.shape
        O, C2, kh, kw = W.shape
        xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
        cols = np.empty((N, C * kh * kw, H * Wi), dtype=np.float32)
        idx = 0
        for c in range(C):
            for i in range(kh):
                for j in range(kw):
                    cols[:, idx, :] = xp[:, c, i:i + H, j:j + Wi].reshape(N, -1)
                    idx += 1
        out = np.einsum('nkc,ock->no', cols, W.reshape(O, -1), optimize=True)
        out = out.reshape(N, O, H, Wi) + b.reshape(1, O, 1, 1)
        return out.astype(np.float32)

    def pool2(a):
        N, C, H, Wi = a.shape
        H2, W2 = H // 2, Wi // 2
        a = a[:, :, :H2 * 2, :W2 * 2].reshape(N, C, H2, 2, W2, 2)
        return a.max(axis=(3, 5)).astype(np.float32)

    def forward(x, cw1, cb1, cw2, cb2, hw, hb):
        c1 = np.maximum(conv2d(x, cw1, cb1), 0.0)
        p1 = pool2(c1)
        c2 = np.maximum(conv2d(p1, cw2, cb2), 0.0)
        p2 = pool2(c2)
        return c1, p1, c2, p2, p2.reshape(p2.shape[0], -1)

    x = np.asarray(f["x"], dtype=np.float32)
    y = np.asarray(f["y"]).astype(np.int64)
    N = x.shape[0]

    cw1 = np.array(w["conv1.weight"], dtype=np.float32)
    cb1 = np.array(w["conv1.bias"], dtype=np.float32)
    cw2 = np.array(w["conv2.weight"], dtype=np.float32)
    cb2 = np.array(w["conv2.bias"], dtype=np.float32)
    hw = np.array(w["head.weight"], dtype=np.float32)
    hb = np.array(w["head.bias"], dtype=np.float32)

    c1, p1, c2, p2, F = forward(x, cw1, cb1, cw2, cb2, hw, hb)
    Fref = np.asarray(f["features"], dtype=np.float32)
    feat_err = float(np.max(np.abs(F - Fref)))

    # ---------------- class statistics ----------------
    C = 10
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0
    cnt = Y.sum(0) + 1e-12
    prior = cnt / N
    M = (Y.T @ F.astype(np.float64)) / cnt[:, None]          # class means [C,d]
    Xc = F.astype(np.float64) - M[y]                          # centered
    # pooled within-class scatter
    S = (Xc.T @ Xc) / (N - C)
    d = S.shape[0]
    tr = float(np.trace(S)) / d
    # analytic shrinkage (Ledoit-Wolf style)
    S2 = S * S
    num = float(np.sum((S @ S) * S2))          # tr(S^3) style term
    den = float(np.sum(S2))
    delta = 1.0 if den <= 0 else min(1.0, num / (den * den))
    delta = max(delta, 1e-4)
    Sig = (1.0 - delta) * S + delta * tr * np.eye(d)
    Sig_inv_mu = np.linalg.solve(Sig, M.T)                      # [d,C]
    lda_w = Sig_inv_mu.T                                        # [C,d]
    lda_b = -0.5 * np.sum(M * Sig_inv_mu.T, axis=1) + np.log(prior)

    # ---------------- ridge path on features (closed form, SVD) ----------------
    Xf = F.astype(np.float64)
    Xc2 = Xf - Xf.mean(0, keepdims=True)
    U, sv, Vt = np.linalg.svd(Xc2, full_matrices=False)
    UtY = U.T @ Y
    UtYc = UtY - UtY.mean(1, keepdims=True)
    sv2 = sv ** 2

    def ridge_from_sv(lam):
        filt = sv2 / (sv2 + lam)
        Wr = Vt.T @ ((filt * (UtYc / sv2))[:, None, :].squeeze(-1) if False else (filt[:, None] * (UtYc / np.maximum(sv2, 1e-12)[:, None])))
        return Wr

    # closed-form ridge solution: W = V diag(sv/(sv^2+lam)) U^T Yc ; intercept = mean(Y) - mean(X)@W
    def ridge_solve(lam):
        filt = sv / (sv2 + lam)
        Wr = Vt.T @ (filt[:, None] * (UtY - U.mean(0) @ Y)[:, :].T * 0 + (filt[:, None] * ((U.T @ Y) / np.maximum(sv, 1e-12)[:, None])))
        # (U.T@Y) uses full Y (not centered) -> intercept handled separately
        Wr = Vt.T @ (filt[:, None] * ((U.T @ Y) / np.maximum(sv, 1e-12)[:, None]))
        b = Y.mean(0) - Xf.mean(0) @ Wr
        return Wr, b

    # LOO error (closed form) for each lambda, vectorized over lambda grid
    lam_grid = np.array([1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0])
    loo = np.zeros(len(lam_grid))
    for i, lam in enumerate(lam_grid):
        filt = sv / (sv2 + lam)
        Wr = Vt.T @ (filt[:, None] * ((U.T @ Y) / np.maximum(sv, 1e-12)[:, None]))
        b = Y.mean(0) - Xf.mean(0) @ Wr
        pred = Xf @ Wr + b
        resid = Y - pred
        h = (sv2 / (sv2 + lam)) * (U.shape[0] - 1) / max(U.shape[0] - 1, 1)
        # leverage of ridge fit (approx): sum over sv of filt^2
        lev = float(np.sum(filt ** 2))
        denom = np.maximum(1.0 - lev, 1e-6)
        loo_i = (resid ** 2).sum(1) / denom
        loo[i] = float(loo_i.mean())
    best_i = int(np.argmin(loo))
    lam_star = float(lam_grid[best_i])
    Wr, br = ridge_solve(lam_star)

    # ---------------- evaluate candidates (train-pool accuracy) ----------------
    def acc(Wm, bm):
        return float((np.argmax(Xf @ Wm + bm, axis=1) == y).mean())

    acc_cur = acc(hw.astype(np.float64), hb.astype(np.float64))
    acc_lda = acc(lda_w, lda_b)
    acc_ridge = acc(Wr, br)

    # ---------------- logit-space LDA (uses current logits + features) -------
    L = np.asarray(f["logits"], dtype=np.float64)
    Lc = L - L.mean(0, keepdims=True)
    ML = (Y.T @ Lc) / cnt[:, None]
    SL = (Lc - ML[y]).T @ (Lc - ML[y]) / (N - C)
    dL = SL.shape[0]
    trL = float(np.trace(SL)) / dL
    SLs = (1.0 - 0.3) * SL + 0.3 * trL * np.eye(dL)
    SiM = np.linalg.solve(SLs, ML.T)
    lw = SiM.T
    lb = -0.5 * np.sum(ML * SiM.T, axis=1) + np.log(prior)
    acc_ldal = acc(lw, lb)

    # ---------------- closed-form conv2 augmentation ------------------------
    # Build class-discriminative channels by solving a ridge least-squares problem
    # for the conv2 pre-activations of a few extra output channels.
    A = p1.reshape(N, -1).astype(np.float64)      # conv2 input (pooled relu conv1)
    nA = A.shape[1]
    # design: for each new channel, target pre-activation = class-mean contrast
    # (ridge-solved in the conv2 input space, i.e. a 3x3x16 filter set).
    # Use class-mean projections of A (centered by global mean) as targets.
    Ac = A - A.mean(0, keepdims=True)
    G = Ac.T @ Ac
    trG = float(np.trace(G)) / nA
    lam_c = 1e-2 * trG + 1e-9
    Ginv = np.linalg.inv(G + lam_c * np.eye(nA))
    # class-mean contrasts in conv2 input space
    cm = np.zeros((C, nA))
    for c in range(C):
        m = (Y[:, c] @ Ac) / cnt[c]
        cm[c] = m
    # pick the classes that need help most (lowest current accuracy)
    cls_acc = np.array([float((np.argmax(L[y == c], axis=1) == c).mean()) for c in range(C)])
    order = np.argsort(cls_acc)
    # new channels: one per weak class, target = contrast between that class and its
    # most-confusable classes, scaled to a reasonable pre-activation magnitude
    conf = {}
    for c in range(C):
        # confusion: mean logit of other classes on samples of class c
        mu = L[y == c].mean(0)
        mu[c] = -1e9
        conf[c] = int(np.argmax(mu))
    new_ch = []
    new_filters = []
    for c in order[:6]:
        other = conf[c]
        t = (cm[c] - cm[other]) / (np.linalg.norm(cm[c]) + 1e-12)
        # ridge solve: filter f minimizing ||A f - t||^2 + lam||f||^2
        fv = Ginv @ t
        nv = np.linalg.norm(fv)
        if nv < 1e-9:
            continue
        fv = fv / nv * 0.35 * np.sqrt(trG / nA) * 3.0
        new_filters.append(fv.reshape(16, 3, 3))
        new_ch.append(c)
    diag = {"feat_repro_err": feat_err, "n_new_channels": len(new_ch)}

    if new_filters:
        nw = len(new_filters)
        cw2_aug = np.concatenate([cw2, np.stack(new_filters).astype(np.float32)], axis=0)
        cb2_aug = np.concatenate([cb2, np.zeros(nw, dtype=np.float32)])
        c2_aug = np.maximum(conv2d(p1, cw2_aug, cb2_aug), 0.0)
        p2_aug = pool2(c2_aug)
        F_aug = p2_aug.reshape(N, -1)
        # refit head on augmented features with ridge (lam chosen by LOO on the same path)
        Xa = F_aug.astype(np.float64)
        Xam = Xa.mean(0, keepdims=True)
        Xac = Xa - Xam
        Ua, sva, Vta = np.linalg.svd(Xac, full_matrices=False)
        UtYa = Ua.T @ Y
        sva2 = sva ** 2

        def ridge_solve_aug(lam):
            filt = sva / (sva2 + lam)
            Wa = Vta.T @ (filt[:, None] * ((Ua.T @ Y) / np.maximum(sva, 1e-12)[:, None]))
            ba = Y.mean(0) - Xam @ Wa
            return Wa, ba

        loo_a = np.zeros(len(lam_grid))
        for i, lam in enumerate(lam_grid):
            filt = sva / (sva2 + lam)
            Wa = Vta.T @ (filt[:, None] * ((Ua.T @ Y) / np.maximum(sva, 1e-12)[:, None]))
            ba = Y.mean(0) - Xam @ Wa
            pred = Xa @ Wa + ba
            resid = Y - pred
            lev = float(np.sum(filt ** 2))
            loo_a[i] = float(((resid ** 2).sum(1) / max(1.0 - lev, 1e-6)).mean())
        bi = int(np.argmin(loo_a))
        lam_a = float(lam_grid[bi])
        Wa, ba = ridge_solve_aug(lam_a)
        acc_aug = float((np.argmax(Xa @ Wa + ba, axis=1) == y).mean())
        diag.update({"lam_ridge": lam_star, "lam_ridge_aug": lam_a,
                     "acc_ridge_aug": acc_aug, "loo_ridge": float(loo[best_i]),
                     "loo_ridge_aug": float(loo_a[bi]), "delta": float(delta)})

        # choose between plain-feature head and augmented-feature head
        use_aug = acc_aug > acc_ridge
        if use_aug:
            w["conv2.weight"] = np.clip(cw2_aug, -1000, 1000).astype(np.float32)
            w["conv2.bias"] = np.clip(cb2_aug, -1000, 1000).astype(np.float32)
            hw_new = Wa
            hb_new = ba
        else:
            hw_new = Wr
            hb_new = br
    else:
        hw_new = Wr
        hb_new = br

    # ---------------- final head selection ---------------------------------
    # candidates: current, ridge, LDA(feature), LDA(logit), augmented-ridge
    cands = []
    cands.append(("cur", hw.astype(np.float64), hb.astype(np.float64)))
    cands.append(("ridge", Wr, br))
    cands.append(("lda", lda_w, lda_b))
    cands.append(("ldal", lw, lb))
    if new_filters:
        cands.append(("ridge_aug", Wa, ba))

    best = None
    for name, Wm, bm in cands:
        a = acc(Wm, bm)
        if best is None or a > best[0]:
            best = (a, name, Wm, bm)
    acc_best, name_best, Wm_best, bm_best = best

    # blend current head with the best closed-form head (equal-ish weights)
    def blend(W1, b1, W2, b2, t):
        return (1 - t) * W1 + t * W2, (1 - t) * b1 + t * b2

    best_blend = None
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        Wb, bb = blend(hw.astype(np.float64), hb.astype(np.float64), Wm_best, bm_best, t)
        a = acc(Wb, bb)
        if best_blend is None or a > best_blend[0]:
            best_blend = (a, t, Wb, bb)
    acc_blend, t_blend, Wf, bf = best_blend

    # keep the augmented conv2 only if the final head on augmented features is used
    if new_filters and not (name_best == "ridge_aug" or (t_blend > 0 and Wm_best is Wa)):
        w["conv2.weight"] = np.clip(cw2, -1000, 1000).astype(np.float32)
        w["conv2.bias"] = np.clip(cb2, -1000, 1000).astype(np.float32)

    w["head.weight"] = np.clip(Wf, -1000, 1000).astype(np.float32)
    w["head.bias"] = np.clip(bf, -1000, 1000).astype(np.float32)

    # final sanity: recompute logits & accuracy with the edited weights
    _, _, _, _, Ffin = forward(x, w["conv1.weight"], w["conv1.bias"],
                               w["conv2.weight"], w["conv2.bias"],
                               w["head.weight"], w["head.bias"])
    lfin = Ffin.astype(np.float64) @ w["head.weight"].astype(np.float64) + w["head.bias"].astype(np.float64)
    acc_final = float((np.argmax(lfin, axis=1) == y).mean())

    diag.update({
        "acc_current_head": acc_cur,
        "acc_ridge": acc_ridge,
        "acc_lda_feat": acc_lda,
        "acc_lda_logit": acc_ldal,
        "chosen_head": name_best,
        "blend_t": float(t_blend),
        "acc_final": acc_final,
        "head_w_norm": float(np.linalg.norm(w["head.weight"])),
        "conv2_w_norm": float(np.linalg.norm(w["conv2.weight"])),
        "cls_acc_current": [round(float(v), 4) for v in cls_acc],
    })
    return {"weights": w, "diagnostics": diag}
