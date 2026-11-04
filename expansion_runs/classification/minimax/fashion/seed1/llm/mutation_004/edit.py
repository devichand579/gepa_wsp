import numpy as np

def edit(w, f):
    # ------------------------------------------------------------------ #
    # 0. helpers: forward pass (verified against cached activations)
    # ------------------------------------------------------------------ #
    def conv3(x, wt, b):
        # x: [N,Cin,H,W], wt: [Cout,Cin,3,3], b: [Cout]
        N, Cin, H, Wd = x.shape
        xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
        out = np.zeros((N, wt.shape[0], H, Wd), dtype=np.float32)
        for kr in range(3):
            for kc in range(3):
                sl = xp[:, :, kr:kr + H, kc:kc + Wd]
                out += sl * wt[:, :, kr, kc][None, :, :, None]
        out += b[None, :, None, None]
        return out

    def pool2(a):
        N, C, H, Wd = a.shape
        return a.reshape(N, C, H // 2, 2, Wd // 2, 2).max(axis=(3, 5))

    def forward(x, w1, b1, w2, b2):
        h = pool2(np.maximum(conv3(x, w1, b1), 0.0))
        h = pool2(np.maximum(conv3(h, w2, b2), 0.0))
        return h.reshape(x.shape[0], -1)

    x = np.asarray(f["x"], dtype=np.float32)
    y = np.asarray(f["y"]).astype(np.int64)
    N = x.shape[0]
    Y = np.zeros((N, 10), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    w1 = np.asarray(w["conv1.weight"], dtype=np.float32)
    b1 = np.asarray(w["conv1.bias"], dtype=np.float32)
    w2 = np.asarray(w["conv2.weight"], dtype=np.float32)
    b2 = np.asarray(w["conv2.bias"], dtype=np.float32)
    hw = np.asarray(w["head.weight"], dtype=np.float32)
    hb = np.asarray(w["head.bias"], dtype=np.float32)

    # ------------------------------------------------------------------ #
    # 1. verify forward pass against cached activations
    # ------------------------------------------------------------------ #
    feat_ref = np.asarray(f["features"], dtype=np.float32)
    feat_par = forward(x, w1, b1, w2, b2)
    fwd_err = float(np.max(np.abs(feat_par - feat_ref)))

    # ------------------------------------------------------------------ #
    # 2. design conv2 filters
    #    keep the parent's "energy" channels (box blur of one conv1 map),
    #    and replace the dead channels with band-pass / channel-contrast
    #    filters built from conv1 maps.
    # ------------------------------------------------------------------ #
    C1 = w1.shape[0]          # 16 conv1 channels
    C2 = w2.shape[0]          # 32 conv2 channels
    dead = np.where(np.abs(w2).max(axis=(1, 2, 3)) < 1e-12)[0]
    alive = np.setdiff1d(np.arange(C2), dead)

    # which conv1 channel does each alive conv2 filter read?
    src = np.zeros(C2, dtype=np.int64)
    for o in alive:
        ch = np.abs(w2[o]).max(axis=(1, 2))
        src[o] = int(np.argmax(ch))

    # ---- new conv2 bank ------------------------------------------------ #
    # group conv1 channels into orientation-ish pairs by filter structure:
    # use the sign pattern of the 3x3 kernels to pair "opposite polarity"
    # filters, then build difference filters (band-pass).
    def polarity(v):
        return float(np.sum(v * v))

    # pair conv1 filters by maximal negative correlation of their kernels
    flat1 = w1.reshape(C1, -1).astype(np.float64)
    nrm = np.linalg.norm(flat1, axis=1, keepdims=True)
    flat1n = flat1 / np.maximum(nrm, 1e-9)
    corr = flat1n @ flat1n.T
    pairs = []
    used = set()
    for i in range(C1):
        if i in used:
            continue
        best, bj = -2.0, -1
        for j in range(i + 1, C1):
            if j in used:
                continue
            if corr[i, j] < best:
                best, bj = corr[i, j], j
        if bj >= 0 and best < -0.5:
            pairs.append((i, bj))
            used.add(i); used.add(bj)
        else:
            used.add(i)

    # difference filters: [a,b] -> conv1_a - conv1_b (band-pass on conv1 maps)
    diff_filters = []
    for (a, b) in pairs:
        fk = np.zeros((C1, 3, 3), dtype=np.float32)
        fk[a] = 1.0
        fk[b] = -1.0
        diff_filters.append(fk)

    # oriented "gradient-of-energy" filters: horizontal / vertical difference
    # of a conv1 map (texture contrast)
    dh = np.zeros((C1, 3, 3), dtype=np.float32)
    dv = np.zeros((C1, 3, 3), dtype=np.float32)
    dh[:, 1, 2] = 1.0; dh[:, 1, 0] = -1.0
    dv[:, 2, 1] = 1.0; dv[:, 0, 1] = -1.0

    cand = []  # candidate conv2 filters for the dead slots
    for fk in diff_filters:
        cand.append(fk)
    cand.append(dh)
    cand.append(dv)
    # mixed filters: centre-surround over conv1 maps (band-pass in channel space)
    for (a, b) in pairs[:max(1, len(pairs) // 2)]:
        fk = np.zeros((C1, 3, 3), dtype=np.float32)
        fk[a] = 1.0
        fk[b] = -0.5
        cand.append(fk)

    new_w2 = w2.copy()
    # scale of new filters: match the energy scale of the parent's box filters
    # (parent uses 1/9 per tap -> response scale ~ mean of conv1 map)
    for slot, o in enumerate(dead):
        fk = cand[slot % len(cand)].astype(np.float32)
        fk = fk * 0.11111111
        new_w2[o] = fk
    new_b2 = b2.copy()

    # ------------------------------------------------------------------ #
    # 3. features for the new conv stack
    # ------------------------------------------------------------------ #
    feat_new = forward(x, w1, b1, new_w2, new_b2).astype(np.float64)
    dead_after = int(np.sum(np.abs(new_w2).max(axis=(1, 2, 3)) < 1e-12))

    # ------------------------------------------------------------------ #
    # 4. closed-form ridge head:  W = (F^T F + lam I)^-1 F^T Y
    #    (centered features, eigendecomposition of the Gram matrix)
    # ------------------------------------------------------------------ #
    def fit_ridge(F, Yt, lam):
        mu = F.mean(axis=0)
        Fc = F - mu
        G = (Fc.T @ Fc) / F.shape[0]
        d, V = np.linalg.eigh(G)
        d = np.maximum(d, 0.0)
        # ridge in the whitened space
        A = V * (d / (d + lam))            # [D,D] spectral shrinkage
        # closed-form coefficients: (G+lam I)^-1 F^T Y  via spectral form
        #   F^T Y = Fc^T Y  (intercept handled separately)
        B = (A @ (Fc.T @ Yt)) / F.shape[0]
        Wc = V @ B
        b = Yt.mean(axis=0) - mu @ Wc
        return Wc, b

    def acc_of(F, Yt, Wc, b):
        L = F @ Wc + b
        return float(np.mean(np.argmax(L, axis=1) == np.argmax(Yt, axis=1)))

    # ---- candidate lambda set, all solved from one eigendecomposition ----
    F_new = feat_new
    mu_new = F_new.mean(axis=0)
    Fc_new = F_new - mu_new
    G_new = (Fc_new.T @ Fc_new) / N
    d_new, V_new = np.linalg.eigh(G_new)
    d_new = np.maximum(d_new, 0.0)
    Fty_new = Fc_new.T @ Y

    # closed-form LOOCV (PRESS) for ridge, per lambda, from the hat matrix
    def loocv_mse(F, Yt, lam):
        mu = F.mean(axis=0)
        Fc = F - mu
        G = (Fc.T @ Fc) / F.shape[0]
        d, V = np.linalg.eigh(G)
        d = np.maximum(d, 0.0)
        shrink = d / (d + lam)
        # hat matrix diagonal: h_ii = sum_j shrink_j * (Fc V)_ij^2 / n
        Z = (Fc @ V) / np.sqrt(N)
        h = (Z * Z) @ shrink
        Wc = V @ ((shrink[:, None] * (V.T @ (Fc.T @ Yt))) / N)
        b = Yt.mean(axis=0) - mu @ Wc
        L = F @ Wc + b
        Yhat = np.eye(10)[np.argmax(Yt, axis=1)]
        R = Yhat - L
        h = np.clip(h, 0.0, 0.999)
        press = np.sum((R / (1.0 - h)[:, None]) ** 2) / F.shape[0]
        return press

    lams = np.array([3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0], dtype=np.float64)
    press_new = np.array([loocv_mse(F_new, Y, l) for l in lams])
    lam_new = float(lams[int(np.argmin(press_new))])

    # reference: same estimator on the parent's conv stack (for comparison)
    F_par = feat_par.astype(np.float64)
    press_par = np.array([loocv_mse(F_par, Y, l) for l in lams])
    lam_par = float(lams[int(np.argmin(press_par))])

    # ------------------------------------------------------------------ #
    # 5. 5-fold closed-form CV accuracy (model selection between the two
    #    conv stacks and a small lambda set) -- closed-form solves only
    # ------------------------------------------------------------------ #
    rng = np.random.RandomState(0)
    fold = rng.permutation(N) % 5

    def cv_acc(F, lams_):
        accs = np.zeros(len(lams_))
        for k in range(5):
            tr = fold != k
            te = ~tr
            Ftr, Ytr = F[tr], Y[tr]
            mu = Ftr.mean(axis=0)
            Ftrc = Ftr - mu
            G = (Ftrc.T @ Ftrc) / tr.sum()
            d, V = np.linalg.eigh(G)
            d = np.maximum(d, 0.0)
            Fty = Ftrc.T @ Ytr
            for li, lam in enumerate(lams_):
                shrink = d / (d + lam)
                Wc = V @ ((shrink[:, None] * (V.T @ Fty)) / tr.sum())
                b = Ytr.mean(axis=0) - mu @ Wc
                L = F[te] @ Wc + b
                accs[li] += np.mean(np.argmax(L, axis=1) == np.argmax(Y[te], axis=1))
        return accs / 5.0

    cv_new = cv_acc(F_new, lams)
    cv_par = cv_acc(F_par, lams)
    best_new = float(np.max(cv_new))
    best_par = float(np.max(cv_par))

    # choose the conv stack that generalises better under closed-form CV
    use_new_stack = best_new >= best_par
    if use_new_stack:
        F_fit = F_new
        w2_out, b2_out = new_w2, new_b2
        lam = float(lams[int(np.argmax(cv_new))])
    else:
        F_fit = F_par
        w2_out, b2_out = w2, b2
        lam = float(lams[int(np.argmax(cv_par))])

    # ------------------------------------------------------------------ #
    # 6. final closed-form head fit on all permitted data
    # ------------------------------------------------------------------ #
    Wc, b = fit_ridge(F_fit, Y, lam)
    Wc = np.asarray(Wc, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # 7. diagnostics
    # ------------------------------------------------------------------ #
    L_new = F_fit @ Wc.astype(np.float64) + b.astype(np.float64)
    pred_new = np.argmax(L_new, axis=1)
    acc_new = float(np.mean(pred_new == y))
    L_par = feat_par.astype(np.float64) @ hw.astype(np.float64) + hb.astype(np.float64)
    acc_par = float(np.mean(np.argmax(L_par, axis=1) == y))

    per_class = []
    for c in range(10):
        m = (y == c)
        per_class.append({
            "cls": c,
            "acc_parent": float(np.mean(np.argmax(L_par[m], axis=1) == c)),
            "acc_new": float(np.mean(pred_new[m] == c)),
            "n": int(m.sum()),
        })

    def ce(L, yy):
        z = L - L.max(axis=1, keepdims=True)
        ez = np.exp(z)
        p = ez / ez.sum(axis=1, keepdims=True)
        return float(np.mean(-np.log(np.maximum(p[np.arange(len(yy)), yy], 1e-12))))

    diag = {
        "fwd_max_abs_err_vs_cached_features": fwd_err,
        "dead_conv2_channels_before": int(len(dead)),
        "dead_conv2_channels_after": dead_after,
        "used_new_conv_stack": bool(use_new_stack),
        "lambda": lam,
        "cv_acc_parent_stack": best_par,
        "cv_acc_new_stack": best_new,
        "cv_acc_selected": float(max(best_new, best_par)),
        "inpool_acc_parent_head": acc_par,
        "inpool_acc_new_head": acc_new,
        "inpool_ce_parent": ce(L_par, y),
        "inpool_ce_new": ce(L_new, y),
        "per_class": per_class,
        "head_weight_absmax": float(np.abs(Wc).max()),
        "head_bias_values": [float(v) for v in b],
        "conv2_absmax": float(np.abs(w2_out).max()),
        "press_new": [float(v) for v in press_new],
        "press_parent": [float(v) for v in press_par],
        "lams": [float(v) for v in lams],
    }

    # ------------------------------------------------------------------ #
    # 8. write back (shapes / names preserved, finite, |.| <= 1000)
    # ------------------------------------------------------------------ #
    out = {}
    for k, v in w.items():
        out[k] = np.array(v, dtype=np.float32, copy=True)
    out["conv2.weight"] = np.clip(np.asarray(w2_out, dtype=np.float32), -1000.0, 1000.0)
    out["conv2.bias"] = np.clip(np.asarray(b2_out, dtype=np.float32), -1000.0, 1000.0)
    out["head.weight"] = np.clip(Wc, -1000.0, 1000.0)
    out["head.bias"] = np.clip(b, -1000.0, 1000.0)
    for k, v in out.items():
        v = np.asarray(v, dtype=np.float32)
        if not np.all(np.isfinite(v)):
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        out[k] = v
    return {"weights": out, "diagnostics": diag}
