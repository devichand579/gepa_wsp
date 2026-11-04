def edit(w, f):
    import numpy as np

    # ------------------------------------------------------------------ setup
    W1 = np.asarray(w["conv1.weight"], dtype=np.float64)
    b1 = np.asarray(w["conv1.bias"], dtype=np.float64)
    W2 = np.asarray(w["conv2.weight"], dtype=np.float64)
    b2 = np.asarray(w["conv2.bias"], dtype=np.float64)
    Hw = np.asarray(w["head.weight"], dtype=np.float64)
    Hb = np.asarray(w["head.bias"], dtype=np.float64)

    X = np.asarray(f["x"], dtype=np.float64)            # (N,1,28,28)
    Y = np.asarray(f["y"], dtype=np.int64)
    N = X.shape[0]
    C = 10
    F = np.asarray(f["features"], dtype=np.float64)     # (N,1568) cached (parent)
    L = np.asarray(f["logits"], dtype=np.float64)

    def conv2d(x, W, b):
        # x: (N,Cin,H,W), W: (Cout,Cin,kh,kw) cross-correlation, pad=1
        Nc, Cin, H, Wd = x.shape
        Cout, _, kh, kw = W.shape
        xp = np.zeros((Nc, Cin, H + 2, Wd + 2))
        xp[:, :, 1:-1, 1:-1] = x
        out = np.zeros((Nc, Cout, H, Wd))
        for i in range(kh):
            for j in range(kw):
                out += np.tensordot(xp[:, :, i:i + H, j:j + Wd], W[:, :, i, j], axes=([1], [1]))
        out += b[None, :, None, None]
        return out

    def maxpool2(a):
        Nc, Cc, H, Wd = a.shape
        H2, W2 = H // 2, Wd // 2
        a = a[:, :, :H2 * 2, :W2 * 2].reshape(Nc, Cc, H2, 2, W2, 2)
        return a.max(axis=(3, 5))

    # ------------------------------------------------- recompute conv1 pooled map
    z1 = conv2d(X, W1, b1)
    P = maxpool2(np.maximum(z1, 0.0))                    # (N,16,14,14)
    # sanity: cached features must match P -> conv2 -> relu -> maxpool
    z2c = conv2d(P, W2, b2)
    Fc = maxpool2(np.maximum(z2c, 0.0)).reshape(N, -1)
    feat_err = float(np.max(np.abs(Fc - F)))
    logit_err = float(np.max(np.abs(F @ Hw.T + Hb - L)))

    # ------------------------------------------------- local (per-position) LDA
    # local vectors: (N*196, 16) ; labels repeated
    V = P.transpose(0, 2, 3, 1).reshape(N * 196, 16)
    lab = np.repeat(Y, 196)
    onehot = np.zeros((N * 196, C))
    onehot[np.arange(N * 196), lab] = 1.0
    # subsample positions for speed (all positions of a subset of samples)
    idx = np.arange(0, N * 196, 7)                      # every 7th position
    Vs = V[idx]
    Ts = onehot[idx]
    mu = Ts.mean(axis=0)                                # (16,)
    Xc = Vs - mu
    Sig = (Xc * Ts).T @ Xc / max(len(Vs), 1)            # (16,16) between-class scatter
    Sxx = (Xc.T @ Xc) / max(len(Vs), 1)
    Sw = Sxx - Sig
    tr = np.trace(Sw) / 16.0
    lam = 0.25 * tr
    A = Sw + lam * np.eye(16)
    Wlda = np.linalg.solve(A, Sig.T).T                 # (10,16)  class scores
    # normalise each class score direction to unit scale
    nrm = np.linalg.norm(Wlda, axis=1, keepdims=True)
    nrm[nrm < 1e-9] = 1.0
    Wlda = Wlda / nrm

    # score maps of the local LDA over all positions of all samples
    Sc = np.tensordot(P, Wlda.T, axes=([1], [0]))       # (N,10,14,14)

    # ------------------------------------------------- build new conv2 (delta kernels)
    # channel c (c<10) : class-evidence channel with per-class threshold
    # channel c (>=10): random-free "detail" channels: identity copies of P channels
    #                   (S = one-hot) so they carry complementary local info.
    W2n = np.zeros((32, 16, 3, 3))
    b2n = np.zeros(32)
    for c in range(C):
        W2n[c, :, 1, 1] = Wlda[c]
    # thresholds: keep only the strongest ~25% of local evidence for each class
    for c in range(C):
        v = Sc[:, c].ravel()
        thr = np.quantile(v, 0.75)
        b2n[c] = -float(thr)
    # 22 detail channels: identity copies of conv1 channels (0..15) plus 6 mixes
    det = list(range(16))
    extra = []
    for k in range(6):
        mix = np.zeros(16)
        mix[k] = 1.0
        mix[(k + 5) % 16] = 1.0
        mix[(k + 9) % 16] = 0.5
        extra.append(mix)
    rows = det + extra
    for j, mix in enumerate(rows):
        o = 10 + j
        W2n[o, :, 1, 1] = mix
        b2n[o] = 0.0

    # ------------------------------------------------- new features from new conv2
    z2n = conv2d(P, W2n, b2n)
    Fn = maxpool2(np.maximum(z2n, 0.0)).reshape(N, -1)   # (N,1568)

    # ------------------------------------------------- closed-form head fit
    def ridge_fit(A, T, lam):
        G = A.T @ A
        G[np.diag_indices_from(G)] += lam * np.trace(G) / A.shape[1]
        return np.linalg.solve(G, A.T @ T)

    # design: [features, 1]
    def fit_head(Fm, Yv, lam):
        A = np.concatenate([Fm, np.ones((Fm.shape[0], 1))], axis=1)
        T = np.zeros((Fm.shape[0], C))
        T[np.arange(Fm.shape[0]), Yv] = 1.0
        B = ridge_fit(A, T, lam)
        return B

    def gcv(Fm, Yv, lams):
        A = np.concatenate([Fm, np.ones((Fm.shape[0], 1))], axis=1)
        T = np.zeros((Fm.shape[0], C))
        T[np.arange(Fm.shape[0]), Yv] = 1.0
        out = []
        for lam in lams:
            G = A.T @ A
            G[np.diag_indices_from(G)] += lam * np.trace(G) / A.shape[1]
            B = np.linalg.solve(G, A.T @ T)
            R = A @ B - T
            dof = np.trace(np.linalg.solve(G, A.T @ A))
            out.append((float(np.mean(R * R) / (1.0 - dof / N)), B, dof))
        return out

    lams = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0])
    res_old = gcv(F, Y, lams)
    res_new = gcv(Fn, Y, lams)

    def acc(B, Fm, Yv):
        A = np.concatenate([Fm, np.ones((Fm.shape[0], 1))], axis=1)
        return float((np.argmax(A @ B, axis=1) == Yv).mean())

    # pick the better design by GCV (deterministic, closed-form)
    i_old = int(np.argmin([r[0] for r in res_old]))
    i_new = int(np.argmin([r[0] for r in res_new]))
    a_old = acc(res_old[i_old][1], F, Y)
    a_new = acc(res_new[i_new][1], Fn, Y)

    if a_new >= a_old:
        Fused, B, lam_used, design = Fn, res_new[i_new][1], float(lams[i_new]), "local_lda"
    else:
        Fused, B, lam_used, design = F, res_old[i_old][1], float(lams[i_old]), "global_ridge"

    # ------------------------------------------------- calibration of logits
    A = np.concatenate([Fused, np.ones((Fused.shape[0], 1))], axis=1)
    raw = A @ B
    # centre raw logits (bias absorbs it), then scale so that CE is minimised by a
    # single temperature (closed form: pick scale maximising margin/entropy proxy)
    raw = raw - raw.mean(axis=1, keepdims=True)
    s = raw.std()
    best = None
    for t in np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]):
        z = raw * (t / max(s, 1e-9))
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=1, keepdims=True)
        ce = -np.log(np.maximum(p[np.arange(N), Y], 1e-12)).mean()
        if best is None or ce < best[0]:
            best = (float(ce), float(t))
    scale = best[1] / max(s, 1e-9)

    Wnew = B[:-1, :] * scale
    bnew = B[-1, :] * scale

    # ------------------------------------------------- write back
    if design == "local_lda":
        w["conv2.weight"] = W2n.astype(np.float32)
        w["conv2.bias"] = b2n.astype(np.float32)
    w["head.weight"] = Wnew.astype(np.float32)
    w["head.bias"] = bnew.astype(np.float32)

    # final self-check with the written weights
    z2f = conv2d(P, np.asarray(w["conv2.weight"], np.float64), np.asarray(w["conv2.bias"], np.float64))
    Ff = maxpool2(np.maximum(z2f, 0.0)).reshape(N, -1)
    lg = Ff @ np.asarray(w["head.weight"], np.float64).T + np.asarray(w["head.bias"], np.float64)
    acc_final = float((np.argmax(lg, axis=1) == Y).mean())
    zz = lg - lg.max(axis=1, keepdims=True)
    pp = np.exp(zz)
    pp /= pp.sum(axis=1, keepdims=True)
    ce_final = float(-np.log(np.maximum(pp[np.arange(N), Y], 1e-12)).mean())

    per_class = []
    for c in range(C):
        m = Y == c
        per_class.append(float((np.argmax(lg[m], axis=1) == c).mean()))

    diag = {
        "design": design,
        "lambda": lam_used,
        "logit_scale": float(scale),
        "train_acc_final": acc_final,
        "train_ce_final": ce_final,
        "per_class_acc_final": per_class,
        "acc_old_design": a_old,
        "acc_new_design": a_new,
        "gcv_old": float(res_old[i_old][0]),
        "gcv_new": float(res_new[i_new][0]),
        "feature_recompute_max_err": feat_err,
        "logit_recompute_max_err": logit_err,
        "conv1_untouched": True,
        "head_w_absmax": float(np.abs(Wnew).max()),
        "head_w_std": float(Wnew.std()),
        "conv2_w_absmax": float(np.abs(W2n).max()) if design == "local_lda" else None,
        "lda_w_absmax": float(np.abs(Wlda).max()),
        "lda_lambda": float(lam),
        "n": int(N),
        "feature_dim": int(F.shape[1]),
    }
    return {"weights": w, "diagnostics": diag}
