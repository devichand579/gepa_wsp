def edit(w, f):
    import numpy as np

    # ---------------- data ----------------
    X = np.asarray(f["x"], dtype=np.int64)          # [N,64]
    Y = np.asarray(f["y"], dtype=np.int64)          # [N]
    N, L = X.shape
    V = w["embed.weight"].shape[0]
    yv = (2.0 * Y.astype(np.float64) - 1.0)          # -1 / +1

    mask = (X != 0)
    rows = np.repeat(np.arange(N), L)
    cols = X.reshape(-1)
    vals = np.ones(N * L, dtype=np.float64)
    C = np.zeros((N, V), dtype=np.float64)
    np.add.at(C, (rows, cols), vals)                 # token counts
    B = (C > 0).astype(np.float64)                   # binary presence
    Lc = np.log1p(C)                                 # log counts

    # ---------------- closed-form ridge fits ----------------
    def ridge(F, lam):
        A = F.T @ F
        A[np.diag_indices_from(A)] += lam
        return np.linalg.solve(A, F.T @ yv)

    def acc(s):
        sc = C @ s
        return float(np.mean(np.sign(sc) == yv))

    # leave-one-out / GCV-style selection using closed-form ridge identities
    def loo_r2(F, lam):
        # H = F (F^T F + lam I)^-1 F^T  (hat matrix), LOO residual = r/(1-h)
        A = F.T @ F
        A[np.diag_indices_from(A)] += lam
        Ai = np.linalg.inv(A)
        H = F @ Ai @ F.T
        r = yv - F @ (Ai @ (F.T @ yv))
        d = np.clip(1.0 - np.diag(H), 1e-6, None)
        loo = r / d
        return 1.0 - float(np.sum(loo ** 2) / np.sum(yv ** 2))

    cands = []
    for name, F in (("count", C), ("log", Lc), ("binary", B)):
        for lam in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0):
            s = ridge(F, lam)
            cands.append((loo_r2(F, lam), name, lam, s))

    cands.sort(key=lambda t: -t[0])
    best = cands[0]
    s = best[3].copy()

    # ---------------- LDA (class-moment) solution as an alternative ----------------
    def lda_s(F, shrink):
        m0 = F[Y == 0].mean(axis=0)
        m1 = F[Y == 1].mean(axis=0)
        d = m1 - m0
        S = np.cov(F, rowvar=False)
        S = S + shrink * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
        w = np.linalg.solve(S, d)
        return w, float(np.mean(np.sign(F @ w) == yv))

    for shrink in (0.05, 0.2, 0.5):
        wl, al = lda_s(C, shrink)
        if al > acc(s):
            s = wl

    # ---------------- build per-token embedding vectors ----------------
    # s_t must satisfy |s_t| <= |v| * sqrt(32) ; choose v = c*(e0-e1)
    smax = float(np.max(np.abs(s)))
    c = (smax + 1e-6) / 8.0 * 1.5 + 1e-3

    vhat = np.zeros(32); vhat[0] = 1.0 / np.sqrt(2.0); vhat[1] = -1.0 / np.sqrt(2.0)
    phat = np.zeros(32); phat[2] = 1.0 / np.sqrt(2.0); phat[3] = -1.0 / np.sqrt(2.0)

    # a_t = s_t / |v| ; |v| = c*sqrt(2)
    av = s / (c * np.sqrt(2.0))
    av = np.clip(av, -np.sqrt(32.0) * 0.999999, np.sqrt(32.0) * 0.999999)
    rem = np.sqrt(np.maximum(32.0 - av ** 2, 0.0))
    U = av[:, None] * vhat[None, :] + rem[:, None] * phat[None, :]   # [V,32]
    U = U - U.mean(axis=1, keepdims=True)                            # enforce zero mean
    U = U / np.maximum(np.sqrt((U ** 2).sum(axis=1, keepdims=True) / 32.0), 1e-12)
    U = U * np.sqrt(32.0)

    # embed.weight[t] = mu*1 + b*u_t  -> LayerNorm gives u_t exactly (eps negligible)
    b = 50.0
    mu = 0.0
    E = mu + b * U
    E[0] = 0.0        # pad row: harmless (excluded from pooling)

    # ---------------- write weights ----------------
    w["embed.weight"] = E.astype(np.float32)
    w["pos.weight"] = np.zeros_like(w["pos.weight"])

    for blk in ("block0", "block1"):
        w[blk + ".proj.weight"] = np.zeros_like(w[blk + ".proj.weight"])
        w[blk + ".proj.bias"] = np.zeros_like(w[blk + ".proj.bias"])
        w[blk + ".ff2.weight"] = np.zeros_like(w[blk + ".ff2.weight"])
        w[blk + ".ff2.bias"] = np.zeros_like(w[blk + ".ff2.bias"])
        w[blk + ".norm1.weight"] = np.ones_like(w[blk + ".norm1.weight"])
        w[blk + ".norm1.bias"] = np.zeros_like(w[blk + ".norm1.bias"])
        w[blk + ".norm2.weight"] = np.ones_like(w[blk + ".norm2.weight"])
        w[blk + ".norm2.bias"] = np.zeros_like(w[blk + ".norm2.bias"])

    w["norm.weight"] = np.ones_like(w["norm.weight"])
    w["norm.bias"] = np.zeros_like(w["norm.bias"])

    v = c * np.array([1.0, -1.0] + [0.0] * 30, dtype=np.float64)
    hw = np.zeros((2, 32), dtype=np.float64)
    hw[0] = -0.5 * v
    hw[1] = 0.5 * v
    w["head.weight"] = hw.astype(np.float32)
    w["head.bias"] = np.zeros_like(w["head.bias"])

    # ---------------- diagnostics ----------------
    sc = C @ s
    pred = (sc > 0).astype(np.int64)
    tr_acc = float(np.mean(pred == Y))
    lg = np.stack([-sc / 2.0, sc / 2.0], axis=1)
    lg = lg - lg.max(axis=1, keepdims=True)
    ex = np.exp(lg)
    p = ex / ex.sum(axis=1, keepdims=True)
    ce = float(-np.mean(np.log(np.clip(p[np.arange(N), Y], 1e-12, None))))
    # sanity: reconstructed score via designed embedding
    Uc = U
    chk = (Uc[X] * v[None, None, :]).sum(axis=2) * mask
    chk = chk.sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
    corr = float(np.corrcoef(chk, sc)[0, 1])
    diag = {
        "selected_feature": best[1],
        "selected_lambda": float(best[2]),
        "loo_r2": float(best[0]),
        "train_acc_bow": tr_acc,
        "train_ce_bow": ce,
        "mean_abs_score": float(np.mean(np.abs(sc))),
        "max_abs_score": float(np.max(np.abs(sc))),
        "v_scale_c": float(c),
        "embed_row_norm_mean": float(np.mean(np.linalg.norm(E, axis=1))),
        "u_zero_mean_max": float(np.max(np.abs(U.mean(axis=1)))),
        "u_norm2_mean": float(np.mean((U ** 2).sum(axis=1))),
        "recon_score_corr": corr,
        "recon_score_max_abs_err": float(np.max(np.abs(chk - sc))),
        "n_tokens_used": int(np.sum(C.sum(axis=0) > 0)),
        "pos_weight_zero": bool(np.all(w["pos.weight"] == 0)),
    }
    return {"weights": w, "diagnostics": diag}
