import numpy as np

def edit(w, f):
    F = np.asarray(f["features"], dtype=np.float64)      # (N,1568)
    y = np.asarray(f["y"]).astype(np.int64).ravel()
    N, D = F.shape
    C = 10

    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    mu_all = F.mean(axis=0)
    Fc = F - mu_all
    Yc = Y - 1.0 / C

    # ---- scatter matrices (closed form) -------------------------------
    G = Fc.T @ Fc                                   # total scatter
    M = Fc.T @ Yc                                   # (D,C) between-class info
    cnt = np.bincount(y, minlength=C).astype(np.float64)
    Mc = np.zeros((C, D), dtype=np.float64)
    for c in range(C):
        Mc[c] = Fc[y == c].mean(axis=0)
    S_B = Mc.T @ (Mc * cnt[:, None])                # between-class scatter
    S_W = G - S_B                                   # within-class scatter

    # ---- eigendecomposition of within-class scatter -------------------
    s, V = np.linalg.eigh(S_W)
    s = np.maximum(s, 0.0)
    Z = V.T @ Fc                                    # (D,N) projections
    ZY = V.T @ M                                   # (D,C)

    trW = float(s.sum())
    trG = float(np.trace(G))
    diagW = trW / D
    diagG = trG / D

    # ---- closed-form PRESS / LOO machinery ---------------------------
    def solve_ridge(lam):
        # ridge onto one-hot using total covariance (centered)
        d = s + diagG + lam
        Wc = V @ (ZY / d[:, None])                  # (D,C)
        return Wc

    def solve_lda(lam):
        # regularized LDA: within-class whitening + Gaussian priors
        d = s + lam
        Wc = V @ (ZY / d[:, None])                  # (D,C)
        # bias from class means and priors
        b = np.zeros(C, dtype=np.float64)
        logpi = np.log(np.maximum(cnt / N, 1e-12))
        for c in range(C):
            mc = Mc[c]
            b[c] = -0.5 * float(mc @ Wc[:, c]) + logpi[c]
        return Wc, b

    def loo_stats(Wc, b, lam, use_total):
        """closed-form leave-one-out via PRESS statistic"""
        if use_total:
            d = s + diagG + lam
        else:
            d = s + lam
        # hat diagonal: h_ii = sum_k Z_ki^2 / d_k
        Hd = (Z * Z / d[:, None]).sum(axis=0)       # (N,)
        Hd = np.clip(Hd, 0.0, 0.999)
        scores = Fc @ Wc + b[None, :]               # (N,C)
        pred = scores.argmax(axis=1)
        resid = scores[np.arange(N), y] - scores.max(axis=1)
        # LOO residual for least-squares style targets (ridge/LDA share
        # the same linear smoother structure)
        e_loo = resid / (1.0 - Hd)
        # LOO decision: class is correct if its score exceeds others by
        # more than the LOO error margin
        srt = np.sort(scores, axis=1)
        margin = srt[:, -1] - srt[:, -2]
        correct_loo = (margin > e_loo)
        acc_loo = float(correct_loo.mean())
        acc_train = float((pred == y).mean())
        return acc_train, acc_loo, float(e_loo.mean()), float(Hd.mean())

    # ---- candidate grid (closed-form solves only) ---------------------
    lams = []
    for a in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        lams.append(a * diagW)
    lams = sorted(set(lams))

    cands = []
    for lam in lams:
        Wc = solve_ridge(lam)
        b = np.zeros(C)
        at, al, ce, hm = loo_stats(Wc, b, lam, True)
        cands.append(("ridge", lam, Wc, b, at, al, ce, hm))
        Wc2, b2 = solve_lda(lam)
        at2, al2, ce2, hm2 = loo_stats(Wc2, b2, lam, False)
        cands.append(("lda", lam, Wc2, b2, at2, al2, ce2, hm2))

    # pick by LOO accuracy, then by LOO residual margin
    best = None
    for c in cands:
        key = (c[5], -c[6])
        if best is None or key > best[0]:
            best = (key, c)
    kind, lam, Wc, b, acc_tr, acc_loo, ce, hm = best[1]

    # ---- assemble final head weights ----------------------------------
    # scores = Fc @ Wc + b  ->  logits = F @ Wc + (b - mu_all @ Wc)
    headW = Wc.T                                     # (10,1568)
    headb = b - mu_all @ Wc

    # scale guard: keep magnitudes sane and finite
    mx = float(np.max(np.abs(headW)))
    if mx > 50.0:
        headW = headW * (50.0 / mx)
    headW = np.clip(headW, -1000.0, 1000.0)
    headb = np.clip(headb, -1000.0, 1000.0)

    w["head.weight"] = headW.astype(np.float32)
    w["head.bias"] = headb.astype(np.float32)

    # ---- diagnostics ----------------------------------------------------
    scores = F @ headW.T + headb[None, :]
    pred = scores.argmax(axis=1)
    acc = float((pred == y).mean())
    p = np.exp(scores - scores.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    ce_full = float(-np.log(np.maximum(p[np.arange(N), y], 1e-12)).mean())
    per_class = []
    for c in range(C):
        m = y == c
        per_class.append(round(float((pred[m] == c).mean()), 4))
    cand_summary = [
        {"kind": c[0], "lam": float(c[1]), "train_acc": round(float(c[4]), 4),
         "loo_acc": round(float(c[5]), 4), "loo_resid": round(float(c[6]), 5)}
        for c in cands
    ]

    diag = {
        "chosen": kind,
        "lambda": float(lam),
        "train_acc": round(acc, 4),
        "train_ce": round(ce_full, 4),
        "loo_acc": round(float(acc_loo), 4),
        "loo_resid": round(float(ce), 5),
        "mean_hat_diag": round(float(hm), 4),
        "per_class_acc": per_class,
        "head_w_std": round(float(headW.std()), 6),
        "head_w_absmax": round(float(np.abs(headW).max()), 4),
        "head_b": [round(float(v), 4) for v in headb],
        "candidates": cand_summary,
        "feature_mean_abs": round(float(np.abs(mu_all).mean()), 5),
        "conv_untouched": True,
    }
    return {"weights": w, "diagnostics": diag}
