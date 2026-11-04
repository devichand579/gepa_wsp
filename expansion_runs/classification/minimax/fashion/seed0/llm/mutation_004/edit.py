def edit(w, f):
    import numpy as np

    X = np.asarray(f["features"], dtype=np.float64)      # [N,1568]
    y = np.asarray(f["y"], dtype=np.int64)                 # [N]
    N, D = X.shape
    C = 10

    # ---------- class moments ----------
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0
    cnt = Y.sum(0)                                        # [C]
    mu = (Y.T @ X) / cnt[:, None]                         # [C,D] class means
    mu_all = X.mean(0)                                   # [D] grand mean

    # ---------- pooled within-class covariance ----------
    Xc = X - mu[y]                                       # centered residuals
    S = (Xc.T @ Xc) / max(N - C, 1)                       # [D,D]
    dS = np.diag(S).copy()

    # ---------- data-driven (Ledoit-Wolf style) shrinkage toward scaled identity ----------
    # target: mu * I  with mu = mean(diag(S))
    mu_d = float(dS.mean())
    # variance of the covariance entries
    Xc2 = Xc ** 2
    var_emp = (Xc2.T @ Xc2) / max(N - C, 1) - dS[:, None] * dS[None, :]
    var_emp = np.maximum(var_emp, 0.0)
    # Frobenius distance to target
    dist2 = float(((dS - mu_d) ** 2).sum() + (S ** 2).sum()
                  - 2.0 * mu_d * float(dS.sum()) + D * mu_d * mu_d)
    b2 = float((var_emp.sum() + (dS ** 2).sum()) / D)     # crude estimate of E||S - target||^2 scale
    # shrinkage coefficient (closed form, Ledoit-Wolf)
    num = 0.0
    for c in range(C):
        Xcc = X - mu[c]
        num += float(((Xcc ** 2).T @ (Xcc ** 2)).sum())
    num /= N
    denom = dist2
    if denom <= 0:
        shrink = 1.0
    else:
        shrink = float(np.clip(num / (N * denom), 0.0, 1.0))
    # guarantee a floor of shrinkage so the inverse is well conditioned
    shrink = float(np.clip(max(shrink, 0.02), 0.0, 1.0))

    Sig = (1.0 - shrink) * S
    Sig[np.diag_indices(D)] += shrink * mu_d + 1e-6

    # ---------- closed-form inverse (symmetric) ----------
    Sig = 0.5 * (Sig + Sig.T)
    try:
        Sig_inv = np.linalg.inv(Sig)
    except np.linalg.LinAlgError:
        Sig_inv = np.linalg.pinv(Sig)
    Sig_inv = 0.5 * (Sig_inv + Sig_inv.T)

    # ---------- LDA discriminant weights ----------
    # w_c = Sig^{-1} (mu_c - mu_all);  bias_c = -0.5 mu_c^T Sig^{-1} mu_c + log prior
    diff = mu - mu_all[None, :]                          # [C,D]
    W = diff @ Sig_inv.T                                 # [C,D]
    b = -0.5 * np.einsum('cd,cd->c', mu, W) + np.log(cnt / N)
    W = W - W.mean(0, keepdims=True)                     # center (bias absorbs it)
    b = b - b.mean()

    # ---------- verify / compare with current head on the pool ----------
    def acc_of(Wm, bm):
        return float((np.argmax(X @ Wm.T + bm, axis=1) == y).mean())

    acc_lda = acc_of(W, b)
    cur = np.asarray(w["head.weight"], dtype=np.float64)
    curb = np.asarray(w["head.bias"], dtype=np.float64)
    acc_cur = acc_of(cur, curb)

    # ---------- choose between LDA and current head by pool accuracy ----------
    if acc_lda < acc_cur:
        W, b = cur, curb
        acc_new = acc_cur
        src = "current"
    else:
        acc_new = acc_lda
        src = "lda"

    # ---------- closed-form per-class affine recalibration of the scores ----------
    Sscores = X @ W.T + b                                # [N,C]
    # target: one-hot (scaled), fit per class a_c * s_c + t_c by least squares
    # solve jointly for [a_c, t_c] with ridge on t
    a = np.ones(C)
    t = np.zeros(C)
    for c in range(C):
        sc = Sscores[:, c]
        yc = Y[:, c]
        # centered least squares: a = cov(s,y)/var(s)
        sm = sc.mean()
        ym = yc.mean()
        vs = float(((sc - sm) ** 2).sum())
        if vs > 1e-12:
            ac = float(((sc - sm) * (yc - ym)).sum() / vs)
        else:
            ac = 1.0
        ac = float(np.clip(ac, 0.05, 20.0))
        tc = ym - ac * sm
        a[c] = ac
        t[c] = tc
    S2 = Sscores * a[None, :] + t[None, :]

    # ---------- global temperature (1-D closed-form scan over a coarse grid) ----------
    def ce_of(M):
        M = M - M.max(1, keepdims=True)
        L = np.log(np.exp(M).sum(1))
        return float((L - M[np.arange(N), y]).mean())

    def acc_of_scores(M):
        return float((np.argmax(M, axis=1) == y).mean())

    best = None
    for k in range(1, 41):
        T = 0.25 * k                     # temperature in [0.25, 10]
        M = S2 / T
        c_ = ce_of(M)
        a_ = acc_of_scores(M)
        key = (-c_, a_)
        if best is None or key > best[0]:
            best = (key, T, c_, a_)
    _, T, ce_best, acc_best = best

    # keep the argmax identical to the uncalibrated version (temperature >0 preserves it)
    final_scores = S2 / T
    acc_final = acc_of_scores(final_scores)

    # ---------- write back ----------
    # fold the affine recalibration into the head:  W' = diag(a)/T * W ,  b' = (diag(a)/T * b + t)/T
    scale = a / T
    Wnew = W * scale[:, None]
    bnew = (b * scale + t) / T

    # safety: keep shapes, finite, bounded
    Wnew = np.clip(Wnew, -1000.0, 1000.0).astype(np.float32)
    bnew = np.clip(bnew, -1000.0, 1000.0).astype(np.float32)
    # ensure finite
    Wnew = np.nan_to_num(Wnew, nan=0.0, posinf=1000.0, neginf=-1000.0)
    bnew = np.nan_to_num(bnew, nan=0.0, posinf=1000.0, neginf=-1000.0)

    w["head.weight"] = Wnew
    w["head.bias"] = bnew

    # ---------- diagnostics ----------
    probs = np.exp(final_scores - final_scores.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    ce = float(-np.log(np.clip(probs[np.arange(N), y], 1e-12, 1.0)).mean())
    pc = np.zeros((C, C))
    for t_ in range(C):
        pc[t_, np.argmax(final_scores[y == t_], axis=1)] += 1
    per_class = [float(pc[i, i] / max(cnt[i], 1)) for i in range(C)]
    diag = {
        "src": src,
        "shrink": round(float(shrink), 6),
        "mu_diag": round(mu_d, 8),
        "acc_current_head": round(acc_cur, 4),
        "acc_lda_raw": round(acc_lda, 4),
        "acc_final": round(acc_final, 4),
        "ce_final_pool": round(ce, 4),
        "ce_before_calib": round(ce_of(S2), 4),
        "temperature": round(float(T), 3),
        "a_scale": [round(float(v), 4) for v in a],
        "t_shift": [round(float(v), 4) for v in t],
        "per_class_acc": [round(v, 4) for v in per_class],
        "head_w_std": round(float(Wnew.std()), 4),
        "head_w_absmax": round(float(np.abs(Wnew).max()), 4),
        "head_b": [round(float(v), 4) for v in bnew],
        "n": int(N), "d": int(D),
    }
    return {"weights": w, "diagnostics": diag}
