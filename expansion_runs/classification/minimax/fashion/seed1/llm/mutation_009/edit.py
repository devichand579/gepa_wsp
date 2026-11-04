def edit(w, f):
    import numpy as np

    X = np.asarray(f["features"], dtype=np.float64)
    y = np.asarray(f["y"]).astype(np.int64)
    N, D = X.shape
    C = 10
    Y1 = np.zeros((N, C), dtype=np.float64)
    Y1[np.arange(N), y] = 1.0

    # ---------- feature conditioning (pool statistics) ----------
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    dead = sd < 1e-9
    live = ~dead
    sd_safe = np.where(dead, 1.0, sd)
    Z = (X - mu) / sd_safe                      # standardized features
    Z[:, dead] = 0.0

    # detect exactly duplicated columns (conv2 duplicates conv1 channels)
    dup = np.zeros(D, dtype=bool)
    for j in range(D):
        if dead[j]:
            continue
        for k in range(j):
            if dead[k]:
                continue
            if np.max(np.abs(Z[:, j] - Z[:, k])) < 1e-6:
                dup[j] = True
                break
    n_live = int(live.sum())
    n_dup = int(dup.sum())

    # ---------- closed-form ridge on standardized features ----------
    def ridge_fit(Xt, Yt, lam):
        p = Xt.shape[1]
        A = Xt.T @ Xt
        A[np.diag_indices(p)] += lam
        B = Xt.T @ Yt
        W = np.linalg.solve(A, B)                # (p, C)
        b = Yt.mean(axis=0) - Xt.mean(axis=0) @ W
        return W, b

    def logits_of(W, b, Xt):
        return Xt @ W + b

    def metrics(L, Yt):
        pred = L.argmax(axis=1)
        acc = float((pred == Yt.argmax(axis=1)).mean())
        P = L - L.max(axis=1, keepdims=True)
        E = np.exp(P)
        P = E / E.sum(axis=1, keepdims=True)
        ce = float(-np.log(np.maximum(P[np.arange(len(Yt)), Yt.argmax(axis=1)], 1e-12)).mean())
        return acc, ce

    # ---------- single train/val split for candidate selection ----------
    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    n_tr = int(0.7 * N)
    tr, va = perm[:n_tr], perm[n_tr:]
    Ztr, Zva = Z[tr], Z[va]
    Ytr, Yva = Y1[tr], Y1[va]

    # candidates: ridge with a few fixed shrinkage levels (closed-form solves only)
    tr_c = np.trace(Ztr.T @ Ztr) / max(1, Ztr.shape[1])
    cands = []
    for mult in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        lam = mult * tr_c
        Wc, bc = ridge_fit(Ztr, Ytr, lam)
        Lva = logits_of(Wc, bc, Zva)
        acc, ce = metrics(Lva, Yva)
        cands.append((acc, -ce, mult, Wc, bc))

    # baseline: current head (scaled for calibration only -> argmax unchanged)
    W0 = np.asarray(w["head.weight"], dtype=np.float64)
    b0 = np.asarray(w["head.bias"], dtype=np.float64)
    L0va = X[va] @ W0.T + b0
    acc0, ce0 = metrics(L0va, Yva)

    cands.sort(key=lambda t: (t[0], t[1]), reverse=True)
    best_acc, best_negce, best_mult, Wc, bc = cands[0]

    # ---------- refit chosen construction on the whole pool ----------
    tr_c_full = np.trace(Z.T @ Z) / max(1, Z.shape[1])
    lam_full = best_mult * tr_c_full
    Wf, bf = ridge_fit(Z, Y1, lam_full)

    # convert back to raw-feature space:  logits = ((f - mu)/sd) @ Wf + bf
    Wraw = Wf / sd_safe[:, None]
    braw = bf - (mu / sd_safe) @ Wf
    Wraw[dead, :] = 0.0

    # ---------- compare on the full pool against the current head ----------
    Lnew = X @ Wraw.T + braw
    acc_new, ce_new = metrics(Lnew, Y1)
    Lold = X @ W0.T + b0
    acc_old, ce_old = metrics(Lold, Y1)

    used_new = (best_acc >= acc0) or (acc_new >= acc_old)
    if used_new:
        W_final, b_final = Wraw, braw
        src = "ridge_refit"
    else:
        # fall back: keep current decision rule, only fix calibration (argmax preserved)
        s = 3.0
        W_final, b_final = s * W0, s * b0
        src = "scaled_current"

    # ---------- final safety: never worse than current on the pool ----------
    Lf = X @ W_final.T + b_final
    acc_f, ce_f = metrics(Lf, Y1)
    if acc_f < acc_old - 1e-12:
        W_final, b_final = W0, b0
        src = "kept_current"
        acc_f, ce_f = acc_old, ce_old

    # ---------- write back (float32, finite, bounded) ----------
    def clean(a, ref):
        a = np.asarray(a, dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        a = np.clip(a, -1000.0, 1000.0)
        return a.reshape(ref.shape).astype(np.float32)

    w["head.weight"] = clean(W_final, w["head.weight"])
    w["head.bias"] = clean(b_final, w["head.bias"])

    diag = {
        "source": src,
        "n_samples": int(N),
        "n_features": int(D),
        "n_dead_features": int(dead.sum()),
        "n_duplicate_features": int(n_dup),
        "n_effective_features": int(n_live - n_dup),
        "chosen_lambda_mult": float(best_mult),
        "val_acc_candidates": [round(float(c[0]), 4) for c in cands],
        "val_acc_current_head": round(float(acc0), 4),
        "val_ce_current_head": round(float(ce0), 4),
        "pool_acc_before": round(float(acc_old), 4),
        "pool_acc_after": round(float(acc_f), 4),
        "pool_ce_before": round(float(ce_old), 4),
        "pool_ce_after": round(float(ce_f), 4),
        "pool_acc_ridge_full": round(float(acc_new), 4),
        "pool_ce_ridge_full": round(float(ce_new), 4),
        "logit_std_before": round(float(Lold.std()), 4),
        "logit_std_after": round(float(Lf.std()), 4),
        "head_w_abs_max": round(float(np.abs(w["head.weight"]).max()), 4),
        "head_b_abs_max": round(float(np.abs(w["head.bias"]).max()), 4),
    }
    return {"weights": w, "diagnostics": diag}
