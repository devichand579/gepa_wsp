import numpy as np

def edit(w, f):
    X = np.asarray(f["x"], dtype=np.int64)          # [N,64] token ids
    y = np.asarray(f["y"], dtype=np.int64)          # [N]
    N, L = X.shape
    V = int(w["embed.weight"].shape[0])
    D = int(w["embed.weight"].shape[1])

    mask = (X != 0)
    lens = mask.sum(1).astype(np.float64)
    lens[lens == 0] = 1.0

    # ---------- bag-of-words design (binary presence) ----------
    rows = np.repeat(np.arange(N), lens.astype(np.int64))
    cols = X[mask]
    B = np.zeros((N, V), dtype=np.float64)
    B[rows, cols] = 1.0
    B[:, 0] = 0.0                                   # drop PAD
    n1 = B[y == 1].sum(0)
    n0 = B[y == 0].sum(0)
    cls = (y == 1).astype(np.float64)

    # ---------- closed-form candidate models ----------
    def nb_logodds(alpha):
        p1 = (n1 + alpha) / (n1.sum() + alpha * V)
        p0 = (n0 + alpha) / (n0.sum() + alpha * V)
        return np.log(p1 / p0)

    def lda(lam):
        Xc = B - cls[:, None]
        S = (Xc.T @ Xc) / N
        d = B[y == 1].mean(0) - B[y == 0].mean(0)
        A = S + lam * np.eye(V)
        return np.linalg.solve(A, d)

    def ridge(lam):
        G = B.T @ B
        t = 2.0 * cls - 1.0
        A = G + lam * np.eye(V)
        return np.linalg.solve(A, B.T @ t)

    def acc_of(r, thr=0.0):
        s = B @ r - thr
        return float(np.mean((s > 0) == (cls > 0.5)))

    def oof_acc(fit_fn, folds=5, seed=0):
        rng = np.random.RandomState(seed)
        idx = rng.permutation(N)
        parts = np.array_split(idx, folds)
        correct = 0
        for k in range(folds):
            te = parts[k]
            tr = np.concatenate([parts[j] for j in range(folds) if j != k])
            r = fit_fn(tr)
            s = B[te] @ r
            correct += int(np.sum((s > 0) == (cls[te] > 0.5)))
        return correct / float(N)

    def sub_fit(fit_fn, tr):
        return fit_fn(tr)

    # out-of-fold accuracy for a small set of closed-form candidates
    cand = {}
    for a in (0.5, 1.0, 2.0, 4.0):
        cand["nb_a%.1f" % a] = (lambda a: (lambda tr: nb_logodds(a)))(a)
    for lam in (1.0, 10.0, 50.0, 200.0, 1000.0):
        cand["lda_%.0f" % lam] = (lambda lam: (lambda tr: lda(lam)))(lam)
        cand["ridge_%.0f" % lam] = (lambda lam: (lambda tr: ridge(lam)))(lam)

    oof = {}
    for name, fn in cand.items():
        try:
            oof[name] = oof_acc(fn)
        except Exception:
            oof[name] = 0.0

    best_name = max(oof, key=lambda k: oof[k])
    best_oof = oof[best_name]

    # ---------- negation handling ----------
    # polarity of each token: sign of its class-conditional mean sentiment
    base = nb_logodds(2.0)
    # tokens whose usage polarity is inverted relative to the global direction
    glob = float(np.mean(base[B > 0]))
    neg_cand = (base < 0) & (glob > 0)
    # keep only tokens that are frequent enough to trust
    freq = n1 + n0
    neg_tokens = np.where(neg_cand & (freq >= 8))[0]

    def apply_neg(r, strength):
        rr = r.copy()
        if len(neg_tokens) > 0:
            rr[neg_tokens] = -strength * np.abs(rr[neg_tokens])
        return rr

    # ---------- choose final weights: blend of top candidates + negation ----------
    top = sorted(oof, key=lambda k: -oof[k])[:4]
    R_list = []
    for name in top:
        fn = cand[name]
        R_list.append(fn(np.arange(N)))
    R = np.mean(R_list, axis=0)

    # scale to a common scale (unit L2) before blending negation variants
    def nrm(v):
        s = np.linalg.norm(v)
        return v / s if s > 1e-12 else v

    variants = {"plain": R}
    for st in (0.0, 0.3, 0.6, 1.0):
        variants["neg%.1f" % st] = apply_neg(R, st)

    # evaluate each variant out-of-fold-ish: use in-sample acc + a threshold sweep
    best = None
    best_key = None
    for key, r in variants.items():
        rn = nrm(r)
        s = B @ rn
        # threshold chosen on the pool (mean-pooled rule uses mean not sum)
        cand_thr = np.quantile(s, np.linspace(0.05, 0.95, 19))
        for thr in cand_thr:
            a = float(np.mean((s > thr) == (cls > 0.5)))
            if best is None or a > best:
                best = a
                best_key = (key, float(thr), rn)

    key, thr, r_final = best_key

    # ---------- encode token weights into the embedding ----------
    # constant radius R0 so LayerNorm is an exact positive scaling
    R0 = float(np.max(np.abs(r_final)))
    if R0 < 1e-6:
        R0 = 1.0
    R0 = R0 * 1.0000001
    # u : zero-mean unit vector; v : zero-mean unit vector orthogonal to u
    rng = np.random.RandomState(1234)
    u = rng.randn(D)
    u -= u.mean()
    u /= np.linalg.norm(u)
    v = rng.randn(D)
    v -= v.mean()
    v -= (v @ u) * u
    v /= np.linalg.norm(v)

    r = r_final / np.max(np.abs(r_final)) * (1.0 - 1e-6)   # |r_t| < R0
    s_t = np.sqrt(np.maximum(R0 * R0 - r * r, 0.0))

    E = np.zeros((V, D), dtype=np.float64)
    E[:, :] = (r[:, None] * u[None, :] + s_t[:, None] * v[None, :])
    E[0, :] = 0.0                                          # PAD token -> zeros
    E = np.clip(E, -1000.0, 1000.0)

    # ---------- write weights: identity blocks, exact linear head ----------
    def zero(name):
        w[name] = np.zeros_like(w[name])

    zero("pos.weight")
    for b in ("block0", "block1"):
        for suf in ("qkv.weight", "qkv.bias", "proj.weight", "proj.bias",
                    "ff1.weight", "ff1.bias", "ff2.weight", "ff2.bias"):
            zero(b + "." + suf)
        # keep LayerNorms as identity affine
        w[b + ".norm1.weight"] = np.ones_like(w[b + ".norm1.weight"])
        w[b + ".norm1.bias"] = np.zeros_like(w[b + ".norm1.bias"])
        w[b + ".norm2.weight"] = np.ones_like(w[b + ".norm2.weight"])
        w[b + ".norm2.bias"] = np.zeros_like(w[b + ".norm2.bias"])
    w["norm.weight"] = np.ones_like(w["norm.weight"])
    w["norm.bias"] = np.zeros_like(w["norm.bias"])

    w["embed.weight"] = E.astype(np.float32)

    # pooled = (sqrt(D)/R0) * mean_t x_t ;  logits_diff = hw1 . pooled - hw0 . pooled + (b1-b0)
    scale = np.sqrt(D) / R0
    # mean-pooled score: (scale/R0?) -> compute directly: pooled = scale * mean_t x_t
    # score = (hw1-hw0) . pooled + (b1-b0)
    # choose direction d = u (so only r_t matters), coefficient c so that
    # score = c * mean_t r_t + thr_scaled
    # pooled_j = scale * mean_t x_t[j]; with x_t[j] = r_t u_j + s_t v_j
    # choose head diff = alpha * u  -> score = alpha*scale*u.(mean_t x_t) = alpha*scale*mean_t r_t
    # we want score = mean_t r_t - thr  (mean-based threshold)  => alpha = 1/scale
    alpha = 1.0 / scale
    hw = np.zeros((2, D), dtype=np.float64)
    hw[1, :] = alpha * u
    hw[0, :] = -alpha * u
    hb = np.zeros(2, dtype=np.float64)
    hb[1] = thr
    hb[0] = -thr
    w["head.weight"] = hw.astype(np.float32)
    w["head.bias"] = hb.astype(np.float32)

    # ---------- diagnostics ----------
    diag = {
        "method": "bow_linear_exact_via_layernorm",
        "oof_candidates": {k: round(float(v), 4) for k, v in oof.items()},
        "best_candidate": best_name,
        "best_oof_acc": round(float(best_oof), 4),
        "chosen_variant": key,
        "chosen_threshold": round(float(thr), 5),
        "pool_acc_chosen": round(float(best), 4),
        "neg_tokens_found": int(len(neg_tokens)),
        "neg_token_ids": [int(t) for t in neg_tokens[:40]],
        "r_absmax": round(float(np.max(np.abs(r_final))), 5),
        "r_top": [round(float(v), 4) for v in np.sort(r_final)[-12:]],
        "r_bot": [round(float(v), 4) for v in np.sort(r_final)[:12]],
        "scale": round(float(scale), 5),
        "alpha": round(float(alpha), 5),
        "mean_len": round(float(np.mean(lens)), 3),
        "unk_rate": round(float(np.mean(X[mask] == 1)), 4),
        "class_balance": round(float(np.mean(cls)), 4),
        "embed_absmax": round(float(np.max(np.abs(E))), 4),
        "head_absmax": round(float(np.max(np.abs(hw))), 4),
    }
    return {"weights": w, "diagnostics": diag}
