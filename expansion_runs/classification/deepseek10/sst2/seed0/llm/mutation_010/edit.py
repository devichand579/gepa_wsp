import numpy as np

def edit(w, f):
    X = np.asarray(f["x"])          # [N,64] int64
    y = np.asarray(f["y"]).astype(np.int64)  # [N]
    N, T = X.shape
    V = w["embed.weight"].shape[0]  # 2000

    # ---- closed-form ridge on length-normalised token counts (term freq) ----
    # build count matrix over non-pad tokens
    mask = (X != 0)
    L = mask.sum(1).astype(np.float64)
    L[L == 0] = 1.0
    # sparse-ish accumulation: use bincount per row via np.add.at on flat index
    rows = np.repeat(np.arange(N), T)
    cols = X.reshape(-1)
    valid = mask.reshape(-1)
    rows = rows[valid]; cols = cols[valid]
    C = np.zeros((N, V), dtype=np.float64)
    np.add.at(C, (rows, cols), 1.0)
    C /= L[:, None]                       # term frequencies

    # target: +1 for positive (class 1), -1 for negative (class 0)
    t = np.where(y == 1, 1.0, -1.0)

    # ridge: solve (C^T C + lam I) a = C^T t   (fit on all 4000)
    lam = 1.0
    CtC = C.T @ C
    Ctt = C.T @ t
    A = CtC + lam * np.eye(V)
    alpha = np.linalg.solve(A, Ctt)       # per-token weights
    # bias via mean residual
    pred0 = C @ alpha
    b = float(np.mean(t - pred0))

    # ---- scale so |alpha_t| <= 5.4 (LayerNorm range ~ sqrt(31)) ----
    amax = float(np.max(np.abs(alpha)))
    scale = 1.0
    if amax > 5.4:
        scale = 5.4 / amax
    a = alpha * scale                       # |a_t| <= 5.4
    c = 1.0 / scale                         # head weight multiplier

    # ---- construct mean-zero, unit-variance embeddings with u[0] = a_t ----
    D = 32
    U = np.zeros((V, D), dtype=np.float64)
    for tok in range(V):
        val = a[tok]
        u = np.zeros(D)
        u[0] = val
        # remaining 31 coords: need sum = -val, sumsq = D - val^2
        rem_sum = -val
        rem_sq = D - val * val
        # distribute over 31 coords: one coord p, 30 coords q
        # p + 30 q = rem_sum ; p^2 + 30 q^2 = rem_sq
        # solve: from first, p = rem_sum - 30 q
        # (rem_sum-30q)^2 + 30 q^2 = rem_sq
        # 900 q^2 - 60 rem_sum q + rem_sum^2 + 30 q^2 = rem_sq
        # 930 q^2 - 60 rem_sum q + (rem_sum^2 - rem_sq) = 0
        Aq = 930.0
        Bq = -60.0 * rem_sum
        Cq = rem_sum * rem_sum - rem_sq
        disc = Bq * Bq - 4 * Aq * Cq
        if disc < 0:
            disc = 0.0
        q = (-Bq - np.sqrt(disc)) / (2 * Aq)
        p = rem_sum - 30.0 * q
        u[1] = p
        u[2:] = q
        # renormalise to exactly unit variance & zero mean (safety)
        u = u - u.mean()
        nrm = np.sqrt((u * u).sum())
        if nrm > 1e-9:
            u = u * (np.sqrt(D) / nrm)
        # re-impose first coordinate exactly (renormalisation shifts it slightly)
        # do a final exact construction: keep u[0] fixed, rescale others
        u0 = val
        rest = u[1:].copy()
        # enforce mean zero: sum(rest) = -u0
        rs = rest.sum()
        if abs(rs) > 1e-12:
            rest = rest - (rs + u0) / rest.size
        # enforce norm: sum(rest^2) = D - u0^2
        target_sq = D - u0 * u0
        cur_sq = (rest * rest).sum()
        if cur_sq > 1e-12:
            rest = rest * np.sqrt(target_sq / cur_sq)
        u = np.concatenate([[u0], rest])
        U[tok] = u

    # ---- write weights ----
    w["embed.weight"] = U.astype(np.float32)
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)

    # keep blocks identity (already zero, but enforce)
    for blk in ["block0", "block1"]:
        w[blk + ".proj.weight"] = np.zeros_like(w[blk + ".proj.weight"], dtype=np.float32)
        w[blk + ".proj.bias"] = np.zeros_like(w[blk + ".proj.bias"], dtype=np.float32)
        w[blk + ".ff2.weight"] = np.zeros_like(w[blk + ".ff2.weight"], dtype=np.float32)
        w[blk + ".ff2.bias"] = np.zeros_like(w[blk + ".ff2.bias"], dtype=np.float32)

    # final norm identity
    w["norm.weight"] = np.ones_like(w["norm.weight"], dtype=np.float32)
    w["norm.bias"] = np.zeros_like(w["norm.bias"], dtype=np.float32)

    # head reads coordinate 0: logit_diff = 2*c*pooled[0] + (b1-b0)
    hw = np.zeros((2, D), dtype=np.float32)
    hw[0, 0] = -c
    hw[1, 0] = c
    w["head.weight"] = hw
    hb = np.zeros(2, dtype=np.float32)
    hb[0] = -0.5 * b
    hb[1] = 0.5 * b
    w["head.bias"] = hb

    # ---- diagnostics ----
    pooled = C @ (a)                       # = mean_t a_t  (predicted pooled[0])
    logit_diff = c * pooled + b            # = mean_t alpha_t + b
    pred = (logit_diff > 0).astype(np.int64)
    acc = float((pred == y).mean())
    diag = {
        "ridge_lambda": lam,
        "amax": amax,
        "scale": float(scale),
        "c": float(c),
        "bias": float(b),
        "train_accuracy": acc,
        "embed_norm_mean": float(np.mean(np.linalg.norm(U, axis=1))),
        "embed_mean": float(U.mean()),
        "embed_std": float(U.std()),
        "head_weight": hw.tolist(),
        "head_bias": hb.tolist(),
        "max_abs_alpha": float(np.max(np.abs(a))),
    }
    return {"weights": w, "diagnostics": diag}
