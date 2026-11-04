import numpy as np

def edit(w, f):
    X = f["x"].astype(np.int64)          # [N,64]
    y = f["y"].astype(np.int64)          # [N]
    N, L = X.shape
    V = w["embed.weight"].shape[0]       # 2000
    D = w["embed.weight"].shape[1]       # 32

    # ---- normalized bag-of-words design matrix Phi [N, V] ----
    Phi = np.zeros((N, V), dtype=np.float64)
    for i in range(N):
        ids = X[i]
        ids = ids[ids != 0]
        if ids.size == 0:
            continue
        np.add.at(Phi[i], ids, 1.0)
        Phi[i] /= ids.size

    # ---- ridge least-squares classifier on +/-1 targets ----
    tgt = np.where(y == 1, 1.0, -1.0).astype(np.float64)
    # center targets so intercept is handled separately
    Pc = Phi - Phi.mean(axis=0, keepdims=True)
    tc = tgt - tgt.mean()

    # closed-form CV over a small lambda grid (no iterative optimization)
    idx = np.arange(N)
    rng = np.random.RandomState(0)
    perm = rng.permutation(N)
    nfold = 4
    folds = np.array_split(perm, nfold)
    lam_grid = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    best_lam, best_cv = lam_grid[0], -1.0
    for lam in lam_grid:
        accs = []
        for kf in range(nfold):
            te = folds[kf]
            tr = np.concatenate([folds[j] for j in range(nfold) if j != kf])
            A = Pc[tr]
            b = tc[tr]
            G = A.T @ A + lam * np.eye(V)
            try:
                sol = np.linalg.solve(G, A.T @ b)
            except np.linalg.LinAlgError:
                sol = np.linalg.lstsq(G, A.T @ b, rcond=None)[0]
            pred = Pc[te] @ sol
            accs.append(np.mean((pred > 0) == (tgt[te] > 0)))
        cv = float(np.mean(accs))
        if cv > best_cv:
            best_cv, best_lam = cv, lam

    # final fit on all data with chosen lambda
    G = Pc.T @ Pc + best_lam * np.eye(V)
    t = np.linalg.solve(G, Pc.T @ tc)          # [V] per-token weights
    b0 = tgt.mean() - Phi.mean(axis=0) @ t     # intercept

    # ---- scale t so the embedding construction is feasible ----
    # We use g = alpha * e_1, u_v[0] = t_v / alpha, need |t_v/alpha| <= sqrt(31).
    maxabs = np.max(np.abs(t)) + 1e-12
    alpha = maxabs / (np.sqrt(31.0) * 0.95)    # margin so |u_v[0]| < sqrt(31)

    # ---- build u_v (zero-mean, unit-variance) with g . u_v = t_v ----
    # g = alpha * e_1  =>  g . u_v = alpha * u_v[0] = t_v
    # remaining 31 comps: mean m = -u_v[0]/31, plus zero-sum filler of norm R
    z = np.zeros(D - 1, dtype=np.float64)
    z[0] = 1.0 / np.sqrt(2.0)
    z[1] = -1.0 / np.sqrt(2.0)                 # zero-sum, unit-norm in R^31

    U = np.zeros((V, D), dtype=np.float64)
    u0 = t / alpha                              # [V]
    m = -u0 / (D - 1)                           # [V]
    R2 = D - u0 * u0 - (D - 1) * m * m
    R2 = np.clip(R2, 0.0, None)
    R = np.sqrt(R2)                             # [V]
    U[:, 0] = u0
    U[:, 1:] = m[:, None] + R[:, None] * z[None, :]

    # safety: re-standardize each row to exactly zero-mean / unit-variance
    U = U - U.mean(axis=1, keepdims=True)
    s = U.std(axis=1, keepdims=True)
    s[s < 1e-8] = 1.0
    U = U / s
    # re-impose the desired first-component signal (LN is scale/mean invariant,
    # so we can freely set embed = U and LN(embed) = U)
    U[:, 0] = u0
    U = U - U.mean(axis=1, keepdims=True)
    s = U.std(axis=1, keepdims=True)
    s[s < 1e-8] = 1.0
    U = U / s

    # ---- write weights ----
    w["embed.weight"] = U.astype(np.float32)
    w["pos.weight"] = np.zeros_like(w["pos.weight"])

    # keep blocks identity (all zero) and block norms zero
    for name in ["block0.norm1.weight", "block0.norm1.bias",
                 "block0.qkv.weight", "block0.qkv.bias",
                 "block0.proj.weight", "block0.proj.bias",
                 "block0.norm2.weight", "block0.norm2.bias",
                 "block0.ff1.weight", "block0.ff1.bias",
                 "block0.ff2.weight", "block0.ff2.bias",
                 "block1.norm1.weight", "block1.norm1.bias",
                 "block1.qkv.weight", "block1.qkv.bias",
                 "block1.proj.weight", "block1.proj.bias",
                 "block1.norm2.weight", "block1.norm2.bias",
                 "block1.ff1.weight", "block1.ff1.bias",
                 "block1.ff2.weight", "block1.ff2.bias"]:
        w[name] = np.zeros_like(w[name])

    w["norm.weight"] = np.ones_like(w["norm.weight"])
    w["norm.bias"] = np.zeros_like(w["norm.bias"])

    # head: logit_1 - logit_0 = g . pooled + (b1 - b0) = alpha * pooled[0] + ...
    # g = alpha * e_1 ; bias split evenly
    head_w = np.zeros((2, D), dtype=np.float64)
    head_w[1, 0] = alpha
    head_w[0, 0] = -alpha
    head_b = np.zeros(2, dtype=np.float64)
    head_b[1] = b0 / 2.0
    head_b[0] = -b0 / 2.0
    w["head.weight"] = head_w.astype(np.float32)
    w["head.bias"] = head_b.astype(np.float32)

    # ---- diagnostics ----
    # quick sanity: recompute training accuracy with the realized weights
    pooled = Phi @ U.T @ np.eye(D)  # placeholder shape guard
    # actual pooled feature = mean_t LN(embed[token]) = mean_t U[token]
    pooled = np.zeros((N, D), dtype=np.float64)
    for i in range(N):
        ids = X[i]
        ids = ids[ids != 0]
        if ids.size == 0:
            continue
        pooled[i] = U[ids].mean(axis=0)
    logit_diff = pooled @ (head_w[1] - head_w[0]) + (head_b[1] - head_b[0])
    acc = float(np.mean((logit_diff > 0) == (y == 1)))

    diag = {
        "cv_best_lambda": float(best_lam),
        "cv_best_acc": float(best_cv),
        "train_acc_realized": acc,
        "alpha": float(alpha),
        "t_absmax": float(np.max(np.abs(t))),
        "u0_absmax": float(np.max(np.abs(u0))),
        "embed_absmax": float(np.max(np.abs(U))),
        "head_absmax": float(np.max(np.abs(head_w))),
        "b0": float(b0),
    }
    return {"weights": w, "diagnostics": diag}
