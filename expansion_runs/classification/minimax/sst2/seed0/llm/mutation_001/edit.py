import numpy as np

def edit(w, f):
    X = np.asarray(f["x"], dtype=np.int64)          # [N,64] token ids
    y = np.asarray(f["y"], dtype=np.int64)          # [N]
    N, L = X.shape
    V = w["embed.weight"].shape[0]                  # 2000
    D = w["embed.weight"].shape[1]                  # 32

    # ---------- 1. bag-of-words design matrix (normalised counts) ----------
    mask = (X != 0)
    nlen = mask.sum(axis=1).astype(np.float64)
    nlen[nlen == 0] = 1.0
    C = np.zeros((N, V), dtype=np.float64)
    for i in range(N):
        row = X[i]
        nz = row[row != 0]
        if nz.size:
            C[i] = np.bincount(nz, minlength=V)
    Z = C / nlen[:, None]                            # row-normalised counts
    t = 2.0 * y.astype(np.float64) - 1.0             # +-1 targets

    # ---------- 2. closed-form ridge with held-out lambda selection ----------
    Zt = Z.T
    G = Zt @ Z                                       # [V,V]
    rhs = Zt @ t

    idx = np.arange(N)
    val = (idx % 5) == 0
    tr = ~val
    Gtr = Z[tr].T @ Z[tr]
    rt = Z[tr].T @ t[tr]
    mtr = Z[tr].mean(axis=0)
    ttr = t[tr].mean()
    # centre for the intercept
    Gtr_c = Gtr - N * np.outer(mtr, mtr)
    rt_c = rt - N * mtr * ttr

    def solve(Gm, rm, lam):
        p = Gm.shape[0]
        A = Gm + lam * np.eye(p)
        try:
            Lc = np.linalg.cholesky(A)
            return np.linalg.solve(Lc.T, np.linalg.solve(Lc, rm))
        except np.linalg.LinAlgError:
            return np.linalg.solve(A + 1e-6 * np.eye(p), rm)

    grid = [3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    best_lam, best_acc = None, -1.0
    for lam in grid:
        wv = solve(Gtr_c, rt_c, lam)
        sc = Z[val] @ wv
        acc = float(np.mean(np.sign(sc) == t[val]))
        if acc > best_acc:
            best_acc, best_lam = acc, lam

    # ---------- 3. final closed-form fit on the whole permitted pool ----------
    m = Z.mean(axis=0)
    tm = t.mean()
    Gc = G - N * np.outer(m, m)
    rc = rhs - N * m * tm
    wv = solve(Gc, rc, best_lam)
    wv = wv / max(1e-9, np.max(np.abs(wv)) + 1e-12)   # normalise scale
    wv = np.clip(wv, -1.0, 1.0)
    b = tm - float(m @ wv)                            # intercept (target +-1 scale)

    # ---------- 4. realise the per-token weights inside the fixed architecture ----------
    # g: mean-zero direction, G = ||g|| large enough that |w_t| <= G*sqrt(D)
    rng = np.random.default_rng(12345)
    g = rng.standard_normal(D)
    g = g - g.mean()
    g = g / np.linalg.norm(g)
    Gnorm = 1.6 * np.sqrt(D)                          # G*sqrt(D) = 1.6*D  >> max|w_t|

    # orthonormal basis of the centred subspace (mean zero), then remove g direction
    B = np.eye(D) - np.ones((D, D)) / D
    Q, _ = np.linalg.qr(B)                            # [D,D], columns span centred space
    Q = Q[:, :D - 1]
    # project out g to get the orthogonal complement inside the centred space
    Qp = Q - np.outer(Q @ g, g)
    Qp, _ = np.linalg.qr(Qp)
    Qp = Qp[:, :D - 2]
    # fixed random unit vector r (mean-zero, orthogonal to g)
    r = Qp @ rng.standard_normal(Qp.shape[1])
    r = r / np.linalg.norm(r)

    wt = np.zeros(V, dtype=np.float64)
    wt[:] = wv
    # feasibility: |w_t| <= Gnorm*sqrt(D)
    cap = 0.9 * Gnorm * np.sqrt(D)
    wt = np.clip(wt, -cap, cap)

    a = wt / (Gnorm ** 2)                             # component along g
    resid = D - a ** 2 * (Gnorm ** 2)                 # = D - w_t^2/G^2
    resid = np.maximum(resid, 0.0)
    z = np.sqrt(resid)[:, None] * r[None, :]          # [V,D]
    U = a[:, None] * g[None, :] + z                   # per-token embedding rows
    # enforce exact mean-zero / unit-variance over the 32 dims
    U = U - U.mean(axis=1, keepdims=True)
    U = U / np.maximum(np.sqrt((U ** 2).mean(axis=1, keepdims=True)), 1e-12)

    # ---------- 5. write the parameters back (shapes/names preserved) ----------
    out = {}
    for k, v in w.items():
        out[k] = np.array(v, dtype=np.float32, copy=True)

    out["embed.weight"][:] = U.astype(np.float32)
    out["pos.weight"][:] = 0.0

    for blk in ("block0", "block1"):
        out[blk + ".proj.weight"][:] = 0.0
        out[blk + ".proj.bias"][:] = 0.0
        out[blk + ".ff1.weight"][:] = 0.0
        out[blk + ".ff1.bias"][:] = 0.0
        out[blk + ".ff2.weight"][:] = 0.0
        out[blk + ".ff2.bias"][:] = 0.0
        out[blk + ".norm1.weight"][:] = 1.0
        out[blk + ".norm1.bias"][:] = 0.0
        out[blk + ".norm2.weight"][:] = 1.0
        out[blk + ".norm2.bias"][:] = 0.0
        out[blk + ".qkv.weight"][:] = 0.0
        out[blk + ".qkv.bias"][:] = 0.0

    out["norm.weight"][:] = 1.0
    out["norm.bias"][:] = 0.0

    # head: logit difference = g . pooled + b  (pooled = mean of per-token U rows)
    out["head.weight"][0] = (-0.5 * Gnorm * g).astype(np.float32)
    out["head.weight"][1] = (0.5 * Gnorm * g).astype(np.float32)
    out["head.bias"][0] = np.float32(-0.5 * b)
    out["head.bias"][1] = np.float32(0.5 * b)

    # ---------- 6. safety: finite, bounded ----------
    for k in out:
        arr = out[k]
        arr[~np.isfinite(arr)] = 0.0
        np.clip(arr, -1000.0, 1000.0, out=arr)

    # ---------- diagnostics ----------
    sc = Z @ wv + b
    pred = (sc > 0).astype(np.int64)
    acc = float(np.mean(pred == y))
    ce = float(np.mean(np.log(np.exp(np.clip(sc, -30, 30)) / (1 + np.exp(np.clip(sc, -30, 30))))))
    ce = float(np.mean(np.maximum(sc, 0) - sc + np.log1p(np.exp(-np.abs(sc)))))
    nz = int(np.sum(np.abs(wv) > 1e-6))
    diag = {
        "lambda_selected": float(best_lam),
        "val_acc_per_lambda": {str(l): float(np.mean(np.sign(Z[val] @ solve(Gtr_c, rt_c, l)) == t[val])) for l in grid},
        "train_fit_acc": acc,
        "train_fit_ce": ce,
        "n_active_tokens": nz,
        "max_abs_token_weight": float(np.max(np.abs(wv))),
        "Gnorm": float(Gnorm),
        "intercept": float(b),
        "mean_len": float(nlen.mean()),
        "embed_row_mean_abs": float(np.abs(U.mean(axis=1)).max()),
        "embed_row_std_minmax": [float(np.sqrt((U ** 2).mean(axis=1)).min()), float(np.sqrt((U ** 2).mean(axis=1)).max())],
        "head_g_norm": float(np.linalg.norm(g)),
        "note": "blocks zeroed -> identity; pos=0; final LN gamma=1 beta=0; embed rows are mean0 unit-var so LN is identity; head diff = g*pooled+b implements ridge BoW logit"
    }
    return {"weights": out, "diagnostics": diag}
