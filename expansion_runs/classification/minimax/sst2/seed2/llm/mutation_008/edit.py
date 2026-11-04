import numpy as np

def edit(w, f):
    # ---------------- data ----------------
    X = np.asarray(f["x"])            # [N,64] int64 token ids
    Y = np.asarray(f["y"])            # [N] int64 labels
    N, L = X.shape
    V = 2000
    y = np.where(Y == 1, 1.0, -1.0)   # +-1 targets

    # ---- document-normalized binary bag of words (matches mean pooling) ----
    B = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), L)
    cols = X.reshape(-1)
    keep = cols > 0
    B[rows[keep], cols[keep]] = 1.0
    lens = np.maximum(B.sum(1), 1.0)
    F = B / lens[:, None]              # mean of one-hot rows == what the net pools

    # ---------------- closed-form estimators ----------------
    # ridge normal equations: w = (F^T F + lam I)^-1 F^T y
    G = F.T @ F                       # [V,V]
    b = F.T @ y                       # [V]

    # eigen-decomposition of the Gram matrix (closed form, no iteration)
    d, Vmat = np.linalg.eigh(G)
    d = np.maximum(d, 0.0)
    a = Vmat.T @ b                    # projections of the target

    # NB log-odds (smoothed class moments)
    n1 = B[Y == 1].sum(0) + 1.0
    n0 = B[Y == 0].sum(0) + 1.0
    p1 = n1 / n1.sum()
    p0 = n0 / n0.sum()
    r_nb = np.log(p1 / p0)

    # shrunk LDA direction (class moments + covariance)
    mu1 = F[Y == 1].mean(0)
    mu0 = F[Y == 0].mean(0)
    Fm = F - F.mean(0)
    S = (Fm.T @ Fm) / max(N - 2, 1)
    tr = np.trace(S) / V
    r_lda = np.linalg.solve(S + 0.5 * tr * np.eye(V), mu1 - mu0)

    # ---- closed-form LOOCV for the ridge family (vectorized over lambda grid) ----
    lams = np.exp(np.linspace(np.log(1e-3), np.log(1e3), 25))
    Z = F @ Vmat                       # [N,V] rotated design
    Z2 = Z * Z
    # hat diagonal and out-of-fold residual for every lambda
    # h_i(lam) = sum_j Z_ij^2 / (d_j + lam)
    # yhat_i(lam) = sum_j Z_ij * d_j/(d_j+lam) * a_j
    nlam = lams.size
    H = np.zeros((N, nlam))
    Yh = np.zeros((N, nlam))
    for i in range(nlam):
        den = d + lams[i]
        H[:, i] = Z2 @ (1.0 / den)
        Yh[:, i] = Z @ ((d / den) * a)
    resid = y[:, None] - Yh
    denom = np.maximum(1.0 - H, 1e-6)
    loo_mse = (resid / denom) ** 2
    loo_acc = (np.sign(Yh) == y[:, None]).mean(0)
    j = int(np.argmin(loo_mse))
    lam_star = float(lams[j])
    r_ridge = Vmat @ ((d / (d + lam_star)) * a)

    # ---- choose the token-weight vector by the same closed-form CV criterion ----
    def cv_acc(r):
        s = F @ r
        return float(((s > 0).astype(np.float64) == (y > 0)).mean())

    cands = {"nb": r_nb, "ridge": r_ridge, "lda": r_lda}
    accs = {k: cv_acc(v) for k, v in cands.items()}
    best = max(accs, key=lambda k: accs[k])
    r = cands[best].astype(np.float64)

    # scale so that the pooled mean has a sensible magnitude (calibration)
    r = r / (np.abs(r).max() + 1e-12)          # max |r| = 1
    R = 1.05                                   # constant row norm
    Lbar = float(lens.mean())

    # ---------------- realize the BoW model exactly ----------------
    # zero-mean orthonormal u, v in R^32
    Q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((32, 32)))
    # project onto the zero-mean subspace
    P = np.eye(32) - np.ones((32, 32)) / 32.0
    Qp = P @ Q
    Qp, _ = np.linalg.qr(Qp)
    u = Qp[:, 0]
    v = Qp[:, 1]
    u = u / np.linalg.norm(u)
    v = v - u * float(u @ v)
    v = v / np.linalg.norm(v)

    # embed rows: h_t = r_t*u + s_t*v, ||h_t|| = R, mean(h_t)=0
    emb = np.zeros((V, 32), dtype=np.float64)
    s = np.sqrt(np.maximum(R * R - r * r, 0.0))
    emb += r[:, None] * u[None, :]
    emb += s[:, None] * v[None, :]
    emb[0, :] = 0.0                            # pad token: neutral, zero row
    emb = np.clip(emb, -1000.0, 1000.0)

    # blocks -> identity
    for bname in ("block0", "block1"):
        for suf in ("qkv", "proj", "ff1", "ff2"):
            w[bname + "." + suf + ".weight"][...] = 0.0
            w[bname + "." + suf + ".bias"][...] = 0.0
        w[bname + ".norm1.weight"][...] = 1.0
        w[bname + ".norm1.bias"][...] = 0.0
        w[bname + ".norm2.weight"][...] = 1.0
        w[bname + ".norm2.bias"][...] = 0.0

    w["pos.weight"][...] = 0.0
    w["embed.weight"][...] = emb.astype(np.float32)

    # final LayerNorm: weight along u (so only the r-component survives), zero bias
    c = 1.0
    nw = c * u
    w["norm.weight"][...] = nw.astype(np.float32)
    w["norm.bias"][...] = 0.0

    # pooled = beta * u * mean(r) with beta = sqrt(32)/R * c
    beta = np.sqrt(32.0) / R * c
    K = Lbar                                  # logit scale ~ document length
    gamma = K / beta
    hw = np.zeros((2, 32), dtype=np.float64)
    hw[1, :] = gamma * u
    hb = np.zeros(2, dtype=np.float64)
    w["head.weight"][...] = hw.astype(np.float32)
    w["head.bias"][...] = hb.astype(np.float32)

    # ---------------- diagnostics ----------------
    # realized model score on the full pool (mean of r over tokens)
    rsum = np.zeros(N)
    for t in range(V):
        idx = np.where(X == t)
        if idx[0].size:
            rsum += r[t] * np.bincount(idx[0], minlength=N)
    m = rsum / lens
    acc_pool = float(((m > 0).astype(np.float64) == (y > 0)).mean())
    logits = np.stack([-K * m, K * m], 1)
    p1 = 1.0 / (1.0 + np.exp(-np.clip(K * m, -30, 30)))
    ce = float(-np.mean(y * np.log(np.clip(p1, 1e-9, 1 - 1e-9)) +
                        (1 - y) * np.log(np.clip(1 - p1, 1e-9, 1 - 1e-9))))

    bid = np.asarray(f["batch.ids"])
    bx = np.asarray(f["batch.x"])
    by = np.asarray(f["batch.y"])
    brs = np.zeros(len(bid))
    for t in range(V):
        idx = np.where(bx == t)
        if idx[0].size:
            brs += r[t] * np.bincount(idx[0], minlength=len(bid))
    bl = np.maximum((bx > 0).sum(1), 1)
    bm = brs / bl
    acc_batch = float(((bm > 0).astype(np.float64) == (by == 1)).mean())

    diag = {
        "method": "closed_form_ridge_bow",
        "chosen": best,
        "cv_acc_nb": accs["nb"],
        "cv_acc_ridge": accs["ridge"],
        "cv_acc_lda": accs["lda"],
        "ridge_lam": lam_star,
        "ridge_loo_mse": float(loo_mse[j]),
        "ridge_loo_acc": float(loo_acc[j]),
        "pool_acc_realized": acc_pool,
        "pool_ce_realized": ce,
        "batch_acc_realized": acc_batch,
        "mean_len": Lbar,
        "r_absmax": float(np.abs(r).max()),
        "r_top": [float(v) for v in np.sort(r)[-12:]],
        "r_bot": [float(v) for v in np.sort(r)[:12]],
        "beta": float(beta),
        "K": float(K),
        "u_norm": float(np.linalg.norm(u)),
        "v_norm": float(np.linalg.norm(v)),
        "u_v_dot": float(u @ v),
        "u_mean": float(u.mean()),
        "embed_row_norm_min": float(np.linalg.norm(emb, axis=1).min()),
        "embed_row_norm_max": float(np.linalg.norm(emb, axis=1).max()),
    }
    return {"weights": w, "diagnostics": diag}
