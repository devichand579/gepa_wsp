def edit(w, f):
    import numpy as np

    # ---------------- helpers ----------------
    def ln(x, g, b):
        mu = x.mean(-1, keepdims=True)
        xc = x - mu
        var = (xc * xc).mean(-1, keepdims=True)
        sd = np.sqrt(var + 1e-5)
        return xc / sd * g + b

    # ---------------- data ----------------
    X = np.asarray(f["x"], dtype=np.int64)      # [N,64]
    Y = np.asarray(f["y"], dtype=np.int64)      # [N]
    N, L = X.shape
    V = 2000

    mask = (X != 0).astype(np.float64)          # [N,L]
    lens = mask.sum(1)
    lens[lens == 0] = 1.0

    # ---------------- vocabulary stats ----------------
    tok_count = np.zeros(V, dtype=np.float64)
    np.add.at(tok_count, X.ravel(), 1.0)
    tok_count[0] = 0.0

    # ---------------- count features (unigram + bigram) ----------------
    # unigram counts
    C1 = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), L)
    np.add.at(C1, (rows, X.ravel()), 1.0)
    C1[:, 0] = 0.0

    # bigram counts: previous token -> current token
    prev = np.zeros_like(X)
    prev[:, 1:] = X[:, :-1]
    prev[:, 0] = 0
    big_valid = (X != 0) & (prev != 0)
    bidx = prev * V + X
    C2 = np.zeros((N, V * V), dtype=np.float64)
    if big_valid.any():
        np.add.at(C2, (np.repeat(np.arange(N), L)[big_valid], bidx[big_valid]), 1.0)

    # ---------------- targets ----------------
    # smoothed label in [0,1] (shrink toward 0.5 for rare tokens)
    y01 = Y.astype(np.float64)
    p0 = (y01.sum() + 1.0) / (N + 2.0)
    ysm = (y01 * (N - p0) + p0 * (1 - y01)) / (N - 1.0)   # shrunk toward prior
    # weight examples by inverse length so long docs don't dominate
    ex_w = 1.0 / np.sqrt(lens)

    # ---------------- ridge fit on unigrams (closed form) ----------------
    # design: [1/L * counts]  -> per-token coefficient beta
    X1 = C1 / lens[:, None]
    A = X1 * ex_w[:, None]
    b = (ysm - p0) * ex_w
    lam = 3.0
    G = A.T @ A + lam * np.eye(V)
    beta = np.linalg.solve(G, A.T @ b)
    beta[0] = 0.0

    # ---------------- bigram offsets (closed form, residual) ----------------
    resid = b - A @ beta
    X2 = C2 / lens[:, None]
    A2 = X2 * ex_w[:, None]
    lam2 = 30.0
    G2 = A2.T @ A2 + lam2 * np.eye(V * V)
    gamma = np.linalg.solve(G2, A2.T @ resid)
    gamma = gamma.reshape(V, V)
    np.fill_diagonal(gamma, 0.0)
    gamma[0, :] = 0.0
    gamma[:, 0] = 0.0
    # shrink bigram offsets
    gamma *= 0.5

    # ---------------- realize beta as d . E[t] ----------------
    # per-token total weight: beta_t + gamma[prev, t] averaged over occurrences
    # build effective per-token coefficient by averaging bigram contributions
    # eff[t] = beta[t] + mean over contexts of gamma[prev, t]
    cnt_prev = np.zeros((V, V), dtype=np.float64)
    if big_valid.any():
        np.add.at(cnt_prev, (prev[big_valid], X[big_valid]), 1.0)
    ctx_cnt = cnt_prev.sum(1)          # [V] number of times token appears as prev
    ctx_sum = cnt_prev @ gamma.T      # [V] sum over contexts of gamma[prev, ctx]
    with np.errstate(invalid="ignore", divide="ignore"):
        avg_gamma = np.where(ctx_cnt > 0, ctx_sum / np.maximum(ctx_cnt, 1), 0.0)
    eff = beta + avg_gamma             # [V]

    # choose d direction and magnitude
    d = np.zeros(32, dtype=np.float64)
    d[0] = 1.0
    dnorm = 1.0
    # scale so that |eff_t| <= dnorm ; rescale eff if needed
    mx = np.max(np.abs(eff))
    scale = 1.0
    if mx > 0.85 * dnorm:
        scale = (0.85 * dnorm) / mx
        eff = eff * scale
    # perpendicular unit vector (zero-mean over dims)
    u = np.zeros(32, dtype=np.float64)
    u[1] = 1.0
    u[2] = -1.0
    u /= np.linalg.norm(u)

    # embedding rows: E[t] = (eff_t/dnorm) * d_hat + perp component, zero-mean & unit-norm
    # use 30 free dims for perpendicular offsets (mean-zero subspace)
    # deterministic pseudo-random but structured perpendicular directions
    rng = np.random.default_rng(12345)
    P = rng.normal(size=(V, 30))
    P -= P.mean(axis=1, keepdims=True)
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
    # orthonormalize against d_hat in the 30-dim subspace (d_hat has no component there)
    # build embedding
    a = eff / dnorm                        # in [-0.85, 0.85]
    bperp = np.sqrt(np.maximum(1.0 - a * a, 0.0))
    # distribute perpendicular part: mostly along u, plus small random spread
    E = np.zeros((V, 32), dtype=np.float64)
    E[:, 0] = a
    E[:, 1] = bperp * (1.0 / np.sqrt(2.0))
    E[:, 2] = -bperp * (1.0 / np.sqrt(2.0))
    # small extra spread in dims 3..31 (keeps zero-mean & unit norm approximately)
    # to keep exact unit norm, renormalize the whole row after adding spread
    spread = 0.25 * (1.0 - bperp * bperp)
    E[:, 3:33 - 3] = 0.0
    # exact: keep only dims 0,1,2 (zero-mean over 32 dims? no)
    # -> instead use full 32 dims with exact zero-mean & unit-norm construction:
    E = np.zeros((V, 32), dtype=np.float64)
    E[:, 0] = a
    E[:, 1] = bperp / np.sqrt(2.0)
    E[:, 2] = -bperp / np.sqrt(2.0)
    # remaining dims get zero-mean random spread, then renormalize row to unit norm
    # (spread is scaled so that final norm is exactly 1)
    R = rng.normal(size=(V, 29))
    R -= R.mean(axis=1, keepdims=True)
    R /= (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    resid_norm = np.sqrt(np.maximum(1.0 - (a * a + bperp * bperp), 0.0))
    E[:, 3:32] = R * resid_norm[:, None]
    # fix exact zero mean and unit norm per row
    E -= E.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(E, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    E /= nrm
    # re-check: after renormalization d.E[t] may drift slightly; rescale a accordingly
    # (recompute a from projection and rebuild once)
    proj = E @ d
    E2 = E.copy()
    E2[:, 0] = a
    # rebuild perpendicular part to satisfy unit norm & zero mean with fixed dim0
    # solve: keep dims1..31 as orthonormal complement of dim0 within zero-mean subspace
    # simpler: recompose
    E = np.zeros((V, 32), dtype=np.float64)
    E[:, 0] = a
    perp_need = np.sqrt(np.maximum(1.0 - a * a, 0.0))
    # basis of zero-mean subspace excluding dim0: use u and random spread
    Q = np.zeros((V, 32), dtype=np.float64)
    Q[:, 1] = 1.0 / np.sqrt(2.0)
    Q[:, 2] = -1.0 / np.sqrt(2.0)
    R2 = rng.normal(size=(V, 29))
    R2 -= R2.mean(axis=1, keepdims=True)
    R2 /= (np.linalg.norm(R2, axis=1, keepdims=True) + 1e-12)
    # orthogonalize R2 against (e0,u) directions
    for vec in (np.eye(32)[0], np.eye(32)[1] / np.sqrt(2.0) - np.eye(32)[2] / np.sqrt(2.0)):
        R2 -= np.outer(R2 @ vec, vec)
    R2 /= (np.linalg.norm(R2, axis=1, keepdims=True) + 1e-12)
    E = a[:, None] * np.eye(32)[0][None, :] + perp_need[:, None] * (
        0.7 * Q + 0.7 * R2
    )
    # final exact cleanup: zero mean, unit norm, then fix dim0 projection
    E -= E.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(E, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    E /= nrm
    # ensure zero mean exactly after normalization (normalization preserves zero mean)
    E -= E.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(E, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    E /= nrm

    # ---------------- assemble weights ----------------
    W = {k: np.array(v, dtype=np.float32) for k, v in w.items()}

    W["embed.weight"] = E.astype(np.float32)
    W["pos.weight"] = np.zeros((64, 32), dtype=np.float32)

    for blk in ("block0", "block1"):
        W[blk + ".norm1.weight"] = np.ones(32, dtype=np.float32)
        W[blk + ".norm1.bias"] = np.zeros(32, dtype=np.float32)
        W[blk + ".norm2.weight"] = np.ones(32, dtype=np.float32)
        W[blk + ".norm2.bias"] = np.zeros(32, dtype=np.float32)
        W[blk + ".qkv.weight"] = np.zeros((96, 32), dtype=np.float32)
        W[blk + ".qkv.bias"] = np.zeros(96, dtype=np.float32)
        W[blk + ".proj.weight"] = np.zeros((32, 32), dtype=np.float32)
        W[blk + ".proj.bias"] = np.zeros(32, dtype=np.float32)
        W[blk + ".ff1.weight"] = np.zeros((64, 32), dtype=np.float32)
        W[blk + ".ff1.bias"] = np.zeros(64, dtype=np.float32)
        W[blk + ".ff2.weight"] = np.zeros((32, 64), dtype=np.float32)
        W[blk + ".ff2.bias"] = np.zeros(32, dtype=np.float32)

    W["norm.weight"] = np.ones(32, dtype=np.float32)
    W["norm.bias"] = np.zeros(32, dtype=np.float32)

    # head: logit1 - logit0 = d . pooled ; pooled = mean of E[token]
    dvec = np.zeros(32, dtype=np.float64)
    dvec[0] = 1.0
    dvec = dvec / np.linalg.norm(dvec)
    hw = np.zeros((2, 32), dtype=np.float64)
    hw[1] = 0.5 * dvec
    hw[0] = -0.5 * dvec
    hb = np.zeros(2, dtype=np.float64)
    hb[1] = 0.5 * p0
    hb[0] = -0.5 * p0
    W["head.weight"] = hw.astype(np.float32)
    W["head.bias"] = hb.astype(np.float32)

    # ---------------- verify with forward simulation ----------------
    def forward(Wd, Xx):
        ids = Xx
        B, T = ids.shape
        h = Wd["embed.weight"][ids] + Wd["pos.weight"][None, :, :]
        for blk in ("block0", "block1"):
            z = ln(h, Wd[blk + ".norm1.weight"], Wd[blk + ".norm1.bias"])
            qkv = z @ Wd[blk + ".qkv.weight"].T + Wd[blk + ".qkv.bias"]
            q, k, v = np.split(qkv, 3, axis=-1)
            att = np.zeros((B, T, 32), dtype=np.float64)
            m = (ids != 0)
            for hd in range(2):
                sl = slice(hd * 16, (hd + 1) * 16)
                sc = q[:, :, sl] @ k[:, :, sl].transpose(0, 2, 1) / 4.0
                sc = np.where(m[:, None, :], sc, -1e9)
                sc = sc - sc.max(axis=-1, keepdims=True)
                e = np.exp(sc)
                e = e * m[:, None, :]
                e = e / (e.sum(axis=-1, keepdims=True) + 1e-12)
                att[:, :, sl] = e @ v[:, :, sl]
            h = h + att @ Wd[blk + ".proj.weight"].T + Wd[blk + ".proj.bias"]
            z2 = ln(h, Wd[blk + ".norm2.weight"], Wd[blk + ".norm2.bias"])
            f = np.maximum(z2 @ Wd[blk + ".ff1.weight"].T + Wd[blk + ".ff1.bias"], 0.0)
            h = h + f @ Wd[blk + ".ff2.weight"].T + Wd[blk + ".ff2.bias"]
        h = ln(h, Wd["norm.weight"], Wd["norm.bias"])
        mm = (ids != 0).astype(np.float64)
        pooled = (h * mm[:, :, None]).sum(1) / np.maximum(mm.sum(1)[:, None], 1.0)
        logits = pooled @ Wd["head.weight"].T + Wd["head.bias"]
        return logits

    lg = forward(W, X)
    pred = lg.argmax(1)
    acc = float((pred == Y).mean())
    margin = lg[:, 1] - lg[:, 0]
    p1 = 1.0 / (1.0 + np.exp(-np.clip(margin, -50, 50)))
    ce = float(-np.mean(Y * np.log(np.clip(p1, 1e-9, 1)) + (1 - Y) * np.log(np.clip(1 - p1, 1e-9, 1))))

    # also check batch
    lgb = forward(W, np.asarray(f["batch.x"], dtype=np.int64))
    accb = float((lgb.argmax(1) == np.asarray(f["batch.y"])).mean())

    diag = {
        "train_acc": round(acc, 4),
        "train_ce": round(ce, 4),
        "batch_acc": round(accb, 4),
        "prior": round(float(p0), 4),
        "beta_absmax": round(float(np.max(np.abs(beta))), 4),
        "eff_absmax": round(float(np.max(np.abs(eff))), 4),
        "gamma_absmax": round(float(np.max(np.abs(gamma))), 4),
        "embed_row_mean_absmax": round(float(np.max(np.abs(E.mean(1)))), 6),
        "embed_row_norm_err": round(float(np.max(np.abs(np.linalg.norm(E, axis=1) - 1.0))), 6),
        "proj_err": round(float(np.max(np.abs(E @ dvec - eff / dnorm))), 6),
        "logit_std": round(float(np.std(margin)), 4),
    }

    for k in W:
        W[k] = np.clip(np.asarray(W[k], dtype=np.float32), -1000.0, 1000.0)
    return {"weights": W, "diagnostics": diag}
