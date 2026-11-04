import numpy as np

def edit(w, f):
    X = np.asarray(f["x"], dtype=np.int64)      # [N,64] token ids
    y = np.asarray(f["y"], dtype=np.int64)      # [N] labels 0/1
    N, L = X.shape
    V = w["embed.weight"].shape[0]

    # ---------------- bag-of-words features ----------------
    mask = (X != 0)
    nlen = mask.sum(1).astype(np.float64)
    nlen[nlen == 0] = 1.0
    C = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), mask.sum(1))
    cols = X[mask]
    np.add.at(C, (rows, cols), 1.0)
    Xn = C / nlen[:, None]          # length-normalised counts
    Xr = C                          # raw counts
    t = (y == 1).astype(np.float64)
    t = 2.0 * t - 1.0               # +/-1 targets

    # ---------------- closed-form ridge with CV over lambda ----------------
    def fit_ridge(F, lam):
        G = F.T @ F
        b = F.T @ t
        n = F.shape[0]
        A = G + lam * n * np.eye(F.shape[1])
        try:
            s = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            s = np.linalg.lstsq(A, b, rcond=None)[0]
        m = F @ s
        return s, m

    def cv_score(F, lams, folds=5):
        idx = np.arange(N)
        rs = np.random.RandomState(0)
        rs.shuffle(idx)
        parts = np.array_split(idx, folds)
        best = (-1.0, lams[0])
        for lam in lams:
            correct = 0
            for k in range(folds):
                te = parts[k]
                tr = np.concatenate([parts[j] for j in range(folds) if j != k])
                Ftr, Fte = F[tr], F[te]
                G = Ftr.T @ Ftr
                b = Ftr.T @ t[tr]
                A = G + lam * Ftr.shape[0] * np.eye(Ftr.shape[1])
                try:
                    s = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    s = np.linalg.lstsq(A, b, rcond=None)[0]
                sc = Fte @ s
                correct += np.sum((sc > 0).astype(np.int64) == (y[te] == 1).astype(np.int64))
            acc = correct / float(N)
            if acc > best[0]:
                best = (acc, lam)
        return best[1], best[0]

    lams = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
    lam_n, cv_n = cv_score(Xn, lams)
    lam_r, cv_r = cv_score(Xr, lams)

    s_n, m_n = fit_ridge(Xn, lam_n)
    s_r, m_r = fit_ridge(Xr, lam_r)

    # intercepts (unpenalised): mean residual
    b_n = float(np.mean(t - m_n))
    b_r = float(np.mean(t - m_r))

    # ---------------- blend the two models (closed form, 2x2) ----------------
    # score = w1*(Xn@s_n) + w2*(Xr@s_r) + c ; fit w on +/-1 targets
    A1 = Xn @ s_n
    A2 = Xr @ s_r
    Z = np.column_stack([A1, A2, np.ones(N)])
    G = Z.T @ Z + 1e-6 * np.eye(3)
    try:
        coef = np.linalg.solve(G, Z.T @ t)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(G, Z.T @ t, rcond=None)[0]
    w1, w2, c0 = float(coef[0]), float(coef[1]), float(coef[2])

    # ---------------- build the embedding family ----------------
    # dim0 carries the normalised-count model, dim1 the raw-count model,
    # dim2 compensates so that every token vector has identical norm R.
    alpha0, alpha1 = 1.0, 1.0
    beta0, beta1 = 0.0, 0.0
    v0 = alpha0 * s_n + beta0
    v1 = alpha1 * s_r + beta1
    # keep |v| well inside R
    R = 4.0
    m0 = float(np.max(np.abs(v0)))
    m1 = float(np.max(np.abs(v1)))
    if m0 > 0.45 * R:
        alpha0 *= 0.45 * R / m0; v0 = alpha0 * s_n
    if m1 > 0.45 * R:
        alpha1 *= 0.45 * R / m1; v1 = alpha1 * s_r

    emb = np.zeros((V, 32), dtype=np.float64)
    # orthonormal mean-zero basis vectors u (dim0), w (dim1), z (dim2)
    u = np.zeros(32); u[0] = 1.0
    u[1] = -1.0
    u /= np.linalg.norm(u)
    wv = np.zeros(32); wv[2] = 1.0; wv[3] = -1.0
    wv /= np.linalg.norm(wv)
    zv = np.zeros(32); zv[4] = 1.0; zv[5] = -1.0
    zv /= np.linalg.norm(zv)

    a0 = v0
    a1 = v1
    rem = R * R - a0 * a0 - a1 * a1
    rem = np.maximum(rem, 1e-6)
    a2 = np.sqrt(rem)
    emb = (a0[:, None] * u[None, :] + a1[:, None] * wv[None, :] + a2[:, None] * zv[None, :])
    # pad row (id 0) -> zero vector (never pooled, but keep it neutral)
    emb[0, :] = 0.0

    sigma = np.sqrt(R * R / 32.0 + 1e-5)

    # ---------------- write weights ----------------
    w["embed.weight"][...] = emb.astype(np.float32)
    w["pos.weight"][...] = 0.0
    for blk in ("block0", "block1"):
        w[blk + ".proj.weight"][...] = 0.0
        w[blk + ".proj.bias"][...] = 0.0
        w[blk + ".ff2.weight"][...] = 0.0
        w[blk + ".ff2.bias"][...] = 0.0

    # head: logit1 - logit0 = 2*(w1*A1 + w2*A2 + c)
    # pooled[0] = (1/(n*sigma)) * sum_t emb[t,0]  -> (alpha0/sigma)*A1 + (beta0/sigma)
    # pooled[1] = (alpha1/sigma)*A2 + (beta1/sigma)
    # pooled[2] = (1/(n*sigma)) * sum_t a2_t   (unused by head)
    k0 = alpha0 / sigma
    k1 = alpha1 / sigma
    hw = np.zeros((2, 32), dtype=np.float64)
    hb = np.zeros(2, dtype=np.float64)
    hw[1, 0] = k0
    hw[0, 0] = -k0
    hw[1, 1] = k1
    hw[0, 1] = -k1
    # intercept: c0 + (beta0*k0 + beta1*k1) folded in
    inter = c0 + beta0 * k0 + beta1 * k1
    hb[1] = inter
    hb[0] = -inter
    w["head.weight"][...] = hw.astype(np.float32)
    w["head.bias"][...] = hb.astype(np.float32)

    # ---------------- verification with an exact NumPy forward ----------------
    def forward_pooled(ids):
        n = ids.shape[0]
        mm = (ids != 0)
        L = ids.shape[1]
        E = emb[ids]                       # [n,L,32]
        h = E.copy()
        # blocks are identity (proj/ff2 zeroed) -> h unchanged
        mu = h.mean(-1, keepdims=True)
        var = ((h - mu) ** 2).mean(-1, keepdims=True)
        hn = (h - mu) / np.sqrt(var + 1e-5)
        cnt = mm.sum(1).astype(np.float64)
        cnt[cnt == 0] = 1.0
        pooled = (hn * mm[:, :, None]).sum(1) / cnt[:, None]
        return pooled

    pooled = forward_pooled(X)
    logits = pooled @ hw.T + hb
    pred = logits.argmax(1)
    acc = float(np.mean(pred == y))
    p = np.exp(logits - logits.max(1, keepdims=True))
    p = p / p.sum(1, keepdims=True)
    ce = float(-np.mean(np.log(np.clip(p[np.arange(N), y], 1e-9, None))))
    cls = [float(np.mean(pred[y == c] == c)) for c in (0, 1)]

    # in-sample accuracy of the blended linear score (sanity check vs forward)
    sc = w1 * A1 + w2 * A2 + c0
    acc_lin = float(np.mean((sc > 0).astype(np.int64) == (y == 1).astype(np.int64)))

    # ---------------- safety: finite / bounded ----------------
    for k in w:
        arr = np.asarray(w[k], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1000.0, neginf=-1000.0)
        np.clip(arr, -1000.0, 1000.0, out=arr)
        w[k][...] = arr

    diag = {
        "pool_acc_after": round(acc, 4),
        "pool_ce_after": round(ce, 4),
        "class_acc_after": [round(v, 4) for v in cls],
        "blend_linear_acc": round(acc_lin, 4),
        "cv_acc_norm": round(float(cv_n), 4),
        "cv_acc_raw": round(float(cv_r), 4),
        "lam_norm": lam_n,
        "lam_raw": lam_r,
        "blend_w": [round(w1, 4), round(w2, 4), round(c0, 4)],
        "sigma": round(float(sigma), 5),
        "alpha0": round(float(alpha0), 5),
        "alpha1": round(float(alpha1), 5),
        "embed_absmax": round(float(np.max(np.abs(emb))), 4),
        "embed_row0_absmax": round(float(np.max(np.abs(emb[0]))), 6),
        "pooled0_std": round(float(np.std(pooled[:, 0])), 4),
        "pooled1_std": round(float(np.std(pooled[:, 1])), 4),
        "head_w0": [round(float(v), 5) for v in hw[0, :4]],
        "head_w1": [round(float(v), 5) for v in hw[1, :4]],
        "head_b": [round(float(hb[0]), 5), round(float(hb[1]), 5)],
        "note": "blocks zeroed (proj/ff2), pos=0; LN-safe constant-norm embeddings",
    }
    return {"weights": w, "diagnostics": diag}
