def edit(w, f):
    import numpy as np
    import math

    x = np.asarray(f["x"], dtype=np.int64)      # [N,64]
    y = np.asarray(f["y"], dtype=np.int64)      # [N]
    N, L = x.shape
    V = w["embed.weight"].shape[0]              # 2000
    D = w["embed.weight"].shape[1]              # 32

    # ---------- 1. bag-of-words normalized count features ----------
    mask = (x != 0).astype(np.float64)
    ntok = np.maximum(mask.sum(axis=1), 1.0)    # [N]
    Phi = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), x.shape[1])
    np.add.at(Phi, (rows, x.ravel()), mask.ravel())
    Phi = Phi / ntok[:, None]                   # normalized counts

    t = (2.0 * y.astype(np.float64) - 1.0)      # +1 positive, -1 negative

    # ---------- 2. closed-form ridge on (Phi, 1) ----------
    A = np.concatenate([Phi, np.ones((N, 1))], axis=1)
    G = A.T @ A
    b = A.T @ t
    # generalized cross-validation over lambda grid (closed form, no training loop)
    best = None
    for lam in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]:
        R = G.copy()
        R[np.arange(R.shape[0]), np.arange(R.shape[0])] += lam
        try:
            beta = np.linalg.solve(R, b)
        except np.linalg.LinAlgError:
            continue
        resid = A @ beta - t
        dof = float(np.trace(np.linalg.solve(R, G)))
        denom = max(1e-9, (1.0 - dof / N))
        gcv = float((resid @ resid) / (N * denom * denom))
        if best is None or gcv < best[0]:
            best = (gcv, lam, beta)
    gcv, lam, beta = best
    gamma_raw = beta[:V]        # per-word weights on normalized counts
    c = float(beta[V])          # intercept

    # ---------- 3. build exact per-token design ----------
    m = float(np.max(np.abs(gamma_raw)))
    if not np.isfinite(m) or m < 1e-12:
        m = 1.0
    g = gamma_raw / m          # |g_v| <= 1
    s = m                      # head scale so that s*g == gamma_raw

    # zero-mean unit-variance basis in R^32
    ones = np.ones(D) / math.sqrt(D)
    rng = np.random.default_rng(0)
    v = rng.standard_normal(D)
    v = v - v.mean()
    v = v / np.linalg.norm(v)
    d_hat = v                                   # head direction
    # orthogonal zero-mean unit vector p_hat
    p = rng.standard_normal(D)
    p = p - (p @ d_hat) * d_hat
    p = p - p.mean()
    p = p / np.linalg.norm(p)

    gclip = np.clip(g, -1.0, 1.0)
    beta_c = np.sqrt(np.maximum(0.0, 1.0 - gclip ** 2))
    U = gclip[:, None] * d_hat[None, :] + beta_c[:, None] * p[None, :]   # [V,32]
    # enforce exactly zero mean and unit variance per row
    U = U - U.mean(axis=1, keepdims=True)
    U = U / np.maximum(np.sqrt((U ** 2).mean(axis=1, keepdims=True)), 1e-8)

    # ---------- 4. write weights ----------
    nw = {k: np.array(vv, dtype=np.float32, copy=True) for k, vv in w.items()}

    nw["embed.weight"][:] = U.astype(np.float32)
    nw["pos.weight"][:] = 0.0

    for bname in ["block0", "block1"]:
        for key in ["qkv.weight", "qkv.bias", "proj.weight", "proj.bias",
                    "ff1.weight", "ff1.bias", "ff2.weight", "ff2.bias"]:
            nw[bname + "." + key] = np.zeros_like(nw[bname + "." + key])
        nw[bname + ".norm1.weight"][:] = 1.0
        nw[bname + ".norm1.bias"][:] = 0.0
        nw[bname + ".norm2.weight"][:] = 1.0
        nw[bname + ".norm2.bias"][:] = 0.0

    nw["norm.weight"][:] = 1.0
    nw["norm.bias"][:] = 0.0

    hw = np.zeros((2, D), dtype=np.float32)
    hw[0] = (-0.5 * s * d_hat).astype(np.float32)
    hw[1] = (0.5 * s * d_hat).astype(np.float32)
    nw["head.weight"] = hw
    hb = np.zeros((2,), dtype=np.float32)
    hb[0] = np.float32(-0.5 * c)
    hb[1] = np.float32(0.5 * c)
    nw["head.bias"] = hb

    # ---------- 5. sanity check with an exact numpy forward pass ----------
    def forward(ids, W):
        B = ids.shape[0]
        h = W["embed.weight"][ids] + W["pos.weight"][None, :, :]
        for bn in ["block0", "block1"]:
            z = layer_norm(h, W[bn + ".norm1.weight"], W[bn + ".norm1.bias"])
            qkv = z @ W[bn + ".qkv.weight"].T + W[bn + ".qkv.bias"]
            q, k, v = np.split(qkv, 3, axis=-1)
            B_, L_, D_ = q.shape
            hd = D_ // 2
            q = q.reshape(B_, L_, 2, hd).transpose(0, 2, 1, 3)
            k = k.reshape(B_, L_, 2, hd).transpose(0, 2, 1, 3)
            v = v.reshape(B_, L_, 2, hd).transpose(0, 2, 1, 3)
            sc = q @ k.transpose(0, 1, 3, 2) / math.sqrt(hd)
            pad = (ids == 0)
            sc = np.where(pad[:, None, None, :], -1e9, sc)
            sc = sc - sc.max(axis=-1, keepdims=True)
            e = np.exp(sc)
            a = e / e.sum(axis=-1, keepdims=True)
            a = a * (~pad)[:, None, None, :]
            a = a / np.maximum(a.sum(axis=-1, keepdims=True), 1e-9)
            ctx = (a @ v).transpose(0, 2, 1, 3).reshape(B_, L_, D_)
            h = h + ctx @ W[bn + ".proj.weight"].T + W[bn + ".proj.bias"]
            z2 = layer_norm(h, W[bn + ".norm2.weight"], W[bn + ".norm2.bias"])
            u1 = np.maximum(z2 @ W[bn + ".ff1.weight"].T + W[bn + ".ff1.bias"], 0.0)
            h = h + u1 @ W[bn + ".ff2.weight"].T + W[bn + ".ff2.bias"]
        h = layer_norm(h, W["norm.weight"], W["norm.bias"])
        m_ = (ids != 0).astype(np.float32)
        pooled = (h * m_[..., None]).sum(axis=1) / np.maximum(m_.sum(axis=1, keepdims=True), 1.0)
        return pooled @ W["head.weight"].T + W["head.bias"]

    def layer_norm(t, g_, b_):
        mu = t.mean(axis=-1, keepdims=True)
        var = t.var(axis=-1, keepdims=True)
        return (t - mu) / np.sqrt(var + 1e-5) * g_ + b_

    W32 = {k: np.asarray(vv, dtype=np.float32) for k, vv in nw.items()}
    lg = forward(x[:512], W32)
    pred = lg.argmax(axis=1)
    acc = float((pred == y[:512]).mean())
    p1 = np.exp(lg - lg.max(axis=1, keepdims=True))
    p1 = p1 / p1.sum(axis=1, keepdims=True)
    ce = float(-np.mean(np.log(np.clip(p1[np.arange(512), y[:512]], 1e-9, 1.0))))

    # also verify the closed-form equivalence: score == s*mean(gamma_raw)+c
    sc_ref = (Phi[:512] @ gamma_raw + c)
    sc_chk = (lg[:, 1] - lg[:, 0])
    corr = float(np.corrcoef(sc_ref, sc_chk)[0, 1]) if sc_chk.std() > 0 else 0.0

    for k in nw:
        nw[k] = np.clip(np.asarray(nw[k], dtype=np.float32), -1000.0, 1000.0)
        nw[k] = np.nan_to_num(nw[k], nan=0.0, posinf=0.0, neginf=0.0)

    diagnostics = {
        "ridge_lambda": float(lam),
        "ridge_gcv": float(gcv),
        "train_acc_check512": acc,
        "train_ce_check512": ce,
        "score_corr_vs_closed_form": corr,
        "n_words_nonzero_gamma": int(np.sum(np.abs(gamma_raw) > 1e-9)),
        "gamma_max_abs": float(m),
        "intercept_c": float(c),
        "head_scale_s": float(s),
        "embed_row_mean_abs": float(np.abs(U.mean(axis=1)).max()),
        "embed_row_std_range": [float(U.std(axis=1).min()), float(U.std(axis=1).max())],
        "top_positive_words": [str(i) for i in np.argsort(-gamma_raw)[:15]],
        "top_negative_words": [str(i) for i in np.argsort(gamma_raw)[:15]],
        "note": "blocks zeroed -> identity; pos=0; model = exact unigram-count linear classifier s*mean_t g(token)+c"
    }
    return {"weights": nw, "diagnostics": diagnostics}
