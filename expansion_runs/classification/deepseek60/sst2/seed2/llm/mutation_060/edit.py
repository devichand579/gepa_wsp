import numpy as np

def edit(w, f):
    x = np.asarray(f["x"])            # [N,64] int64
    y = np.asarray(f["y"]).astype(np.float64)   # [N]
    N, L = x.shape
    V = w["embed.weight"].shape[0]    # 2000

    # ---- 1. Build normalized bag-of-words counts (exclude PAD=0) ----
    mask = (x != 0)
    n = mask.sum(axis=1).astype(np.float64)
    n[n == 0] = 1.0
    C = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), L)
    cols = x.reshape(-1)
    valid = mask.reshape(-1)
    np.add.at(C, (rows[valid], cols[valid]), 1.0)
    X = C / n[:, None]                # [N,V]  (this is exactly `pooled` for a given embed)

    # ---- 2. Closed-form ridge with 5-fold CV over lambda ----
    t = 2.0 * y - 1.0                 # {-1,+1}
    Xm = X.mean(axis=0, keepdims=True)
    Xc = X - Xm
    ym = t.mean()
    tc = t - ym

    lam_grid = [1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    rng = np.random.RandomState(0)
    idx = rng.permutation(N)
    folds = np.array_split(idx, 5)

    best_lam, best_acc = None, -1.0
    for lam in lam_grid:
        accs = []
        for k in range(5):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(5) if j != k])
            Xtr, ttr = Xc[tr], tc[tr]
            Xte, tte = Xc[te], tc[te]
            G = Xtr.T @ Xtr + lam * np.eye(V)
            b = Xtr.T @ ttr
            try:
                wv = np.linalg.solve(G, b)
            except np.linalg.LinAlgError:
                wv = np.linalg.lstsq(G, b, rcond=None)[0]
            pred = Xte @ wv
            accs.append(np.mean((pred > 0) == (tte > 0)))
        a = float(np.mean(accs))
        if a > best_acc:
            best_acc, best_lam = a, lam

    # ---- 3. Final fit on all data ----
    G = Xc.T @ Xc + best_lam * np.eye(V)
    b = Xc.T @ tc
    try:
        wv = np.linalg.solve(G, b)
    except np.linalg.LinAlgError:
        wv = np.linalg.lstsq(G, b, rcond=None)[0]
    # intercept: t ≈ ym + (X - Xm) @ wv  =>  bias_term = ym - Xm @ wv
    bias = ym - float(Xm @ wv)

    # ---- 4. Realize wv via embed + head ----
    # score = d @ pooled + b0  where d = head[1]-head[0], b0 = head.bias[1]-head.bias[0]
    # We need d @ v_t = wv[t] for each token t, with v_t zero-mean, unit-var.
    # Choose d = e0 (first axis) in the zero-mean subspace; build v_t accordingly.
    # Use d = (e0 - 1/32) normalized so d @ 1 = 0 and |d| = 1.
    d = np.zeros(32, dtype=np.float64)
    d[0] = 1.0
    d -= d.mean()                      # zero-mean
    d /= np.linalg.norm(d)             # unit norm, still zero-mean

    # scale so that max |wv| fits inside the achievable range sqrt(32)
    wmax = np.max(np.abs(wv))
    if wmax < 1e-12:
        wmax = 1.0
    scale = (np.sqrt(32.0) * 0.999) / wmax
    wv_s = wv * scale
    # bias must scale with the same factor (score = scale*(...))
    bias_s = bias * scale

    # For each token t: v_t = wv_s[t]*d + u_t, with u_t ⊥ d, zero-mean, |v_t|^2 = 32.
    # Build an orthonormal basis of the zero-mean subspace orthogonal to d.
    # Zero-mean subspace: vectors orthogonal to ones(32).
    ones = np.ones(32, dtype=np.float64) / np.sqrt(32.0)
    # basis for complement of span{ones, d}
    # start from identity, project out ones and d
    B = np.eye(32, dtype=np.float64)
    B = B - np.outer(ones, ones) @ B
    B = B - np.outer(d, d) @ B
    # orthonormalize columns via QR on the nonzero part
    # Use SVD to get a clean orthonormal basis of the 30-dim subspace
    U, S, _ = np.linalg.svd(B, full_matrices=False)
    keep = S > 1e-8
    Q = U[:, keep]                     # [32, 30], orthonormal, ⊥ ones and ⊥ d
    m = Q.shape[1]                     # 30

    embed_new = np.zeros((V, 32), dtype=np.float64)
    # residual norm for each token
    resid2 = 32.0 - wv_s ** 2
    resid2 = np.clip(resid2, 0.0, None)
    resid = np.sqrt(resid2)
    # distribute residual along Q[:,0] (any fixed direction works; u_t only needs to be ⊥ d)
    embed_new = wv_s[:, None] * d[None, :] + resid[:, None] * Q[:, 0][None, :]

    # ---- 5. Write weights ----
    w["embed.weight"] = embed_new.astype(np.float32)

    # head: logits = head @ pooled + head.bias ; we want logit[1]-logit[0] = d @ pooled + b0
    # set head[0] = -d/2, head[1] = +d/2  (so difference = d, sum = 0)
    head = np.zeros((2, 32), dtype=np.float64)
    head[0, :] = -d / 2.0
    head[1, :] = +d / 2.0
    w["head.weight"] = head.astype(np.float32)
    w["head.bias"] = np.array([-bias_s / 2.0, +bias_s / 2.0], dtype=np.float32)

    # ensure pos and blocks stay zero (identity) -> already zero, but be explicit
    for k in ["pos.weight"]:
        if k in w:
            w[k] = np.zeros_like(w[k])
    for blk in ["block0", "block1"]:
        for suf in ["qkv.weight", "qkv.bias", "proj.weight", "proj.bias",
                    "ff1.weight", "ff1.bias", "ff2.weight", "ff2.bias"]:
            k = f"{blk}.{suf}"
            if k in w:
                w[k] = np.zeros_like(w[k])
    for k in ["norm.weight"]:
        if k in w:
            w[k] = np.ones_like(w[k])
    for k in ["norm.bias"]:
        if k in w:
            w[k] = np.zeros_like(w[k])

    # ---- diagnostics ----
    diag = {
        "selected_lambda": float(best_lam),
        "cv_accuracy": float(best_acc),
        "w_norm": float(np.linalg.norm(wv)),
        "w_max_abs": float(wmax),
        "scale": float(scale),
        "bias": float(bias),
        "vocab": int(V),
        "nonzero_tokens": int((np.abs(wv) > 1e-9).sum()),
        "embed_row_norm_mean": float(np.linalg.norm(embed_new, axis=1).mean()),
        "head_w_norm": float(np.linalg.norm(head)),
    }
    return {"weights": w, "diagnostics": diag}
