import numpy as np

def edit(w, f):
    X = np.asarray(f["x"], dtype=np.int64)          # [N,64] token ids
    y = np.asarray(f["y"], dtype=np.int64)          # [N]
    N, L = X.shape
    V = 2000
    D = 32

    # ---------------- 1) binary bag-of-words design matrix ----------------
    rows = np.repeat(np.arange(N), L)
    cols = X.ravel()
    vals = np.ones(rows.shape[0], dtype=np.float64)
    B = np.zeros((N, V), dtype=np.float64)
    np.add.at(B, (rows, cols), vals)
    B[:, 0] = 0.0                                   # drop PAD column
    B = (B > 0).astype(np.float64)                  # binary presence
    Bc = B - B.mean(axis=0, keepdims=True)
    t = y.astype(np.float64)
    t = 2.0 * t - 1.0                              # +-1 targets

    # ---------------- 2) closed-form ridge fit of token weights -------------
    # centre features and targets, solve normal equations for ridge
    ybar = t.mean()
    tc = t - ybar
    G = Bc.T @ Bc
    c = Bc.T @ tc
    lam_grid = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    best = None
    for lam in lam_grid:
        A = G + lam * np.eye(V)
        try:
            wt = np.linalg.solve(A, c)
        except Exception:
            continue
        s = Bc @ wt
        acc = float(np.mean((s > 0) == (t > 0)))
        if best is None or acc > best[0]:
            best = (acc, lam, wt)
    acc_lin, lam, wt = best
    wt = wt / max(1e-9, np.max(np.abs(wt)))       # normalise to [-1,1]

    # ---------------- 3) build the "identity" network -----------------------
    W = {k: np.array(v, dtype=np.float64) for k, v in w.items()}

    # embeddings: token weight along dim0, big constant offset along dim1
    off = 40.0
    E = np.zeros((V, D), dtype=np.float64)
    E[:, 0] = wt
    E[:, 1] = off
    W["embed.weight"] = E
    W["pos.weight"] = np.zeros((64, D), dtype=np.float64)

    for blk in ("block0", "block1"):
        W[blk + ".qkv.weight"] = np.zeros((96, D), dtype=np.float64)
        W[blk + ".qkv.bias"] = np.zeros(96, dtype=np.float64)
        W[blk + ".proj.weight"] = np.zeros((D, D), dtype=np.float64)
        W[blk + ".proj.bias"] = np.zeros(D, dtype=np.float64)
        W[blk + ".ff1.weight"] = np.zeros((64, D), dtype=np.float64)
        W[blk + ".ff1.bias"] = np.zeros(64, dtype=np.float64)
        W[blk + ".ff2.weight"] = np.zeros((D, 64), dtype=np.float64)
        W[blk + ".ff2.bias"] = np.zeros(D, dtype=np.float64)
        W[blk + ".norm1.weight"] = np.ones(D, dtype=np.float64)
        W[blk + ".norm1.bias"] = np.zeros(D, dtype=np.float64)
        W[blk + ".norm2.weight"] = np.ones(D, dtype=np.float64)
        W[blk + ".norm2.bias"] = np.zeros(D, dtype=np.float64)

    # final LayerNorm: identity affine
    W["norm.weight"] = np.ones(D, dtype=np.float64)
    W["norm.bias"] = np.zeros(D, dtype=np.float64)

    # ---------------- 4) numpy forward pass (to get pooled features) --------
    def forward_pooled(Wd, Xx):
        Bx, Lx = Xx.shape
        h = Wd["embed.weight"][Xx] + Wd["pos.weight"][None, :, :]
        mask = (Xx != 0).astype(np.float64)[:, :, None]
        for blk in ("block0", "block1"):
            z = layer_norm(h, Wd[blk + ".norm1.weight"], Wd[blk + ".norm1.bias"])
            qkv = z @ Wd[blk + ".qkv.weight"].T + Wd[blk + ".qkv.bias"]
            q, k, v = np.split(qkv, 3, axis=-1)
            # 2 heads of width 16
            att = np.zeros((Bx, Lx, Lx))
            for hd in range(2):
                sl = slice(hd * 16, (hd + 1) * 16)
                sc = q[:, :, sl] @ k[:, :, sl].transpose(0, 2, 1) / 4.0
                sc = np.where(mask[:, 0, :] > 0, sc, -1e9)
                e = np.exp(sc - sc.max(axis=-1, keepdims=True))
                e = e * mask[:, 0, :][:, None, :]
                a = e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-12)
                att[:, :, sl] = a @ v[:, :, sl]
            h = h + att @ Wd[blk + ".proj.weight"].T + Wd[blk + ".proj.bias"]
            z2 = layer_norm(h, Wd[blk + ".norm2.weight"], Wd[blk + ".norm2.bias"])
            f1 = np.maximum(z2 @ Wd[blk + ".ff1.weight"].T + Wd[blk + ".ff1.bias"], 0.0)
            h = h + f1 @ Wd[blk + ".ff2.weight"].T + Wd[blk + ".ff2.bias"]
        hn = layer_norm(h, Wd["norm.weight"], Wd["norm.bias"])
        denom = np.maximum(mask.sum(axis=1), 1.0)
        return (hn * mask).sum(axis=1) / denom

    def layer_norm(h, g, b):
        mu = h.mean(axis=-1, keepdims=True)
        var = h.var(axis=-1, keepdims=True)
        return (h - mu) / np.sqrt(var + 1e-5) * g + b

    # ---------------- 5) closed-form head (ridge LS on +-1 targets) ----------
    P = forward_pooled(W, X)
    Pc = P - P.mean(axis=0, keepdims=True)
    H = Pc.T @ Pc
    hc = Pc.T @ tc
    best_head = None
    for hl in [1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        A = H + hl * np.eye(D)
        try:
            beta = np.linalg.solve(A, hc)
        except Exception:
            continue
        s = Pc @ beta
        a = float(np.mean((s > 0) == (t > 0)))
        if best_head is None or a > best_head[0]:
            best_head = (a, hl, beta)
    acc_head, hl, beta = best_head
    # centre the head so that the intercept is the class prior
    b0 = float(ybar)
    W["head.weight"] = np.stack([-beta, beta], axis=0)
    W["head.bias"] = np.array([-b0, b0], dtype=np.float64)

    # ---------------- 6) final sanity check on the implemented net ---------
    Pn = forward_pooled(W, X)
    lg = Pn @ W["head.weight"].T + W["head.bias"]
    pred = lg.argmax(axis=1)
    acc_impl = float(np.mean(pred == y))
    lg_old = np.asarray(f["logits"], dtype=np.float64)
    acc_old = float(np.mean(lg_old.argmax(axis=1) == y))

    out = {}
    for k, v in W.items():
        arr = np.asarray(v, dtype=np.float32)
        arr = np.clip(arr, -1000.0, 1000.0)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1000.0, neginf=-1000.0)
        out[k] = arr.astype(np.float32)

    diag = {
        "acc_before": round(acc_old, 4),
        "acc_after": round(acc_impl, 4),
        "acc_ridge_tokenfit": round(acc_lin, 4),
        "ridge_lambda": float(lam),
        "head_ridge": float(hl),
        "n_tokens_used": int(np.sum(np.abs(wt) > 1e-6)),
        "top_pos_tokens": [int(i) for i in np.argsort(-wt)[:15]],
        "top_neg_tokens": [int(i) for i in np.argsort(wt)[:15]],
        "pooled_dim0_std": float(np.std(Pn[:, 0])),
        "pooled_dim1_mean": float(np.mean(Pn[:, 1])),
        "corr_pooled0_label": float(np.corrcoef(Pn[:, 0], t)[0, 1]),
        "logit_margin_mean": float(np.mean(lg[:, 1] - lg[:, 0])),
    }
    return {"weights": out, "diagnostics": diag}
