import numpy as np

def edit(w, f):
    x = np.asarray(f["x"], dtype=np.int64)          # [N,64]
    y = np.asarray(f["y"], dtype=np.int64)          # [N]
    N, L = x.shape
    V = w["embed.weight"].shape[0]

    # ---------- token statistics (mean-normalized counts) ----------
    mask = (x != 0).astype(np.float64)
    Ln = mask.sum(1)                                # true token counts
    Ln[Ln == 0] = 1.0
    cnt = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), x.reshape(-1).shape[0])
    np.add.at(cnt, (rows, x.reshape(-1)), 1.0)
    mean_cnt = cnt / Ln[:, None]

    t = 2.0 * y.astype(np.float64) - 1.0            # +/-1 targets

    # ---------- closed-form ridge on mean-normalized counts ----------
    def ridge_fit(A, t, lam):
        d = A.shape[1]
        G = A.T @ A
        G.flat[:: d + 1] += lam
        b = A.T @ t
        return np.linalg.solve(G, b)

    def acc_of(beta, b0, A):
        s = A @ beta + b0
        return float(np.mean(np.sign(s) == t))

    # intercept handled by centering
    mu = mean_cnt.mean(0)
    Ac = mean_cnt - mu
    tc = t - t.mean()

    # split-half CV over lambda
    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    h1, h2 = perm[: N // 2], perm[N // 2:]
    lams = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    cv = []
    for lam in lams:
        a1, a2 = [], []
        for tr, te in ((h1, h2), (h2, h1)):
            A_tr = Ac[tr]
            beta = ridge_fit(A_tr, tc[tr], lam)
            b0 = t.mean() - mu @ beta
            a1.append(acc_of(beta, b0, Ac[te]))
        cv.append(float(np.mean(a1)))
    lam_best = lams[int(np.argmax(cv))]
    beta = ridge_fit(Ac, tc, lam_best)
    b0 = float(t.mean() - mu @ beta)

    # ---------- encode scores into embeddings (magnitude preserving) ----------
    # row layout: channel0 = score, channel1 = constant anchor, rest = 0
    C0 = 100.0
    emb = np.zeros((V, 32), dtype=np.float64)
    emb[:, 0] = beta
    emb[:, 1] = C0
    emb[0, :] = 0.0                                   # PAD row -> all zeros (masked out anyway)

    # ---------- forward pass (blocks are identity: all their weights are 0) ----------
    def forward_pooled(emb_w):
        h = emb_w[x]                                   # [N,64,32]
        m = h.mean(-1, keepdims=True)
        v = h - m
        sd = np.sqrt((v * v).mean(-1, keepdims=True) + 1e-5)
        z = v / sd
        mm = mask[:, :, None]
        pooled = (z * mm).sum(1) / Ln[:, None]
        return pooled

    pooled = forward_pooled(emb)

    # ---------- closed-form head fit on pooled features ----------
    d = pooled.shape[1]
    G = pooled.T @ pooled
    G.flat[:: d + 1] += 1e-3
    B = np.linalg.solve(G, pooled.T @ t)
    bb = float(t.mean() - pooled.mean(0) @ B)

    # ---------- assemble weights ----------
    out = {}
    for k, v in w.items():
        out[k] = np.array(v, dtype=np.float32, copy=True)

    out["embed.weight"] = emb.astype(np.float32)
    out["pos.weight"] = np.zeros_like(out["pos.weight"])
    out["norm.weight"] = np.ones_like(out["norm.weight"])
    out["norm.bias"] = np.zeros_like(out["norm.bias"])
    for blk in ("block0", "block1"):
        for nm in ("qkv", "proj", "ff1", "ff2"):
            out[blk + "." + nm + ".weight"] = np.zeros_like(out[blk + "." + nm + ".weight"])
            out[blk + "." + nm + ".bias"] = np.zeros_like(out[blk + "." + nm + ".bias"])
        for nm in ("norm1", "norm2"):
            out[blk + "." + nm + ".weight"] = np.ones_like(out[blk + "." + nm + ".weight"])
            out[blk + "." + nm + ".bias"] = np.zeros_like(out[blk + "." + nm + ".bias"])

    out["head.weight"] = B.astype(np.float32).reshape(2, 32)
    out["head.bias"] = np.array([bb, -bb], dtype=np.float32)

    # ---------- diagnostics ----------
    logits = pooled @ B + bb
    pred = (logits > 0).astype(np.int64)
    acc = float(np.mean(pred == y))
    ce = float(np.mean(np.logaddexp(0.0, -t * logits)))
    bidx = np.asarray(f["batch.ids"], dtype=np.int64)
    by = np.asarray(f["batch.y"], dtype=np.int64)
    bacc = float(np.mean(pred[bidx] == by))
    bce = float(np.mean(np.logaddexp(0.0, -t[bidx] * logits[bidx])))
    diag = {
        "design": "ridge on mean-normalized token counts -> embed ch0=score, ch1=const anchor; head = closed-form LS on pooled",
        "lambda": lam_best,
        "cv_acc": cv,
        "train_acc": acc,
        "train_ce": ce,
        "batch_acc": bacc,
        "batch_ce": bce,
        "beta_absmax": float(np.abs(beta).max()),
        "beta_nonzero": int(np.count_nonzero(beta)),
        "pooled0_std": float(pooled[:, 0].std()),
        "pooled0_classmean": [float(pooled[y == 0, 0].mean()), float(pooled[y == 1, 0].mean())],
        "head_row0_absmax": float(np.abs(B[0]).max()),
        "embed_absmax": float(np.abs(emb).max()),
    }

    # ---------- safety: finite, bounded, float32 ----------
    for k in out:
        arr = np.asarray(out[k], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(arr, -1000.0, 1000.0, out=arr)
        out[k] = arr

    return {"weights": out, "diagnostics": diag}
