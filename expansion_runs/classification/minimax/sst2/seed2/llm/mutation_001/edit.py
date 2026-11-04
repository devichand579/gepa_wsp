import numpy as np

def edit(w, f):
    x = np.asarray(f["x"], dtype=np.int64)
    y = np.asarray(f["y"], dtype=np.int64)
    V = int(w["embed.weight"].shape[0])
    D = int(w["embed.weight"].shape[1])
    N, L = x.shape

    # ---------------- masks ----------------
    mask = (x != 0).astype(np.float64)          # valid token positions
    lens = mask.sum(1).clip(1.0)
    ybin = (y == 1).astype(np.float64)
    tvec = 2.0 * ybin - 1.0                     # +1 positive, -1 negative

    # ---------------- token statistics ----------------
    tok_count = np.zeros(V, dtype=np.float64)
    tok_pos = np.zeros(V, dtype=np.float64)
    for i in range(N):
        ids = x[i][mask[i] > 0]
        if ids.size:
            np.add.at(tok_count, ids, 1.0)
            np.add.at(tok_pos, ids, ybin[i])

    # ---------------- closed-form ridge BoW on token counts ----------------
    # design: counts / sqrt(L)  (length-normalised), plus intercept
    Xb = np.zeros((N, V + 1), dtype=np.float64)
    for i in range(N):
        ids = x[i][mask[i] > 0]
        if ids.size:
            np.add.at(Xb[i], ids, 1.0 / np.sqrt(ids.size))
    Xb[:, V] = 1.0
    lam = 5.0
    A = Xb.T @ Xb
    reg = lam * np.eye(V + 1)
    reg[V, V] = 0.0
    beta = np.linalg.solve(A + reg, Xb.T @ tvec)
    tok_score = beta[:V]                        # per-token sentiment score

    # ---------------- content codes for tokens ----------------
    rng = np.random.default_rng(12345)
    codes = rng.standard_normal((V, D)).astype(np.float64)
    codes[0] = 0.0                              # pad token unused
    codes -= codes.mean(0, keepdims=True)

    # ---------------- build normalised embeddings ----------------
    # desired normalised vector g_t: dim0 = s_t (sentiment), dims1.. = content code
    # constraints: sum_d g_t[d] = 0, mean_d g_t[d]^2 = 1
    s = np.clip(tok_score, -2.0, 2.0)
    g = np.zeros((V, D), dtype=np.float64)
    g[:, 0] = s
    rest = codes[:, 1:].copy()                  # (V, D-1)
    rest_mean = rest.mean(1, keepdims=True)
    rest = rest - rest_mean
    # need sum(rest) = -s  and  sum(rest^2) = D - s^2
    rest_normsq = np.maximum((rest ** 2).sum(1), 1e-9)
    # project rest onto the direction that carries the needed sum
    ones = np.ones((V, D - 1)) / (D - 1)
    r_par = (rest @ ones)[:, None] * ones       # component along ones
    r_perp = rest - r_par
    perp_normsq = np.maximum((r_perp ** 2).sum(1), 1e-9)
    need_sum = -s                               # required sum of rest
    # rest = r_perp + a*ones ; sum(rest) = a ; so a = need_sum
    a = need_sum
    base = r_perp + a[:, None] * ones
    base_normsq = np.maximum((base ** 2).sum(1), 1e-9)
    target_normsq = np.maximum(D - s ** 2, 1e-6)
    scale = np.sqrt(target_normsq / base_normsq)
    rest = base * scale[:, None]
    g[:, 1:] = rest
    # enforce exact zero-mean and unit-variance numerically
    g -= g.mean(1, keepdims=True)
    g /= np.maximum(np.sqrt((g ** 2).mean(1, keepdims=True)), 1e-9)

    # realise as embeddings with constant mean m and constant std sd
    m = 0.0
    sd = 1.0
    E = g * sd + m                              # (V, D)
    E[0] = 0.0                                  # pad row -> zeros

    # ---------------- write weights ----------------
    w["embed.weight"] = E.astype(np.float32)
    w["pos.weight"] = np.zeros_like(w["pos.weight"])

    for blk in ("block0", "block1"):
        for k in ("qkv.weight", "qkv.bias", "proj.weight", "proj.bias",
                  "ff1.weight", "ff1.bias", "ff2.weight", "ff2.bias"):
            w[blk + "." + k] = np.zeros_like(w[blk + "." + k])
        for k in ("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"):
            arr = np.zeros_like(w[blk + "." + k])
            if k.endswith("weight"):
                arr[:] = 1.0
            w[blk + "." + k] = arr

    w["norm.weight"] = np.ones_like(w["norm.weight"])
    w["norm.bias"] = np.zeros_like(w["norm.bias"])

    # ---------------- recompute true pooled features with new weights ----------------
    # blocks are identities -> h = embed(ids); final LayerNorm with w=1,b=0
    # pooled = masked mean of LayerNorm(embed_t)
    # LayerNorm(embed_t) = (embed_t - mean_d)/std_d ; embed rows have constant mean/std
    # so pooled = mean_t g_t  (exactly, up to float error)
    pooled = np.zeros((N, D), dtype=np.float64)
    for i in range(N):
        ids = x[i][mask[i] > 0]
        if ids.size:
            pooled[i] = g[ids].mean(0)

    # ---------------- ridge fit of head on true pooled features ----------------
    H = np.concatenate([pooled, np.ones((N, 1))], axis=1)
    lamh = 1e-3
    Ah = H.T @ H
    regh = lamh * np.eye(D + 1)
    regh[D, D] = 0.0
    theta = np.linalg.solve(Ah + regh, H.T @ tvec)
    head_w = theta[:D]
    head_b = theta[D]

    w["head.weight"] = np.stack([-head_w, head_w]).astype(np.float32)
    w["head.bias"] = np.array([-head_b, head_b], dtype=np.float32)

    # ---------------- diagnostics ----------------
    logits = pooled @ head_w + head_b
    pred = (logits > 0).astype(np.int64)
    acc = float((pred == y).mean())
    # BoW-only sanity check
    bow = Xb @ beta
    bow_acc = float((((bow > 0).astype(np.int64)) == y).mean())
    # margin stats
    marg = logits[np.abs(tvec) > 0]
    diag = {
        "design": "identity blocks, zero pos, exact-affine LayerNorm embeddings, ridge head",
        "n_tokens_used": int((tok_count > 0).sum()),
        "tok_score_absmax": float(np.abs(tok_score).max()),
        "tok_score_std": float(tok_score.std()),
        "bow_train_acc": bow_acc,
        "head_train_acc": acc,
        "logit_std": float(logits.std()),
        "margin_mean": float(marg.mean()),
        "margin_absmean": float(np.abs(marg).mean()),
        "frac_correct": acc,
        "pooled_norm_mean": float(np.linalg.norm(pooled, axis=1).mean()),
        "embed_absmax": float(np.abs(E).max()),
        "head_w_absmax": float(np.abs(head_w).max()),
        "head_b": float(head_b),
        "corr_logit_label": float(np.corrcoef(logits, tvec)[0, 1]),
    }
    # sanity: all finite, bounded
    for k, v in w.items():
        arr = np.asarray(v)
        if not np.all(np.isfinite(arr)):
            diag["nonfinite_" + k] = True
        diag.setdefault("max_abs", 0.0)
        diag["max_abs"] = max(diag["max_abs"], float(np.abs(arr).max()))
    return {"weights": w, "diagnostics": diag}
