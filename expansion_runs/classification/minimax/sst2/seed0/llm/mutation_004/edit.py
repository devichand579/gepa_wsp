import numpy as np

def edit(w, f):
    # ---------------- data ----------------
    X = np.asarray(f["x"], dtype=np.int64)          # [N,64] token ids
    Y = np.asarray(f["y"], dtype=np.int64)          # [N] labels
    N, Lmax = X.shape
    V = w["embed.weight"].shape[0]
    D = w["embed.weight"].shape[1]

    mask = (X != 0)
    L = mask.sum(1).astype(np.float64)
    L[L == 0] = 1.0

    # ---------------- bag-of-words (binary presence) ----------------
    rows = np.repeat(np.arange(N), L.astype(np.int64))
    cols = X[mask]
    B = np.zeros((N, V), dtype=np.float64)
    np.add.at(B, (rows, cols), 1.0)

    # ---------------- class moments: NB log-odds ----------------
    y1 = (Y == 1)
    cnt1 = B[y1].sum(0)
    cnt0 = B[~y1].sum(0)
    alpha = 1.0
    tot1 = cnt1.sum() + alpha * V
    tot0 = cnt0.sum() + alpha * V
    nb = np.log((cnt1 + alpha) / tot1) - np.log((cnt0 + alpha) / tot0)

    # ---------------- ridge on mean-normalised presence ----------------
    A = B / L[:, None]                              # [N,V] mean presence
    t = np.where(y1, 1.0, -1.0)
    AtA = A.T @ A
    AtA[np.diag_indices(V)] += 1.0                  # ridge on raw-count scale
    beta = np.linalg.solve(AtA, A.T @ t)
    # blend class-moment estimate with ridge estimate (both closed form)
    s_tok = 0.5 * (beta / (np.abs(beta).max() + 1e-9)) + 0.5 * (nb / (np.abs(nb).max() + 1e-9))

    # scale so |s| stays in the near-linear regime of the per-token LayerNorm
    mx = np.abs(s_tok).max()
    if mx > 1e-9:
        s_tok = s_tok * (0.45 / mx)
    s_tok[0] = 0.0                                  # pad token carries no signal

    # ---------------- build the bag-of-words encoder ----------------
    E = np.zeros((V, D), dtype=np.float64)
    E[:, 0] = s_tok
    E[:, 1] = 1.0
    E[:, 2] = -1.0

    # ---------------- exact NumPy forward pass (blocks zeroed) ----------------
    def forward(emb, posw, ids):
        h = emb[ids] + posw[None, :, :]
        # block0 / block1 are identity (all qkv/proj/ff1/ff2 weights & biases = 0)
        m = (ids != 0)
        cnt = m.sum(1).astype(np.float64)
        cnt[cnt == 0] = 1.0
        mu = (h * m[:, :, None]).sum(1) / cnt[:, None]
        d = h - mu[:, None, :]
        var = (d * d * m[:, :, None]).sum(1) / cnt[:, None]
        inv = 1.0 / np.sqrt(var + 1e-5)
        hn = d * inv[:, :, None]
        pooled = (hn * m[:, :, None]).sum(1) / cnt[:, None]
        return pooled

    posw = np.zeros((Lmax, D), dtype=np.float64)
    pooled = forward(E, posw, X)                    # [N,D]

    # ---------------- ridge fit of the head on the realised features ----------------
    G = pooled.mean(0)
    C = pooled - G
    S = C.T @ C
    S[np.diag_indices(D)] += 1e-3
    Sc = C.T @ t
    wv = np.linalg.solve(S, Sc)
    b = float(t.mean() - G @ wv)
    # orient so that positive label -> positive logit
    if (wv @ (pooled[y1].mean(0) - pooled[~y1].mean(0))) < 0:
        wv = -wv
        b = -b

    # ---------------- write weights ----------------
    out = {k: np.array(v, dtype=np.float32, copy=True) for k, v in w.items()}

    out["embed.weight"][:, :] = 0.0
    out["embed.weight"][:, 0] = s_tok.astype(np.float32)
    out["embed.weight"][:, 1] = 1.0
    out["embed.weight"][:, 2] = -1.0
    out["embed.weight"][0, :] = 0.0                # pad row neutral

    out["pos.weight"][:] = 0.0

    for bname in ("block0", "block1"):
        out[bname + ".qkv.weight"][:] = 0.0
        out[bname + ".qkv.bias"][:] = 0.0
        out[bname + ".proj.weight"][:] = 0.0
        out[bname + ".proj.bias"][:] = 0.0
        out[bname + ".ff1.weight"][:] = 0.0
        out[bname + ".ff1.bias"][:] = 0.0
        out[bname + ".ff2.weight"][:] = 0.0
        out[bname + ".ff2.bias"][:] = 0.0
        out[bname + ".norm1.weight"][:] = 1.0
        out[bname + ".norm1.bias"][:] = 0.0
        out[bname + ".norm2.weight"][:] = 1.0
        out[bname + ".norm2.bias"][:] = 0.0

    out["norm.weight"][:] = 1.0
    out["norm.bias"][:] = 0.0

    out["head.weight"][:] = 0.0
    out["head.weight"][1, :] = wv.astype(np.float32)
    out["head.weight"][0, :] = (-wv).astype(np.float32)
    out["head.bias"][1] = np.float32(b)
    out["head.bias"][0] = np.float32(-b)

    # ---------------- diagnostics ----------------
    for k in out:
        out[k] = np.clip(np.nan_to_num(out[k], nan=0.0, posinf=0.0, neginf=0.0), -1000.0, 1000.0).astype(np.float32)

    # realised logits with the written weights (recompute to be safe)
    emb = out["embed.weight"].astype(np.float64)
    pooled2 = forward(emb, np.zeros((Lmax, D)), X)
    lg = pooled2 @ wv + b
    pred = (lg > 0).astype(np.int64)
    acc = float((pred == Y).mean())
    p = 1.0 / (1.0 + np.exp(-np.clip(lg, -30, 30)))
    ce = float(-(y1 * np.log(np.clip(p, 1e-9, 1)) + (~y1) * np.log(np.clip(1 - p, 1e-9, 1))).mean())

    # top sentiment tokens (diagnostic only)
    vocab = None
    try:
        vocab = f["vocabulary_id_to_word"]
    except Exception:
        vocab = None

    diag = {
        "train_acc_new": acc,
        "train_ce_new": ce,
        "train_acc_old": float((np.asarray(f["logits"]).argmax(1) == Y).mean()),
        "mean_len": float(L.mean()),
        "tok_weight_std": float(np.std(s_tok)),
        "tok_weight_absmax": float(np.abs(s_tok).max()),
        "nb_absmax": float(np.abs(nb).max()),
        "head_w_norm": float(np.linalg.norm(wv)),
        "head_bias": float(b),
        "pooled_std": float(np.std(pooled2, axis=0)[:6].tolist() and np.std(pooled2, axis=0).tolist()[:6]),
        "pooled_mean": float(np.mean(pooled2)),
        "pooled_dim0_std": float(np.std(pooled2[:, 0])),
        "pooled_dim0_range": [float(pooled2[:, 0].min()), float(pooled2[:, 0].max())],
        "acc_pos": float((pred[y1] == 1).mean()),
        "acc_neg": float((pred[~y1] == 0).mean()),
        "design": "zeroed blocks + pos -> bag-of-words; embed[t]=[s_t,1,-1,0..]; ridge head on recomputed features",
    }
    if vocab is not None:
        idx = np.argsort(-s_tok)[:12]
        diag["top_positive_tokens"] = [(str(vocab[i]), round(float(s_tok[i]), 4)) for i in idx]
        idx = np.argsort(s_tok)[:12]
        diag["top_negative_tokens"] = [(str(vocab[i]), round(float(s_tok[i]), 4)) for i in idx]

    return {"weights": out, "diagnostics": diag}
