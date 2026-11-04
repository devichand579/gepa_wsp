import numpy as np

def edit(w, f):
    X = np.asarray(f["x"], dtype=np.int64)          # [N,64] token ids
    Y = np.asarray(f["y"], dtype=np.int64)          # [N]
    N, L = X.shape
    V = w["embed.weight"].shape[0]
    D = w["embed.weight"].shape[1]

    # ---------------- 1. per-token sentiment scores (closed-form NB-style contrast) -------------
    mask = (X > 0)                                  # non-pad
    tok_count = np.zeros(V, dtype=np.float64)
    tok_pos   = np.zeros(V, dtype=np.float64)
    np.add.at(tok_count, X.ravel(), mask.ravel().astype(np.float64))
    np.add.at(tok_pos,   X.ravel(), (mask.ravel() * Y.ravel()).astype(np.float64))

    n_pos = float((Y == 1).sum())
    p_pos = n_pos / N
    # rows containing token t
    has = np.zeros(V, dtype=np.float64)
    np.add.at(has, X.ravel(), mask.ravel().astype(np.float64))
    # rows NOT containing token t (approximate via total rows minus rows containing)
    rows_with = has
    rows_without = N - rows_with
    pos_with = tok_pos
    neg_with = tok_count - tok_pos
    pos_without = n_pos - pos_with
    neg_without = (N - n_pos) - neg_with

    eps = 1.0
    rate_pos = (pos_with + eps) / (rows_with + 2 * eps)
    rate_neg = (neg_with + eps) / (rows_with + 2 * eps)
    rate_pos0 = (pos_without + eps) / (rows_without + 2 * eps)
    rate_neg0 = (neg_without + eps) / (rows_without + 2 * eps)

    score = np.log(rate_pos / rate_neg) - np.log(rate_pos0 / rate_neg0)
    # shrink by count (Laplace-ish), keep tokens that actually occur
    cnt = tok_count
    shrink = cnt / (cnt + 4.0)
    score = score * shrink
    score[0] = 0.0            # pad: never pooled anyway
    score[1] = 0.0            # <unk>: neutral
    score = np.clip(score, -3.0, 3.0)

    # ---------------- 2. embed scores into the network ----------------------------------------
    cvec = np.zeros(D, dtype=np.float64)
    cvec[0] = 1.0                                   # constant direction (orthogonal to d)
    dvec = np.zeros(D, dtype=np.float64)
    dvec[1] = 1.0                                   # signal direction
    cvec /= np.linalg.norm(cvec)
    dvec -= dvec.dot(cvec) * cvec
    dvec /= np.linalg.norm(dvec)

    cscale = 25.0
    C = cscale * cvec                              # [D]
    E = np.zeros((V, D), dtype=np.float64)
    E[:] = C[None, :]
    E += score[:, None] * dvec[None, :]
    E[0, :] = C                                    # pad row neutral
    E[1, :] = C                                    # unk row neutral

    w["embed.weight"] = E.astype(np.float32)
    w["pos.weight"] = np.zeros_like(w["pos.weight"])
    for b in ("block0", "block1"):
        w[b + ".proj.weight"] = np.zeros_like(w[b + ".proj.weight"])
        w[b + ".proj.bias"] = np.zeros_like(w[b + ".proj.bias"])
        w[b + ".ff2.weight"] = np.zeros_like(w[b + ".ff2.weight"])
        w[b + ".ff2.bias"] = np.zeros_like(w[b + ".ff2.bias"])
    w["norm.weight"] = np.ones_like(w["norm.weight"])
    w["norm.bias"] = np.zeros_like(w["norm.bias"])

    # ---------------- 3. forward pass with my weights (numpy) ---------------------------------
    def forward(emb):
        h = emb[X]                                        # [N,L,D]
        mu = h.mean(axis=-1, keepdims=True)
        var = h.var(axis=-1, keepdims=True)
        hn = (h - mu) / np.sqrt(var + 1e-5)
        m = mask.astype(np.float64)[:, :, None]
        denom = np.maximum(m.sum(axis=1, keepdims=True), 1.0)
        pooled = (hn * m).sum(axis=1) / denom             # [N,D]
        return pooled

    pooled = forward(E)

    # ---------------- 4. closed-form head: ridge LS of y on [pooled, 1] ------------------------
    A = np.concatenate([pooled, np.ones((N, 1))], axis=1)  # [N, D+1]
    t = (Y.astype(np.float64) * 2.0 - 1.0)                # +-1 targets
    lam = 1.0
    G = A.T @ A
    reg = lam * np.eye(G.shape[0])
    reg[-1, -1] = 0.0
    try:
        beta = np.linalg.solve(G + reg, A.T @ t)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(A, t, rcond=None)[0]

    # head.weight rows: [negative, positive]; logits = W @ pooled + b
    hw = np.zeros((2, D), dtype=np.float64)
    hw[1] = 0.5 * beta[:D]
    hw[0] = -0.5 * beta[:D]
    hb = np.zeros(2, dtype=np.float64)
    hb[1] = 0.5 * beta[D]
    hb[0] = -0.5 * beta[D]

    # keep magnitudes sane and finite
    def clip(a, lim=1000.0):
        return np.clip(np.nan_to_num(a, nan=0.0, posinf=lim, neginf=-lim), -lim, lim)

    w["head.weight"] = clip(hw).astype(np.float32)
    w["head.bias"] = clip(hb).astype(np.float32)

    # ---------------- 5. diagnostics ----------------------------------------------------------
    logits = pooled @ w["head.weight"].T + w["head.bias"][None, :]
    pred = logits.argmax(axis=1)
    acc = float((pred == Y).mean())
    # analytic check of the intended mechanism: mean token score vs target
    ssum = np.zeros(N)
    for i in range(N):
        ids = X[i][mask[i]]
        ssum[i] = score[ids].sum() if ids.size else 0.0
    mean_score = ssum / np.maximum(mask.sum(axis=1), 1)
    nb_acc = float((((mean_score > 0).astype(np.int64)) == Y).mean())
    corr = float(np.corrcoef(mean_score, (Y * 2 - 1))[0, 1])
    pooled_norm = float(np.linalg.norm(pooled, axis=1).mean())

    return {"weights": w,
            "diagnostics": {
                "train_acc_after": acc,
                "nb_unigram_acc": nb_acc,
                "corr_mean_tok_score_label": corr,
                "mean_pooled_norm": pooled_norm,
                "head_w_absmax": float(np.abs(w["head.weight"]).max()),
                "embed_absmax": float(np.abs(w["embed.weight"]).max()),
                "score_absmax": float(np.abs(score).max()),
                "n_tokens_used": int((cnt > 0).sum()),
                "mean_len": float(mask.sum(axis=1).mean()),
                "prior_pos": p_pos,
            }}
