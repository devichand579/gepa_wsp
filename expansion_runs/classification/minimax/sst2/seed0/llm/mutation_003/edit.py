import numpy as np

def edit(w, f):
    # ---------------- data ----------------
    x = np.asarray(f["x"], dtype=np.int64)      # [N,64]
    y = np.asarray(f["y"], dtype=np.int64)      # [N]
    N = x.shape[0]
    L = x.shape[1]
    D = 32

    # ---------------- bag-of-words design matrix ----------------
    V = w["embed.weight"].shape[0]
    rows = np.repeat(np.arange(N), L)
    cols = x.reshape(-1)
    mask = cols > 0
    rows = rows[mask]
    cols = cols[mask]
    cnt = np.zeros((N, V), dtype=np.float64)
    np.add.at(cnt, (rows, cols), 1.0)

    # tf-idf style scaling (sublinear tf, idf damped)
    df = (cnt > 0).sum(axis=0).astype(np.float64)
    idf = np.log((1.0 + N) / (1.0 + df)) + 1.0
    tf = np.log1p(cnt)
    Xb = tf * idf[None, :]

    # ---------------- ridge on token features -> token sentiment scores ----------------
    t = (y == 1).astype(np.float64)
    tm = t.mean()
    Xc = Xb - Xb.mean(axis=0, keepdims=True)
    tc = t - tm
    A = Xc.T @ Xc
    lam = 10.0
    A[np.diag_indices_from(A)] += lam
    beta = np.linalg.solve(A, Xc.T @ tc)      # [V]
    beta[0] = 0.0                              # pad token carries no signal

    # token-level score = average of its feature-row coefficients
    tok_score = beta.copy()
    # shrink rare tokens toward zero (they are noisy)
    tok_score = tok_score * (df / (df + 5.0))

    # ---------------- construct embeddings: sign of token score ----------------
    # u: zero-mean, unit-norm direction in R^32 (so LayerNorm(embed_t) = sign*u)
    u = np.zeros(D, dtype=np.float64)
    u[0] = 1.0
    u[1] = -1.0
    u[2] = 0.5
    u[3] = -0.5
    u = u - u.mean()
    u = u / np.linalg.norm(u)

    emb = np.zeros((V, D), dtype=np.float64)
    sgn = np.sign(tok_score)
    sgn[0] = 0.0
    emb += sgn[:, None] * u[None, :]

    # ---------------- identity blocks, zero positional ----------------
    zero_blocks = {}
    for b in ("block0", "block1"):
        for suf in ("qkv.weight", "qkv.bias", "proj.weight", "proj.bias",
                    "ff1.weight", "ff1.bias", "ff2.weight", "ff2.bias"):
            zero_blocks[b + "." + suf] = np.zeros_like(w[b + "." + suf])
    for nm in ("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"):
        zero_blocks["block0." + nm] = np.ones_like(w["block0." + nm]) if nm.endswith("weight") \
            else np.zeros_like(w["block0." + nm])
        zero_blocks["block1." + nm] = np.ones_like(w["block1." + nm]) if nm.endswith("weight") \
            else np.zeros_like(w["block1." + nm])

    # ---------------- recompute pooled features with the new weights ----------------
    def forward_pooled(embW, posW, blocks):
        h = embW[x]                                   # [N,L,D]
        h = h + posW[None, :, :]
        for b in ("block0", "block1"):
            # qkv=0 -> attention output 0 ; proj=0 ; ff=0  => block is identity
            pass
        # final LayerNorm (weight 1, bias 0)
        mu = h.mean(axis=-1, keepdims=True)
        v = h - mu
        sd = np.sqrt((v * v).mean(axis=-1, keepdims=True) + 1e-5)
        hn = v / sd
        valid = (x != 0).astype(np.float64)[:, :, None]
        cntv = valid.sum(axis=1)
        pooled = (hn * valid).sum(axis=1) / np.maximum(cntv, 1.0)
        return pooled

    posW = np.zeros_like(w["pos.weight"], dtype=np.float64)
    pooled = forward_pooled(emb, posW, zero_blocks)

    # ---------------- closed-form head: ridge on pooled features ----------------
    P = pooled - pooled.mean(axis=0, keepdims=True)
    G = P.T @ P
    lam_h = 1.0
    G[np.diag_indices_from(G)] += lam_h
    wh = np.linalg.solve(G, P.T @ tc)               # [D]
    bh = tm - pooled.mean(axis=0) @ wh               # intercept for centered target

    # ---------------- assemble weights ----------------
    out = {}
    for k, v in w.items():
        out[k] = np.array(v, dtype=np.float32, copy=True)

    out["embed.weight"] = emb.astype(np.float32)
    out["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)
    for k, v in zero_blocks.items():
        out[k] = v.astype(np.float32)
    out["norm.weight"] = np.ones_like(w["norm.weight"], dtype=np.float32)
    out["norm.bias"] = np.zeros_like(w["norm.bias"], dtype=np.float32)
    out["head.weight"] = wh.astype(np.float32)[None, :]
    out["head.bias"] = np.array([bh, bh], dtype=np.float32)

    # ---------------- diagnostics ----------------
    logits_new = pooled @ wh + bh
    pred = (logits_new > tm).astype(np.int64)
    acc = float((pred == y).mean())
    p = 1.0 / (1.0 + np.exp(-(logits_new - tm)))
    ce = float(-np.mean(y * np.log(np.clip(p, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1))))

    # simple 5-fold CV of the same closed-form pipeline (no training loops)
    rng = np.random.RandomState(0)
    perm = rng.permutation(N)
    folds = np.array_split(perm, 5)
    cv_correct = 0
    for k in range(5):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(5) if j != k])
        Xtr, ttr = Xb[tr], t[tr]
        mtr = ttr.mean()
        Xtrc = Xtr - Xtr.mean(axis=0, keepdims=True)
        A2 = Xtrc.T @ Xtrc
        A2[np.diag_indices_from(A2)] += lam
        b2 = np.linalg.solve(A2, Xtrc.T @ (ttr - mtr))
        b2[0] = 0.0
        sc = b2 * (df / (df + 5.0))
        sg = np.sign(sc); sg[0] = 0.0
        E = sg[:, None] * u[None, :]
        hh = E[x[tr]]
        mu = hh.mean(axis=-1, keepdims=True); vv = hh - mu
        sd = np.sqrt((vv * vv).mean(axis=-1, keepdims=True) + 1e-5)
        hn = vv / sd
        valid = (x[tr] != 0).astype(np.float64)[:, :, None]
        pool_tr = (hn * valid).sum(axis=1) / np.maximum(valid.sum(axis=1), 1.0)
        hh2 = E[x[te]]
        mu2 = hh2.mean(axis=-1, keepdims=True); vv2 = hh2 - mu2
        sd2 = np.sqrt((vv2 * vv2).mean(axis=-1, keepdims=True) + 1e-5)
        hn2 = vv2 / sd2
        valid2 = (x[te] != 0).astype(np.float64)[:, :, None]
        pool_te = (hn2 * valid2).sum(axis=1) / np.maximum(valid2.sum(axis=1), 1.0)
        Pc = pool_tr - pool_tr.mean(axis=0, keepdims=True)
        G2 = Pc.T @ Pc
        G2[np.diag_indices_from(G2)] += lam_h
        w2 = np.linalg.solve(G2, Pc.T @ (ttr - mtr))
        b2h = mtr - pool_tr.mean(axis=0) @ w2
        lg = pool_te @ w2 + b2h
        cv_correct += int(((lg > mtr).astype(np.int64) == y[te]).sum())

    diagnostics = {
        "hypothesis": "identity blocks + sign(token ridge score) embeddings + closed-form ridge head",
        "train_accuracy_new": round(acc, 4),
        "train_cross_entropy_new": round(ce, 4),
        "cv5_accuracy_new": round(cv_correct / float(N), 4),
        "old_accuracy": 0.484,
        "old_cross_entropy": 0.7149,
        "n_tokens_with_signal": int((np.abs(tok_score) > 1e-9).sum()),
        "frac_pos_sign_tokens": round(float((sgn > 0).mean()), 4),
        "frac_neg_sign_tokens": round(float((sgn < 0).mean()), 4),
        "pooled_std": round(float(pooled.std()), 4),
        "head_weight_norm": round(float(np.linalg.norm(wh)), 4),
        "embed_absmax": round(float(np.abs(emb).max()), 4),
    }
    return {"weights": out, "diagnostics": diagnostics}
