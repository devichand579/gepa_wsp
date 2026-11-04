import numpy as np

def edit(w, f):
    # ------------------------------------------------------------------
    # 0. data
    # ------------------------------------------------------------------
    x = np.asarray(f["x"], dtype=np.int64)          # [N,64] token ids
    y = np.asarray(f["y"], dtype=np.int64)          # [N] labels
    N, L = x.shape
    V, D = 2000, 32
    mask = (x != 0).astype(np.float64)              # [N,64]
    lens = mask.sum(1)                              # [N]

    # ------------------------------------------------------------------
    # 1. reproduce the current forward pass (verifies my reading of the
    #    architecture; also gives me the exact current feature geometry)
    # ------------------------------------------------------------------
    E = np.asarray(w["embed.weight"], dtype=np.float64)   # [V,D]

    def layernorm(h, g, b):
        m = h.mean(-1, keepdims=True)
        v = h.var(-1, keepdims=True)
        return (h - m) / np.sqrt(v + 1e-5) * g + b

    def forward(emb, head_w, head_b, pos_w=None):
        h = emb[x]                                     # [N,L,D]
        if pos_w is not None:
            h = h + pos_w[None, :, :]
        for blk in ("block0", "block1"):
            z = layernorm(h, w[blk + ".norm1.weight"], w[blk + ".norm1.bias"])
            qkv = z @ w[blk + ".qkv.weight"].T + w[blk + ".qkv.bias"]
            q, k, v = np.split(qkv, 3, axis=-1)
            nh = 2
            out = np.zeros_like(q)
            for hh in range(nh):
                sl = slice(hh * 16, (hh + 1) * 16)
                sc = q[:, :, sl] @ k[:, :, sl].transpose(0, 2, 1) / 4.0
                sc = np.where(mask[:, None, :] > 0, sc, -1e30)
                p = np.exp(sc - sc.max(-1, keepdims=True))
                p = p * mask[:, None, :]
                p = p / p.sum(-1, keepdims=True)
                out[:, :, sl] = p @ v[:, :, sl]
            h = h + out @ w[blk + ".proj.weight"].T + w[blk + ".proj.bias"]
            z2 = layernorm(h, w[blk + ".norm2.weight"], w[blk + ".norm2.bias"])
            f1 = np.maximum(z2 @ w[blk + ".ff1.weight"].T + w[blk + ".ff1.bias"], 0.0)
            h = h + f1 @ w[blk + ".ff2.weight"].T + w[blk + ".ff2.bias"]
        h = layernorm(h, w["norm.weight"], w["norm.bias"])
        pooled = (h * mask[:, :, None]).sum(1) / lens[:, None]
        return pooled, pooled @ head_w.T + head_b

    cur_feat = np.asarray(f["features"], dtype=np.float64)
    pooled0, logits0 = forward(E, w["head.weight"], w["head.bias"])
    diag = {
        "repro_feat_max_abs_err": float(np.abs(pooled0 - cur_feat).max()),
        "repro_logit_max_abs_err": float(np.abs(logits0 - np.asarray(f["logits"], np.float64)).max()),
        "cur_acc": float(((logits0.argmax(1) == y).mean())),
        "cur_feat0_std": float(cur_feat[:, 0].std()),
        "cur_feat01_corr": float(np.corrcoef(cur_feat[:, 0], cur_feat[:, 1])[0, 1]),
    }

    # ------------------------------------------------------------------
    # 2. closed-form per-token weights (discriminative Naive-Bayes style)
    #
    #    score(doc) = sum_t  s_t  with  s_t = w[token]  (signed counts)
    #    We want a linear separator on the token-count vector, so we fit
    #    w by ridge least squares on the *label-signed* token counts:
    #        target_i = (2*y_i - 1) * len_i        (so that
    #        sum_t w[token] ~ target_i  =>  sign of score = label)
    #    Counts are built with a closed-form sparse accumulation.
    # ------------------------------------------------------------------
    ysign = 2.0 * y - 1.0
    # token counts per document (dense [N,V] is 4000x2000 -> fine)
    C = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), mask.astype(np.int64).ravel())
    cols = x.ravel()
    np.add.at(C, (rows, cols), 1.0)
    C[:, 0] = 0.0                                     # drop pad

    # ridge normal equations, solved in the (much smaller) dual when possible
    lam = 1.0
    A = C * ysign[:, None]                             # signed counts
    target = lens.copy()                               # want sum_t w = +len if pos
    # normal equations: (A^T A + lam I) w = A^T target
    G = A.T @ A
    b = A.T @ target
    G[np.diag_indices_from(G)] += lam
    try:
        wtok = np.linalg.solve(G, b)
    except np.linalg.LinAlgError:
        wtok = np.linalg.lstsq(G, b, rcond=None)[0]
    wtok = np.nan_to_num(wtok, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------
    # 3. design the new embedding: 8 channels of 4 dims each.
    #    channel j: dims 4j..4j+3 = [u, -u, w, -w], u^2+w^2 = 16
    #    => LayerNorm output = [u, -u, w, -w] exactly (std = 1)
    #    u = clipped token score (channel 0), w = +sqrt(16-u^2) (constant-ish)
    #    channels 1..7 carry "context" scores (token weight * neighbour weight)
    #    so that the head can combine several bag-of-words views.
    # ------------------------------------------------------------------
    AMAX = 3.2                                          # |u| bound (keeps w real)
    u0 = np.clip(wtok, -AMAX, AMAX)

    # context channels: u_j = clip(wtok[t] * wtok[t-1], bound) (left neighbour)
    # and a "position-free" second view: u_j = clip(wtok[t]*wtok[t+1])
    def neighbour_scores(u):
        left = np.zeros_like(u)
        left[:, 1:] = u[:, :-1]
        right = np.zeros_like(u)
        right[:, :-1] = u[:, 1:]
        return left, right

    left, right = neighbour_scores(u0)
    chan_u = [u0]
    for v in (left, right):
        chan_u.append(np.clip(v, -AMAX, AMAX))
    # remaining channels: smoothed (local mean of token scores)
    sm = np.zeros_like(u0)
    sm[:, 1:] = 0.5 * (u0[:, :-1] + u0[:, 1:])
    chan_u.append(np.clip(sm, -AMAX, AMAX))
    # pad channels with zeros (they will just be constant 0)
    while len(chan_u) < 8:
        chan_u.append(np.zeros_like(u0))

    Enew = np.zeros((V, D), dtype=np.float64)
    for j, u in enumerate(chan_u):
        wj = np.sqrt(np.maximum(16.0 - u * u, 0.0))
        Enew[:, 4 * j + 0] = u
        Enew[:, 4 * j + 1] = -u
        Enew[:, 4 * j + 2] = wj
        Enew[:, 4 * j + 3] = -wj

    # ------------------------------------------------------------------
    # 4. analytic head: pooled0 = mean_t u0 ; pooled1 = -pooled0
    #    pooled2 = mean_t w0  (nuisance, constant-ish) ; pooled3 = -pooled2
    #    channels 1..7 give extra views; fit head by ridge least squares
    #    on the *new* features with the true labels (closed form).
    # ------------------------------------------------------------------
    def forward_new(emb):
        h = emb[x]
        for blk in ("block0", "block1"):
            z = layernorm(h, w[blk + ".norm1.weight"], w[blk + ".norm1.bias"])
            qkv = z @ w[blk + ".qkv.weight"].T + w[blk + ".qkv.bias"]
            q, k, v = np.split(qkv, 3, axis=-1)
            out = np.zeros_like(q)
            for hh in range(2):
                sl = slice(hh * 16, (hh + 1) * 16)
                sc = q[:, :, sl] @ k[:, :, sl].transpose(0, 2, 1) / 4.0
                sc = np.where(mask[:, None, :] > 0, sc, -1e30)
                p = np.exp(sc - sc.max(-1, keepdims=True))
                p = p * mask[:, None, :]
                p = p / p.sum(-1, keepdims=True)
                out[:, :, sl] = p @ v[:, :, sl]
            h = h + out @ w[blk + ".proj.weight"].T + w[blk + ".proj.bias"]
            z2 = layernorm(h, w[blk + ".norm2.weight"], w[blk + ".norm2.bias"])
            f1 = np.maximum(z2 @ w[blk + ".ff1.weight"].T + w[blk + ".ff1.bias"], 0.0)
            h = h + f1 @ w[blk + ".ff2.weight"].T + w[blk + ".ff2.bias"]
        h = layernorm(h, w["norm.weight"], w["norm.bias"])
        return (h * mask[:, :, None]).sum(1) / lens[:, None]

    Fnew = forward_new(Enew)                            # [N,32]
    # design matrix for the head: features + intercept
    P = np.concatenate([Fnew, np.ones((N, 1))], axis=1)
    # ridge least squares to the +/-1 targets (linear classifier)
    lamh = 1e-3
    H = P.T @ P
    H[np.diag_indices_from(H)] += lamh
    rhs = P.T @ ysign
    beta = np.linalg.solve(H, rhs)
    # head.weight rows: class0 = -beta, class1 = +beta (logit = beta . p + b)
    head_w = np.zeros((2, D), dtype=np.float64)
    head_w[1] = beta[:D]
    head_w[0] = -beta[:D]
    head_b = np.zeros(2, dtype=np.float64)
    head_b[1] = beta[D]
    head_b[0] = -beta[D]

    # ------------------------------------------------------------------
    # 5. sanity: accuracy / CE of the new construction on the pool
    # ------------------------------------------------------------------
    lg = Fnew @ head_w.T + head_b
    p = np.exp(lg - lg.max(1, keepdims=True))
    p = p / p.sum(1, keepdims=True)
    new_acc = float((lg.argmax(1) == y).mean())
    new_ce = float(-np.log(np.clip(p[np.arange(N), y], 1e-12, 1.0)).mean())
    diag.update({
        "new_acc": new_acc,
        "new_ce": new_ce,
        "wtok_absmax": float(np.abs(wtok).max()),
        "wtok_nnz": int((np.abs(wtok) > 1e-8).sum()),
        "head_beta": [float(v) for v in beta[:8]],
        "new_feat0_std": float(Fnew[:, 0].std()),
        "new_feat0_cls_mean": [float(Fnew[y == c, 0].mean()) for c in (0, 1)],
    })

    # ------------------------------------------------------------------
    # 6. write back (float32, finite, |.| <= 1000)
    # ------------------------------------------------------------------
    def put(key, arr):
        a = np.asarray(arr, dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        a = np.clip(a, -1000.0, 1000.0)
        w[key] = a.reshape(np.shape(w[key])).astype(np.float32)

    put("embed.weight", Enew)
    put("head.weight", head_w)
    put("head.bias", head_b)
    # keep the (zero) block weights untouched; pos stays zero (masked mean
    # makes position irrelevant for a bag-of-words model)
    for k in w:
        if k in ("embed.weight", "head.weight", "head.bias"):
            continue
        w[k] = np.asarray(w[k], dtype=np.float32)

    return {"weights": w, "diagnostics": diag}
