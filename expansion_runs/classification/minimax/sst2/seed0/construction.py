def edit(w, f):
    import numpy as np

    # ---------------- data ----------------
    X = np.asarray(f["x"], dtype=np.int64)          # [N,64]
    Y = np.asarray(f["y"], dtype=np.int64)          # [N]
    N, L = X.shape
    V = w["embed.weight"].shape[0]                 # 2000
    D = w["embed.weight"].shape[1]                 # 32

    mask = (X != 0)
    Llen = np.maximum(mask.sum(1), 1).astype(np.float64)

    # counts (dense V columns is fine: 4000x2000 float64 = 64MB)
    B = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), L)
    cols = X.reshape(-1)
    nz = cols != 0
    np.add.at(B, (rows[nz], cols[nz]), 1.0)
    Bn = B / Llen[:, None]                        # length-normalized bag of words

    # ---------------- closed-form ridge on b/L ----------------
    t = np.where(Y == 1, 1.0, -1.0)
    G = Bn.T @ Bn
    bvec = Bn.T @ t
    lam = 3.0
    A = np.linalg.solve(G + lam * np.eye(V), bvec)
    # in-sample sanity
    s = Bn @ A
    acc = float(((s > 0).astype(np.int64) == Y).mean())

    # ---------------- realize per-token scores in the architecture ----------------
    # pooled = (1/L) * sum_t count_t * u_t   with u_t zero-mean / unit-variance
    # head delta = w[1]-w[0] must satisfy u_t . delta = A[t]
    # construction: delta = e0 * d ; u_t = A[t]/d * e0 + orthogonal part fixing mean & var
    d = 1.0
    delta = np.zeros(D); delta[0] = d
    # orthonormal basis of delta-complement
    Q = np.linalg.qr(np.eye(D) - np.outer(delta, delta))[0]   # [D,D], first col ~ delta
    # choose u_t = (A[t]/d)*delta + z_t, z_t in complement with:
    #   sum_j u_t[j] = 0  -> sum_j z_t[j] = -A[t]/d * sum_j delta[j] = -A[t]/d
    #   var_j u_t = 1     -> mean_j u_t^2 = 1
    # use two complement directions: one to fix the mean, one to fix the variance
    c1 = Q[:, 1]; c2 = Q[:, 2]
    s1 = c1.sum(); s2 = c2.sum()
    # z_t = alpha_t*c1 + beta_t*c2 + gamma_t*c3
    c3 = Q[:, 3]
    # mean constraint: alpha*s1 + beta*s2 + gamma*s3 = -A[t]/d
    # variance: mean_j u^2 = 1 -> (A[t]/d)^2 * mean(delta^2) + alpha^2 + beta^2 + gamma^2 = 1
    #   (delta has norm 1, mean(delta^2)=1/D)
    # choose gamma = 0 for simplicity, solve alpha,beta from mean + variance
    # mean(delta)=0 -> contributes nothing to sum; good.
    # mean_j u_t^2 = (A[t]/d)^2 / D + alpha^2 + beta^2 = 1
    # alpha*s1 + beta*s2 = -A[t]/d
    # alpha^2 + beta^2 = 1 - (A[t]/d)^2/D
    # solve: alpha = (-A[t]/d - beta*s2)/s1 ; substitute -> quadratic in beta
    Ad = A / d
    rhs = 1.0 - (Ad ** 2) / D
    # quadratic: beta^2 (1 + s2^2/s1^2) + beta*(2*Ad*s2/s1^2) + (Ad^2/s1^2 - rhs) = 0
    aa = 1.0 + (s2 / s1) ** 2
    bb = 2.0 * Ad * s2 / (s1 ** 2)
    cc = (Ad ** 2) / (s1 ** 2) - rhs
    disc = np.maximum(bb ** 2 - 4 * aa * cc, 0.0)
    beta = (-bb + np.sqrt(disc)) / (2 * aa)
    alpha = (-Ad - beta * s2) / s1
    U = np.zeros((V, D))
    U[:, 0] = Ad
    U += alpha[:, None] * c1[None, :]
    U += beta[:, None] * c2[None, :]
    # enforce exact zero-mean and unit-variance (numerical cleanup)
    U -= U.mean(axis=1, keepdims=True)
    U /= np.maximum(U.std(axis=1, keepdims=True), 1e-8)

    # ---------------- write weights ----------------
    E = np.zeros((V, D), dtype=np.float32)
    E[:] = U.astype(np.float32)
    E[0] = 0.0                                   # pad row: masked out anyway
    w["embed.weight"] = E

    # positional embedding -> 0 (no positional info; keeps LN(e_t) exact)
    w["pos.weight"] = np.zeros_like(w["pos.weight"])

    # blocks -> exact identity (zeroed projections, LayerNorms are pure pre-ops)
    for bname in ("block0", "block1"):
        for key in (bname + ".qkv.weight", bname + ".qkv.bias",
                    bname + ".proj.weight", bname + ".proj.bias",
                    bname + ".ff1.weight", bname + ".ff1.bias",
                    bname + ".ff2.weight", bname + ".ff2.bias"):
            w[key] = np.zeros_like(w[key])
        for key in (bname + ".norm1.weight", bname + ".norm2.weight"):
            w[key] = np.ones_like(w[key])
        for key in (bname + ".norm1.bias", bname + ".norm2.bias"):
            w[key] = np.zeros_like(w[key])

    w["norm.weight"] = np.ones_like(w["norm.weight"])
    w["norm.bias"] = np.zeros_like(w["norm.bias"])

    # ---------------- head: closed-form ridge on the realized pooled features ----
    # pooled = (1/L) * sum_t count_t * u_t  ->  pooled = Bn @ U
    pooled = Bn @ U                                  # [N,D]
    Gp = pooled.T @ pooled
    bp = pooled.T @ t
    lamh = 1e-6
    Wh = np.linalg.solve(Gp + lamh * np.eye(D), bp)   # [D] direction of (w1-w0)
    # bias: choose so that mean logit-diff = 0 on pool (balanced classes)
    sd = pooled @ Wh
    bias = -float(sd.mean())
    # realize delta = Wh (must equal u_t . delta = A[t]); verify and correct scale
    # u_t . Wh  ->  per-token realized scores; refit A to match exactly:
    A_real = U @ Wh
    # rebuild embeddings so that u_t . Wh == A_refit[t] exactly:
    #   solve u_t in the plane spanned by Wh and complement with mean/var constraints
    Whn = Wh / max(np.linalg.norm(Wh), 1e-8)
    dvec = Whn
    Q2 = np.linalg.qr(np.eye(D) - np.outer(dvec, dvec))[0]
    e1 = Q2[:, 1]; e2 = Q2[:, 2]
    s1 = e1.sum(); s2 = e2.sum()
    # want u_t . dvec = A_real[t] (call it ar)
    ar = A_real
    # u_t = ar*dvec + alpha*e1 + beta*e2 ; mean: alpha*s1+beta*s2 = -ar*sum(dvec)
    m0 = float(dvec.sum())
    aa = 1.0 + (s2 / s1) ** 2
    bb = 2.0 * (ar * m0) * s2 / (s1 ** 2)
    cc = (ar * m0) ** 2 / (s1 ** 2) - (1.0 - (ar ** 2) / D)
    disc = np.maximum(bb ** 2 - 4 * aa * cc, 0.0)
    beta = (-bb + np.sqrt(disc)) / (2 * aa)
    alpha = (-ar * m0 - beta * s2) / s1
    U2 = np.zeros((V, D))
    U2 += ar[:, None] * dvec[None, :]
    U2 += alpha[:, None] * e1[None, :]
    U2 += beta[:, None] * e2[None, :]
    U2 -= U2.mean(axis=1, keepdims=True)
    U2 /= np.maximum(U2.std(axis=1, keepdims=True), 1e-8)
    E2 = U2.astype(np.float32)
    E2[0] = 0.0
    w["embed.weight"] = E2

    # head weights: w[1]-w[0] = Wh ; w[0] = -Wh/2 + c ; w[1] = Wh/2 + c  (bias split)
    hw = np.asarray(w["head.weight"], dtype=np.float64)
    hb = np.asarray(w["head.bias"], dtype=np.float64)
    cvec = 0.5 * (hw[0] + hw[1])
    newhw = np.stack([-0.5 * Wh + cvec, 0.5 * Wh + cvec], axis=0)
    newhb = np.array([0.5 * bias, -0.5 * bias], dtype=np.float64)
    w["head.weight"] = newhw.astype(np.float32)
    w["head.bias"] = newhb.astype(np.float32)

    # ---------------- verify by exact forward simulation ----------------
    def forward_logits(emb, posw, headw, headb, Xs):
        B_, Ls = Xs.shape
        h = emb[Xs] + posw[None, :, :]
        for bname in ("block0", "block1"):
            z = layernorm(h, w[bname + ".norm1.weight"], w[bname + ".norm1.bias"])
            qkv = z @ w[bname + ".qkv.weight"].T + w[bname + ".qkv.bias"]
            q, k, v = np.split(qkv, 3, axis=-1)
            att = np.zeros_like(v)
            for hd in range(2):
                sl = slice(hd * 16, (hd + 1) * 16)
                sc = q[:, :, sl] @ k[:, :, sl].transpose(0, 2, 1) / 4.0
                m = (Xs != 0)[:, None, :]
                sc = np.where(m, sc, -1e9)
                e = np.exp(sc - sc.max(axis=-1, keepdims=True))
                e = np.where(m, e, 0.0)
                e = e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-30)
                att[:, :, sl] = e @ v[:, :, sl]
            h = h + att @ w[bname + ".proj.weight"].T + w[bname + ".proj.bias"]
            z2 = layernorm(h, w[bname + ".norm2.weight"], w[bname + ".norm2.bias"])
            f = np.maximum(z2 @ w[bname + ".ff1.weight"].T + w[bname + ".ff1.bias"], 0.0)
            h = h + f @ w[bname + ".ff2.weight"].T + w[bname + ".ff2.bias"]
        h = layernorm(h, w["norm.weight"], w["norm.bias"])
        m = (Xs != 0)[:, :, None]
        pooled = (h * m).sum(1) / np.maximum(m.sum(1), 1)
        return pooled @ headw.T + headb

    def layernorm(h, g, b):
        mu = h.mean(-1, keepdims=True)
        var = h.var(-1, keepdims=True)
        return (h - mu) / np.sqrt(var + 1e-5) * g + b

    lg = forward_logits(w["embed.weight"], w["pos.weight"],
                        w["head.weight"], w["head.bias"], X)
    pred = lg.argmax(1)
    acc_sim = float((pred == Y).mean())
    ce = float(-np.mean(np.log(np.clip(1.0 / (1.0 + np.exp(-(lg[:, 1] - lg[:, 0]))), 1e-12, 1))))
    margin = float(np.mean(lg[:, 1] - lg[:, 0]))
    pos_rate = float(pred.mean())

    # safety: finite, magnitude bounds
    for k in list(w.keys()):
        arr = np.asarray(w[k])
        if not np.all(np.isfinite(arr)):
            w[k] = np.nan_to_num(arr, nan=0.0, posinf=100.0, neginf=-100.0)
        w[k] = np.clip(np.asarray(w[k], dtype=np.float32), -1000.0, 1000.0)

    return {"weights": w,
            "diagnostics": {
                "ridge_acc_normalized_bow": acc,
                "sim_acc": acc_sim,
                "sim_ce": ce,
                "mean_margin": margin,
                "pred_pos_rate": pos_rate,
                "label_pos_rate": float((Y == 1).mean()),
                "n_tokens_used": int((B.sum(0) > 0).sum()),
                "lam": lam,
                "max_abs_embed": float(np.abs(w["embed.weight"]).max()),
                "max_abs_head": float(np.abs(w["head.weight"]).max()),
                "head_norm": float(np.linalg.norm(w["head.weight"][1] - w["head.weight"][0])),
                "per_token_score_absmax": float(np.abs(A).max()),
                "per_token_score_std": float(A.std()),
                "batch_acc": float((f["logits"].argmax(1) == np.asarray(f["batch.y"])).mean()),
            }}
