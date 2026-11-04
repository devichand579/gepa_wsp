import numpy as np

def edit(w, f):
    # ---- read feedback ----
    X = f["x"]                       # [N,64] int64 token ids
    y = f["y"].astype(np.float64)    # [N]
    N, L = X.shape
    V = w["embed.weight"].shape[0]   # 2000

    # ---- build length-normalised unigram count matrix ----
    # counts of each vocab token per example, normalised by #non-pad tokens
    mask = (X != 0).astype(np.float64)          # [N,64]
    counts = np.zeros((N, V), dtype=np.float64)
    for i in range(N):
        ids = X[i][mask[i] > 0]
        if ids.size:
            np.add.at(counts[i], ids, 1.0)
    denom = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
    F = counts / denom                          # [N,V] normalised unigram counts

    # ---- closed-form ridge on log-odds target, CV over alpha ----
    # target: +1 for positive, -1 for negative
    t = 2.0 * y - 1.0
    # centre target
    t = t - t.mean()

    # K-fold CV
    rng = np.random.RandomState(0)
    idx = rng.permutation(N)
    K = 5
    folds = np.array_split(idx, K)

    alphas = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    best_alpha = alphas[0]
    best_cv = -1.0
    for a in alphas:
        accs = []
        for k in range(K):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(K) if j != k])
            Ftr, ttr = F[tr], t[tr]
            Fte = F[te]
            # ridge: w = (F^T F + a I)^-1 F^T t  (no intercept, target centred)
            A = Ftr.T @ Ftr + a * np.eye(V)
            b = Ftr.T @ ttr
            try:
                coef = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                coef = np.linalg.lstsq(A, b, rcond=None)[0]
            pred = Fte @ coef
            accs.append(np.mean((pred > 0) == (y[te] > 0.5)))
        cv = float(np.mean(accs))
        if cv > best_cv:
            best_cv = cv
            best_alpha = a

    # ---- final fit on all data ----
    A = F.T @ F + best_alpha * np.eye(V)
    b = F.T @ t
    try:
        coef = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(A, b, rcond=None)[0]

    # per-token score s(v) = coef[v]; intercept b0 from mean
    s = coef.astype(np.float64)
    # scale so that the head can use a fixed multiplier
    scale = 1.0
    s = s * scale

    # ---- encode s(v) into embed[:,0], satisfy zero-mean/unit-variance ----
    # For each token v: embed[v] = [s_v, r_v, c_v, c_v, ..., c_v]
    # We need mean over dims = 0 and variance over dims = 1.
    # Choose dims 2..31 all equal to c_v, and dim 1 = r_v.
    # mean = (s_v + r_v + 30 c_v)/32 = 0  ->  r_v + 30 c_v = -s_v
    # var  = (s_v^2 + r_v^2 + 30 c_v^2)/32 = 1  ->  s_v^2 + r_v^2 + 30 c_v^2 = 32
    # Solve: let r_v = -s_v - 30 c_v. Substitute:
    # s_v^2 + (s_v + 30 c_v)^2 + 30 c_v^2 = 32
    # s_v^2 + s_v^2 + 60 s_v c_v + 900 c_v^2 + 30 c_v^2 = 32
    # 2 s_v^2 + 60 s_v c_v + 930 c_v^2 = 32
    # 930 c_v^2 + 60 s_v c_v + (2 s_v^2 - 32) = 0
    # c_v = [-60 s_v ± sqrt(3600 s_v^2 - 4*930*(2 s_v^2 - 32))] / (2*930)
    #     = [-60 s_v ± sqrt(3600 s_v^2 - 7440 s_v^2 + 119040)] / 1860
    #     = [-60 s_v ± sqrt(119040 - 3840 s_v^2)] / 1860
    # Need 119040 - 3840 s_v^2 >= 0 -> s_v^2 <= 31.0 -> |s_v| <= 5.568
    # Clip s_v to [-5.5, 5.5] to stay safe.
    s = np.clip(s, -5.5, 5.5)

    disc = 119040.0 - 3840.0 * s * s
    disc = np.maximum(disc, 0.0)
    sq = np.sqrt(disc)
    c = (-60.0 * s + sq) / 1860.0          # choose + branch
    r = -s - 30.0 * c

    E = np.zeros((V, 32), dtype=np.float64)
    E[:, 0] = s
    E[:, 1] = r
    E[:, 2:] = c[:, None]

    # ---- head: logit1 - logit0 = s(v) (times scale), intercept from bias ----
    # We want logit1 - logit0 = s(v) * scale.  With head reading dim 0 only:
    # head[0,0] = -0.5, head[1,0] = +0.5  -> diff = 1.0 * pooled[0]
    # pooled[0] = mean_t s(v_t) (since LN(embed[v])=embed[v]).
    # So diff = mean_t s(v_t).  We want diff = sum_t s(v_t)/n = mean_t s(v_t).
    # Good.  But the ridge target was t = ±1 (not log-odds).  That's fine for
    # argmax; we just need the sign of the mean score to match the label.
    # Add intercept via head.bias: logit1 - logit0 += (bias1 - bias0).
    # We set bias so that the decision threshold is 0 (target centred already).
    head = np.zeros((2, 32), dtype=np.float64)
    head[0, 0] = -0.5
    head[1, 0] = 0.5
    bias = np.zeros(2, dtype=np.float64)

    # ---- write weights ----
    w["embed.weight"] = E.astype(np.float32)
    w["head.weight"] = head.astype(np.float32)
    w["head.bias"] = bias.astype(np.float32)

    # zero out blocks / pos to keep the degenerate-but-clean structure
    for k in ["pos.weight"]:
        w[k] = np.zeros_like(w[k], dtype=np.float32)
    for blk in ["block0", "block1"]:
        for sub in ["qkv.weight", "qkv.bias", "proj.weight", "proj.bias",
                    "ff1.weight", "ff1.bias", "ff2.weight", "ff2.bias"]:
            key = f"{blk}.{sub}"
            if key in w:
                w[key] = np.zeros_like(w[key], dtype=np.float32)
    for key in ["block0.norm1.weight", "block0.norm1.bias",
                "block0.norm2.weight", "block0.norm2.bias",
                "block1.norm1.weight", "block1.norm1.bias",
                "block1.norm2.weight", "block1.norm2.bias",
                "norm.weight", "norm.bias"]:
        if key in w:
            if key.endswith("weight"):
                w[key] = np.ones_like(w[key], dtype=np.float32)
            else:
                w[key] = np.zeros_like(w[key], dtype=np.float32)

    # ---- diagnostics ----
    diag = {
        "best_alpha": float(best_alpha),
        "cv_acc": float(best_cv),
        "s_absmax": float(np.max(np.abs(s))),
        "s_std": float(np.std(s)),
        "embed_nonzero": int(np.count_nonzero(E)),
        "head_norm": float(np.linalg.norm(head)),
    }
    return {"weights": w, "diagnostics": diag}
