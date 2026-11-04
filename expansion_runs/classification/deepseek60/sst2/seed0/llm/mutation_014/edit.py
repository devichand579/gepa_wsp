import numpy as np
import math

def edit(w, f):
    # ---- 1. Pull the feedback pool -------------------------------------
    X = np.asarray(f["x"], dtype=np.int64)          # [N,64]
    y = np.asarray(f["y"], dtype=np.int64).astype(np.float64)  # [N]
    N, L = X.shape
    V = w["embed.weight"].shape[0]                  # 2000

    # ---- 2. Build length-normalized unigram count features -------------
    # counts[n, v] = (# occurrences of v in example n) / (# non-pad tokens)
    mask = (X != 0).astype(np.float64)              # [N,64]
    lens = mask.sum(axis=1)                         # [N]
    lens = np.maximum(lens, 1.0)
    counts = np.zeros((N, V), dtype=np.float64)
    for n in range(N):
        ids = X[n]
        ids = ids[ids != 0]
        if ids.size:
            np.add.at(counts[n], ids, 1.0)
    counts /= lens[:, None]                          # normalized counts

    # ---- 3. Closed-form ridge fit on the log-odds target ---------------
    # target t = +1 for positive, -1 for negative
    t = 2.0 * y - 1.0
    # center features (intercept handled separately)
    mu = counts.mean(axis=0, keepdims=True)
    C = counts - mu
    tmu = t.mean()
    tc = t - tmu

    def ridge_fit(alpha):
        # solve (C^T C + alpha I) beta = C^T tc
        G = C.T @ C
        G[np.diag_indices_from(G)] += alpha
        rhs = C.T @ tc
        beta = np.linalg.solve(G, rhs)
        b0 = tmu - mu.ravel() @ beta
        return beta, b0

    # ---- 4. Choose alpha by K-fold CV ----------------------------------
    rng = np.random.RandomState(0)
    idx = rng.permutation(N)
    K = 5
    folds = np.array_split(idx, K)
    alphas = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    best_alpha, best_acc = alphas[0], -1.0
    for a in alphas:
        accs = []
        for k in range(K):
            va = folds[k]
            tr = np.concatenate([folds[j] for j in range(K) if j != k])
            Ctr, ttr = C[tr], tc[tr]
            G = Ctr.T @ Ctr
            G[np.diag_indices_from(G)] += a
            beta = np.linalg.solve(G, Ctr.T @ ttr)
            b0 = t[tr].mean() - counts[tr].mean(axis=0) @ beta
            pred = np.where(counts[va] @ beta + b0 >= 0.0, 1.0, 0.0)
            accs.append((pred == y[va]).mean())
        acc = float(np.mean(accs))
        if acc > best_acc:
            best_acc, best_alpha = acc, a

    beta, b0 = ridge_fit(best_alpha)

    # ---- 5. Encode per-token scores exactly into embed -----------------
    # We want logit1 - logit0 = 4 * mean_t s(v_t) + head_bias_diff
    # so per-token score s(v) = beta[v] (already the linear coefficient),
    # and the intercept goes into head.bias.
    s = beta.astype(np.float64)                      # [V]

    # Build embed rows: [s, -s, a, -a, 0...] with mean 0 and var 1.
    # sum of squares = 2 s^2 + 2 a^2 = 32  =>  a = sqrt(16 - s^2)
    # If |s| > 4, we must rescale to keep var = 1 and |values| <= 1000.
    smax = np.max(np.abs(s)) if s.size else 0.0
    scale = 1.0
    if smax > 3.9:
        scale = 3.9 / smax
    s_enc = s * scale

    E = np.zeros((V, 32), dtype=np.float64)
    a2 = np.maximum(16.0 - s_enc ** 2, 1e-6)
    a = np.sqrt(a2)
    E[:, 0] = s_enc
    E[:, 1] = -s_enc
    E[:, 2] = a
    E[:, 3] = -a
    # rows for pad (id 0) should be zero-mean/unit-var too; keep as is.

    w["embed.weight"] = E.astype(np.float32)

    # ---- 6. Set head so that logit1 - logit0 = 4 * pooled[0] -----------
    # pooled[0] = mean_t LN(embed[v_t])[0] = mean_t s_enc(v_t)
    # We want the effective coefficient to be 4 * scale (to undo the shrink).
    coef = 4.0 * scale
    head = np.zeros((2, 32), dtype=np.float64)
    head[0, 0] = 0.0
    head[1, 0] = coef
    w["head.weight"] = head.astype(np.float32)

    # intercept: b0 is on the t = ±1 scale; logit diff = 2*t_hat.
    # logit1 - logit0 = coef * mean_s + (head_bias1 - head_bias0)
    # We want this = 2 * (beta·counts + b0) = 2*scale^{-1} * coef/4 * ... 
    # Simpler: set head.bias so that logit diff matches 2*(score + b0).
    # score = coef * mean_s = 4*scale * mean(s_enc/scale) = 4 * mean(s).
    # So logit diff = 4 * mean(s) + (hb1 - hb0). Want = 2*(mean(s)*? )
    # We fit beta on t=±1, so decision is beta·counts + b0 >= 0.
    # logit diff should be proportional: use 2*(beta·counts + b0) = 2*mean(s)+2*b0
    # But our coef gives 4*mean(s). Rescale: choose coef=2 instead of 4.
    coef2 = 2.0
    head[1, 0] = coef2
    w["head.weight"] = head.astype(np.float32)

    hb = np.zeros(2, dtype=np.float64)
    hb[0] = -b0
    hb[1] = b0
    w["head.bias"] = hb.astype(np.float32)

    # ---- 7. Zero out everything else (keep identity blocks) ------------
    zero_names = [
        "pos.weight",
        "block0.qkv.weight", "block0.qkv.bias",
        "block0.proj.weight", "block0.proj.bias",
        "block0.ff1.weight", "block0.ff1.bias",
        "block0.ff2.weight", "block0.ff2.bias",
        "block1.qkv.weight", "block1.qkv.bias",
        "block1.proj.weight", "block1.proj.bias",
        "block1.ff1.weight", "block1.ff1.bias",
        "block1.ff2.weight", "block1.ff2.bias",
    ]
    for k in zero_names:
        if k in w:
            w[k] = np.zeros_like(w[k], dtype=np.float32)

    for k in ["block0.norm1.weight", "block0.norm2.weight",
              "block1.norm1.weight", "block1.norm2.weight",
              "norm.weight"]:
        if k in w:
            w[k] = np.ones_like(w[k], dtype=np.float32)
    for k in ["block0.norm1.bias", "block0.norm2.bias",
              "block1.norm1.bias", "block1.norm2.bias",
              "norm.bias"]:
        if k in w:
            w[k] = np.zeros_like(w[k], dtype=np.float32)

    # ---- 8. Diagnostics ------------------------------------------------
    diag = {
        "best_alpha": float(best_alpha),
        "cv_acc": float(best_acc),
        "s_absmax": float(np.max(np.abs(s))) if s.size else 0.0,
        "s_std": float(np.std(s)) if s.size else 0.0,
        "scale": float(scale),
        "coef": float(coef2),
        "b0": float(b0),
        "embed_nonzero": int(np.count_nonzero(w["embed.weight"])),
    }
    return {"weights": w, "diagnostics": diag}
