import numpy as np

def edit(w, f):
    X = f["x"].astype(np.int64)            # [N,64]
    y = f["y"].astype(np.int64)            # [N]
    N, T = X.shape
    V = w["embed.weight"].shape[0]         # 2000
    D = w["embed.weight"].shape[1]         # 32

    # ---- 1. Build a richer closed-form BoW representation ----
    # token counts (excluding PAD=0) per example
    mask = (X != 0)
    counts = np.zeros((N, V), dtype=np.float64)
    for j in range(V):
        counts[:, j] = ((X == j) & mask).sum(axis=1)
    lengths = mask.sum(axis=1).astype(np.float64)
    lengths[lengths == 0] = 1.0
    # length-normalised BoW features
    B = counts / lengths[:, None]          # [N,V]

    # ---- 2. Closed-form regularised linear discriminant on BoW ----
    # class means
    m0 = B[y == 0].mean(axis=0)
    m1 = B[y == 1].mean(axis=0)
    # pooled within-class covariance (diagonal + shrinkage for stability)
    c0 = B[y == 0] - m0
    c1 = B[y == 1] - m1
    S = (c0.T @ c0 + c1.T @ c1) / max(N - 2, 1)
    lam = 1e-2 * (np.trace(S) / V + 1e-12)
    Sd = np.diag(np.diag(S)) + lam * np.eye(V)
    # LDA direction
    d = np.linalg.solve(Sd, (m1 - m0))     # [V]

    # ---- 3. Encode the BoW direction into embed + head ----
    # We want head_1 - head_0 = w such that (w · mean_t LN(embed[v_t])) ~ mean_t d(v_t).
    # Use embed[v] = d(v) * e0 + C * e1  with e0,e1 zero-mean orthonormal in R^D.
    e0 = np.zeros(D); e0[0] = 1.0
    e1 = np.zeros(D); e1[1] = 1.0
    # make them zero-mean over the D dims
    e0 = e0 - e0.mean(); e1 = e1 - e1.mean()
    e0 /= (np.linalg.norm(e0) + 1e-12)
    e1 = e1 - (e1 @ e0) * e0
    e1 /= (np.linalg.norm(e1) + 1e-12)

    C = 8.0
    # scale d so that LN output magnitude is reasonable
    dscale = np.std(d) + 1e-12
    dd = d / dscale
    E = np.zeros((V, D), dtype=np.float32)
    for v in range(V):
        vec = dd[v] * e0 + C * e1
        # LN will normalise; store raw vector
        E[v] = vec.astype(np.float32)
    w["embed.weight"] = E

    # head: (head_1 - head_0) should be along e0 with appropriate scale.
    # LN(embed[v]) = sqrt(D) * vec/||vec||.  For large C, ~ sqrt(D)*(dd[v]/C) e0 + sqrt(D) e1.
    # So (head_1-head_0)·LN ~ sqrt(D)*(w·e0)*(dd[v]/C).
    # We want this ~ dd[v] (up to a global scale), so w·e0 = C/sqrt(D).
    scale = C / np.sqrt(D)
    w0 = -0.5 * scale * e0
    w1 =  0.5 * scale * e0
    # bias: match class priors / decision threshold
    # compute realised features on the pool to set bias
    LN = np.zeros((N, D), dtype=np.float64)
    for v in range(V):
        rows = (X == v) & mask
        if not rows.any():
            continue
        vec = E[v].astype(np.float64)
        mu = vec.mean()
        sd = vec.std() + 1e-12
        LN[rows] = (vec - mu) / sd
    feat = np.zeros((N, D), dtype=np.float64)
    for i in range(N):
        m = mask[i]
        if m.any():
            feat[i] = LN[i][m].mean(axis=0)
    # fit a small linear head on these realised features (closed form, ridge)
    F = feat
    Y = np.zeros((N, 2), dtype=np.float64)
    Y[np.arange(N), y] = 1.0
    # ridge regression to targets +/-1
    tgt = np.where(y == 1, 1.0, -1.0)
    A = F.T @ F + 1e-3 * np.eye(D)
    b = F.T @ tgt
    wdir = np.linalg.solve(A, b)           # [D]
    # project onto the two head rows
    w["head.weight"] = np.stack([-0.5 * wdir, 0.5 * wdir]).astype(np.float32)
    # bias: choose threshold so that sign matches
    scores = F @ wdir
    # optimal threshold via class means
    thr = 0.5 * (scores[y == 0].mean() + scores[y == 1].mean())
    w["head.bias"] = np.array([-thr, thr], dtype=np.float32)

    # keep everything else identity / zero (blocks, pos, norms)
    for k in ["pos.weight"]:
        w[k] = np.zeros_like(w[k])
    for k in list(w.keys()):
        if k.endswith(".weight") and ("qkv" in k or "proj" in k or "ff1" in k or "ff2" in k):
            w[k] = np.zeros_like(w[k])
        if k.endswith(".bias") and ("qkv" in k or "proj" in k or "ff1" in k or "ff2" in k):
            w[k] = np.zeros_like(w[k])
    for k in ["block0.norm1.weight","block0.norm2.weight","block1.norm1.weight",
              "block1.norm2.weight","norm.weight"]:
        w[k] = np.ones_like(w[k])
    for k in ["block0.norm1.bias","block0.norm2.bias","block1.norm1.bias",
              "block1.norm2.bias","norm.bias"]:
        w[k] = np.zeros_like(w[k])

    # clip for safety
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)

    diag = {
        "bow_dir_norm": float(np.linalg.norm(d)),
        "embed_norm_mean": float(np.linalg.norm(E, axis=1).mean()),
        "head_w_norm": float(np.linalg.norm(w["head.weight"])),
        "head_bias": [float(w["head.bias"][0]), float(w["head.bias"][1])],
        "train_acc_est": float(((F @ wdir) > thr).astype(np.float64).mean()),
    }
    return {"weights": w, "diagnostics": diag}
